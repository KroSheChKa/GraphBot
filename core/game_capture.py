"""
Capture a color screenshot of the Graphwar game field for the approximator UI.
"""

import base64
import ctypes
import json
from pathlib import Path

import cv2
import numpy as np
import win32gui
import win32ui

from core.detection import find_active_player, find_all_players, load_active_params, load_players_params
from core.forbidden_mask import build_forbidden_mask, load_forbidden_params
from core.field_geometry import pixel_to_game
from core.field_capture_archive import save_clean_field_capture
from core.window_capture import (
    DEFAULT_GAME_WINDOW_NAME,
    find_game_window,
    get_capture_field,
    load_capture_margins,
)

GAME_PRECISION = 5
PW_CLIENTONLY = 0x00000001

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRID_CAPTURE_CONFIG_PATH = PROJECT_ROOT / "config" / "capture_config.json"

# The approximator UI renders a 1000x600 canvas (5:3) covering game space
# x=-25..25, y=-15..15, i.e. 20 px per game unit on both axes (isotropic).
# The captured field must match this exactly so cell dimensions and the
# (0,0) origin align with the target UI grid.
TARGET_WIDTH = 1000
TARGET_HEIGHT = 600

# Natural output sizing caps (pixels). The captured graph is sent to the UI at
# its native auto-cropped resolution so it fills the canvas without scaling or
# truncation. These bounds only engage for pathological dimensions, using
# isotropic scaling that preserves all graph content (no cropping, no padding).
NATURAL_MIN_WIDTH = 400
NATURAL_MIN_HEIGHT = 240
NATURAL_MAX_WIDTH = 1600
NATURAL_MAX_HEIGHT = 960

DEFAULT_GRID_CAPTURE_PARAMS = {
    "target_width": TARGET_WIDTH,
    "target_height": TARGET_HEIGHT,
    "white_thresh": 235,
    "open_kernel": 5,
    "min_component": 1000,
}


def load_grid_capture_params(path=GRID_CAPTURE_CONFIG_PATH):
    params = DEFAULT_GRID_CAPTURE_PARAMS.copy()
    try:
        if path.exists():
            with open(path, encoding="utf-8") as file:
                data = json.load(file)
            override = data.get("grid_capture", {})
            for key in DEFAULT_GRID_CAPTURE_PARAMS:
                if key in override:
                    params[key] = int(override[key])
    except Exception:
        pass
    return params


def auto_crop_white_grid(bgr, params=None):
    """Crop to the white graph playing field, excluding all UI elements.

    Uses morphological dilation to bridge gaps between white graph regions
    separated by black obstacle circles, identifies the largest connected
    component (the graph itself) on the dilated mask, then computes the tight
    bounding box from original white pixels within that region. This excludes
    stray white UI text/elements outside the graph while capturing the full
    graph extent even when obstacles split the white area into multiple parts.
    Returns (cropped_bgr, box) where box is (x, y, w, h) in input pixels,
    or (bgr.copy(), None) when no usable white region is found.
    """
    p = {**DEFAULT_GRID_CAPTURE_PARAMS, **(params or {})}
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, int(p["white_thresh"]), 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (int(p["open_kernel"]), int(p["open_kernel"]))
    )
    # Dilate to connect white graph regions split by black obstacle circles.
    dilated = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=2)
    # Remove small noise specks; keep only substantial white structures.
    dilated = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, kernel)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, 8)
    best = None
    best_area = 0
    for label in range(1, num):
        x, y, w, h, area = (int(value) for value in stats[label])
        if area < int(p["min_component"]):
            continue
        if w < 40 or h < 40:
            continue
        if area > best_area:
            best_area = area
            best = (x, y, w, h)
    if best is None:
        return bgr.copy(), None
    dx, dy, dw, dh = best
    # Tighten to the actual white pixels of the original mask within the
    # dilated component's region, avoiding dilation-driven expansion.
    region_mask = mask[dy : dy + dh, dx : dx + dw]
    white_points = np.column_stack(np.where(region_mask == 255))
    if white_points.size == 0:
        return bgr.copy(), None
    ry_coords, rx_coords = white_points[:, 0], white_points[:, 1]
    x0 = dx + int(rx_coords.min())
    x1 = dx + int(rx_coords.max())
    y0 = dy + int(ry_coords.min())
    y1 = dy + int(ry_coords.max())
    w = x1 - x0 + 1
    h = y1 - y0 + 1
    if w < 40 or h < 40:
        return bgr.copy(), None
    height, width = bgr.shape[:2]
    if w * h < width * height * 0.05:
        return bgr.copy(), None
    box = (x0, y0, w, h)
    return bgr[y0 : y0 + h, x0 : x0 + w].copy(), box


def resize_to_grid(bgr, target_width, target_height):
    """Resize isotropically to fill the target canvas, center-cropping overflow.

    Preserves cell dimensions (no anisotropic distortion) and keeps the grid
    centered so the game origin (0,0) remains at the canvas center, matching
    the target UI grid. The graph fills the entire canvas with no padding;
    any overflow beyond the target dimensions is cropped symmetrically from
    the center so the origin stays aligned.
    """
    height, width = bgr.shape[:2]
    scale = max(target_width / width, target_height / height)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(bgr, (new_w, new_h), interpolation=interp)
    canvas = np.full((target_height, target_width, 3), 255, dtype=np.uint8)
    top = (target_height - new_h) // 2
    left = (target_width - new_w) // 2
    src_top = max(0, -top)
    src_left = max(0, -left)
    dst_top = max(0, top)
    dst_left = max(0, left)
    copy_h = min(new_h - src_top, target_height - dst_top)
    copy_w = min(new_w - src_left, target_width - dst_left)
    if copy_h > 0 and copy_w > 0:
        canvas[dst_top : dst_top + copy_h, dst_left : dst_left + copy_w] = resized[
            src_top : src_top + copy_h, src_left : src_left + copy_w
        ]
    return canvas


def apply_size_caps(bgr, min_w=NATURAL_MIN_WIDTH, min_h=NATURAL_MIN_HEIGHT,
                    max_w=NATURAL_MAX_WIDTH, max_h=NATURAL_MAX_HEIGHT):
    """Isotropically resize to stay within size caps, preserving all content.

    Scales the image so both dimensions fall within [min_w, min_h] x
    [max_w, max_h] without cropping or padding any graph content.  Returns
    the (possibly resized) image.
    """
    height, width = bgr.shape[:2]
    scale = 1.0
    if width > max_w or height > max_h:
        scale = min(scale, max_w / width, max_h / height)
    if width < min_w or height < min_h:
        scale = max(scale, min_w / width, min_h / height)
    if scale == 1.0:
        return bgr
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(bgr, (new_w, new_h), interpolation=interp)


def _print_window(hwnd, device_context):
    """Call PrintWindow from user32 (not exported by every pywin32 build)."""
    user32 = ctypes.windll.user32
    user32.PrintWindow.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
    user32.PrintWindow.restype = ctypes.c_bool
    return bool(user32.PrintWindow(hwnd, device_context, PW_CLIENTONLY))


def fmt_game(value):
    return round(float(value), GAME_PRECISION)


def field_to_game(field_x, field_y, field_width, field_height):
    game_x, game_y = pixel_to_game(field_x, field_y, field_width, field_height)
    return fmt_game(game_x), fmt_game(game_y)


def grab_client_bgr(hwnd):
    """Render a window's client area off-screen using Win32 ``PrintWindow``.

    Unlike a desktop grab, this does not activate, move, or uncover Graphwar.
    It also means another application's pixels can never accidentally become a
    Graphwar field capture.  Some applications decline ``PrintWindow``; callers
    receive an error in that case rather than a misleading desktop screenshot.
    """
    _, _, width, height = win32gui.GetClientRect(hwnd)
    if width < 1 or height < 1:
        raise RuntimeError("Game window has no drawable client area")

    window_dc_handle = None
    window_dc = None
    memory_dc = None
    bitmap = None
    previous_bitmap = None
    try:
        window_dc_handle = win32gui.GetWindowDC(hwnd)
        window_dc = win32ui.CreateDCFromHandle(window_dc_handle)
        memory_dc = window_dc.CreateCompatibleDC()
        bitmap = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(window_dc, width, height)
        previous_bitmap = memory_dc.SelectObject(bitmap)

        # PW_CLIENTONLY: bitmap coordinates are exactly GetClientRect(), so
        # configured capture margins remain client-relative.
        if not _print_window(hwnd, memory_dc.GetSafeHdc()):
            raise RuntimeError("Windows could not render the Graphwar window off-screen")

        bgra = np.frombuffer(bitmap.GetBitmapBits(True), dtype=np.uint8)
        bgra = bgra.reshape((height, width, 4))
        return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
    finally:
        if memory_dc is not None and previous_bitmap is not None:
            memory_dc.SelectObject(previous_bitmap)
        if bitmap is not None:
            win32gui.DeleteObject(bitmap.GetHandle())
        if memory_dc is not None:
            memory_dc.DeleteDC()
        if window_dc is not None:
            window_dc.DeleteDC()
        if window_dc_handle is not None:
            win32gui.ReleaseDC(hwnd, window_dc_handle)


def crop_client_field(client_bgr, margins):
    """Crop configured field margins from a client-area screenshot."""
    height, width = client_bgr.shape[:2]
    ml = int(margins.get("margin_left", 0))
    mt = int(margins.get("margin_top", 0))
    mr = int(margins.get("margin_right", 0))
    mb = int(margins.get("margin_bottom", 0))
    field_width = width - ml - mr
    field_height = height - mt - mb
    if field_width < 1 or field_height < 1:
        raise ValueError(
            f"Invalid capture region: client {width}x{height}, margins L{ml} T{mt} R{mr} B{mb}"
        )
    return client_bgr[mt : mt + field_height, ml : ml + field_width].copy()


def encode_png_data_url(bgr):
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("PNG encode failed")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def capture_game_field(
    window_title=DEFAULT_GAME_WINDOW_NAME,
    target_x=None,
    target_y=None,
    margins=None,
    settle_sec=None,
):
    """
    Capture the configured Graphwar field without changing the user's desktop.

    ``target_x``, ``target_y``, and ``settle_sec`` are retained as ignored
    compatibility arguments for callers from older versions.  The capture is
    rendered directly from the Graphwar window, not from desktop pixels.

    Returns:
        dict with keys ok, image (data URL), width, height, raw_width,
        raw_height, field — or ok=False, error.  ``width``/``height`` are the
        native auto-cropped graph dimensions (clamped to size caps); the UI
        sizes its canvas to match them rather than scaling to a fixed canvas.
    """
    hwnd = find_game_window(window_title)
    if hwnd is None:
        return {"ok": False, "error": f"Window «{window_title}» not found"}

    margins = margins or load_capture_margins()
    grid_params = load_grid_capture_params()
    try:
        field = get_capture_field(hwnd, margins)
        client_bgr = grab_client_bgr(hwnd)
        field_bgr = crop_client_field(client_bgr, margins)
        # Auto-crop to the white grid playing area, removing peripheral UI.
        grid_bgr, _crop_box = auto_crop_white_grid(field_bgr, grid_params)
        # Send the graph at its native resolution, clamped to safe size caps,
        # so it fills the UI canvas without scaling or truncation.
        bgr = apply_size_caps(grid_bgr)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:
        return {"ok": False, "error": f"Could not capture Graphwar quietly: {exc}"}

    width, height = bgr.shape[1], bgr.shape[0]
    field = {
        "left": int(field.get("left", 0)),
        "top": int(field.get("top", 0)),
        "width": width,
        "height": height,
    }
    try:
        image = encode_png_data_url(bgr)
    except RuntimeError as exc:
        return {"ok": False, "error": str(exc)}

    field_archive = None
    field_archive_error = None
    try:
        # Save the exact same raw crop that is sent to the browser. All
        # detection and visualization happens after this point.
        field_archive = save_clean_field_capture(bgr)
    except Exception as exc:
        # An archive write must never make a usable field capture fail.
        field_archive_error = str(exc)

    active_result = find_active_player(
        bgr,
        field["width"],
        active_params=load_active_params(),
        players_params=load_players_params(),
    )
    active_norm = None
    active_anchor = None
    active_circle = active_result.get("active")
    if active_circle is not None:
        cx, cy, radius = active_circle
        gx, gy = field_to_game(cx, cy, field["width"], field["height"])
        active_norm = [gx, gy]
        detection = active_result.get("active_detection") or {}
        uncertainty_px = float(detection.get("uncertainty_px", 1.0))
        scale_x = 50.0 / field["width"]
        scale_y = 30.0 / field["height"]
        active_anchor = {
            "pixel": {
                "x": round(float(cx), 4),
                "y": round(float(cy), 4),
                "radius": round(float(radius), 4),
            },
            "game": {"x": gx, "y": gy},
            "confidence": round(float(detection.get("confidence", 0.0)), 4),
            "uncertainty_px": round(uncertainty_px, 4),
            "uncertainty_game": {
                "x": round(uncertainty_px * scale_x, 5),
                "y": round(uncertainty_px * scale_y, 5),
            },
            "method": detection.get("method", active_result.get("method", "unknown")),
            "needs_review": bool(detection.get("needs_review", True)),
        }

    detected_enemies = []
    try:
        enemy_result = find_all_players(bgr, field["width"])
        enemies = enemy_result.get("enemies", [])
        for cx, cy, radius in sorted(enemies, key=lambda item: item[0]):
            gx, gy = field_to_game(cx, cy, field["width"], field["height"])
            detected_enemies.append(
                {
                    "pixel": {
                        "x": round(float(cx), 4),
                        "y": round(float(cy), 4),
                        "radius": round(float(radius), 4),
                    },
                    "game": {"x": gx, "y": gy},
                }
            )
    except Exception:
        detected_enemies = []

    forbidden_grid = None
    forbidden_stats = None
    forbidden_error = None
    try:
        forbidden_result = build_forbidden_mask(
            bgr,
            params=load_forbidden_params(),
            players=active_result.get("players"),
        )
        forbidden_grid = forbidden_result["grid_payload"]
        forbidden_stats = forbidden_result["stats"]
    except Exception as exc:
        # Field capture remains usable even if a newly tuned mask is invalid.
        forbidden_error = str(exc)

    return {
        "ok": True,
        "image": image,
        "width": field["width"],
        "height": field["height"],
        "raw_width": field["width"],
        "raw_height": field["height"],
        "field": field,
        "field_archive": field_archive,
        "field_archive_error": field_archive_error,
        "active_norm": active_norm,
        "active_method": active_result.get("method"),
        "active_anchor": active_anchor,
        "detected_enemies": detected_enemies,
        "forbidden_grid": forbidden_grid,
        "forbidden_stats": forbidden_stats,
        "forbidden_error": forbidden_error,
    }
