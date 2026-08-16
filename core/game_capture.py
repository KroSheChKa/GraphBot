"""
Capture a color screenshot of the Graphwar game field for the approximator UI.
"""

import base64
import ctypes

import cv2
import numpy as np
import win32gui
import win32ui

from core.detection import find_active_player, load_active_params, load_players_params
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
        dict with keys ok, image (data URL), width, height, field — or ok=False, error.
    """
    hwnd = find_game_window(window_title)
    if hwnd is None:
        return {"ok": False, "error": f"Window «{window_title}» not found"}

    margins = margins or load_capture_margins()
    try:
        field = get_capture_field(hwnd, margins)
        bgr = crop_client_field(grab_client_bgr(hwnd), margins)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:
        return {"ok": False, "error": f"Could not capture Graphwar quietly: {exc}"}
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
        "field": field,
        "field_archive": field_archive,
        "field_archive_error": field_archive_error,
        "active_norm": active_norm,
        "active_method": active_result.get("method"),
        "active_anchor": active_anchor,
        "forbidden_grid": forbidden_grid,
        "forbidden_stats": forbidden_stats,
        "forbidden_error": forbidden_error,
    }
