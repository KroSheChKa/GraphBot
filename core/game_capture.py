"""
Capture a color screenshot of the Graphwar game field for the approximator UI.
"""

import base64
import time

import cv2
import mss
import numpy as np
import win32con
import win32gui

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

DEFAULT_WINDOW_POSITION = (-7, 0)
SETTLE_SEC = 0.2
GAME_PRECISION = 5


def fmt_game(value):
    return round(float(value), GAME_PRECISION)


def field_to_game(field_x, field_y, field_width, field_height):
    game_x, game_y = pixel_to_game(field_x, field_y, field_width, field_height)
    return fmt_game(game_x), fmt_game(game_y)


def focus_game_window(hwnd):
    if not hwnd:
        return False

    if win32gui.IsIconic(hwnd):
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)

    try:
        win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
        win32gui.BringWindowToTop(hwnd)
        win32gui.SetForegroundWindow(hwnd)
        return True
    except Exception:
        return False


def move_game_window(hwnd, target_x, target_y):
    if not hwnd:
        return False

    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    width = right - left
    height = bottom - top

    if left == target_x and top == target_y:
        return False

    win32gui.MoveWindow(hwnd, target_x, target_y, width, height, True)
    return True


def grab_field_bgr(field):
    with mss.mss() as sct:
        shot = np.array(sct.grab(field))
    return cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)


def encode_png_data_url(bgr):
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("PNG encode failed")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def capture_game_field(
    window_title=DEFAULT_GAME_WINDOW_NAME,
    target_x=DEFAULT_WINDOW_POSITION[0],
    target_y=DEFAULT_WINDOW_POSITION[1],
    margins=None,
    settle_sec=SETTLE_SEC,
):
    """
    Focus Graphwar, move to the corner, grab the configured field region.

    Returns:
        dict with keys ok, image (data URL), width, height, field — or ok=False, error.
    """
    hwnd = find_game_window(window_title)
    if hwnd is None:
        return {"ok": False, "error": f"Window «{window_title}» not found"}

    focus_game_window(hwnd)
    if settle_sec > 0:
        time.sleep(settle_sec)

    move_game_window(hwnd, target_x, target_y)
    if settle_sec > 0:
        time.sleep(settle_sec)

    margins = margins or load_capture_margins()
    try:
        field = get_capture_field(hwnd, margins)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}

    bgr = grab_field_bgr(field)
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
