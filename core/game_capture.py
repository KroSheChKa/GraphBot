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

from core.window_capture import (
    DEFAULT_GAME_WINDOW_NAME,
    find_game_window,
    get_capture_field,
    load_capture_margins,
)

DEFAULT_WINDOW_POSITION = (-7, 0)
SETTLE_SEC = 0.2


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
        return {"ok": False, "error": f"Окно «{window_title}» не найдено"}

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

    return {
        "ok": True,
        "image": image,
        "width": field["width"],
        "height": field["height"],
        "field": field,
    }
