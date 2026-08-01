"""
Helpers to capture the Graphwar client area via Win32 (no hardcoded screen coords).
"""

import ctypes
import json
from pathlib import Path

import win32gui

DEFAULT_GAME_WINDOW_NAME = "Graphwar"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "capture_config.json"

DEFAULT_MARGINS = {
    "margin_left": 15,
    "margin_top": 15,
    "margin_right": 16,
    "margin_bottom": 135,
}


def _set_process_dpi_awareness():
    """Keep Win32 window coordinates in the same pixel space as mss."""
    try:
        # Match mss: use per-monitor DPI awareness on modern Windows.
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        # Fallback for older Windows versions without shcore.dll.
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


# mss normally sets this when its first capture object is created. Set it before
# the first GetWindowRect/ClientToScreen call so the first screenshot is aligned.
_set_process_dpi_awareness()


def load_capture_margins(path=CONFIG_PATH):
    if not path.exists():
        return DEFAULT_MARGINS.copy()
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    margins = data.get("margins", DEFAULT_MARGINS)
    return {key: int(margins.get(key, DEFAULT_MARGINS[key])) for key in DEFAULT_MARGINS}


def save_capture_margins(margins, path=CONFIG_PATH):
    payload = {"margins": {key: int(margins[key]) for key in DEFAULT_MARGINS}}
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    return path


def find_game_window(title=DEFAULT_GAME_WINDOW_NAME):
    hwnd = win32gui.FindWindow(None, title)
    return hwnd if hwnd else None


def get_client_screen_rect(hwnd):
    """
    Client area of the window in absolute screen coordinates.

    Returns:
        (left, top, width, height)
    """
    left, top = win32gui.ClientToScreen(hwnd, (0, 0))
    _, _, right, bottom = win32gui.GetClientRect(hwnd)
    return left, top, right, bottom


def get_capture_field(hwnd, margins=None):
    """
    Build an mss-compatible region dict from the window client area.

    margins keys (pixels inside the client, not screen):
        margin_left, margin_top, margin_right, margin_bottom
    """
    cx, cy, cw, ch = get_client_screen_rect(hwnd)
    margins = margins or {}

    ml = int(margins.get("margin_left", 0))
    mt = int(margins.get("margin_top", 0))
    mr = int(margins.get("margin_right", 0))
    mb = int(margins.get("margin_bottom", 0))

    width = cw - ml - mr
    height = ch - mt - mb

    if width < 1 or height < 1:
        raise ValueError(
            f"Invalid capture region: client {cw}x{ch}, margins L{ml} T{mt} R{mr} B{mb}"
        )

    return {
        "left": cx + ml,
        "top": cy + mt,
        "width": width,
        "height": height,
    }
