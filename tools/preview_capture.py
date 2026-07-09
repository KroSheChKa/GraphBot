"""
Live preview of the Graphwar capture region (OpenCV).

Uses the window client area from Win32 instead of hardcoded screen offsets.
Adjust inner margins with trackbars until the red frame matches the game field.

Controls:
  q / Esc  — quit
  s        — print current field dict to console
  r        — reset margins to 0
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import mss
import numpy as np
import win32gui

from core.window_capture import (
    DEFAULT_GAME_WINDOW_NAME,
    find_game_window,
    get_capture_field,
    get_client_screen_rect,
    load_capture_margins,
    save_capture_margins,
)

WINDOW_NAME = "GraphBot — capture preview"
OVERVIEW_NAME = "GraphBot — full client (crop outline)"
TRACKBAR_MAX = 200


def _on_trackbar(_):
    pass


def _read_margins():
    return {
        "margin_left": cv2.getTrackbarPos("left", WINDOW_NAME),
        "margin_top": cv2.getTrackbarPos("top", WINDOW_NAME),
        "margin_right": cv2.getTrackbarPos("right", WINDOW_NAME),
        "margin_bottom": cv2.getTrackbarPos("bottom", WINDOW_NAME),
    }


def _draw_overlay(frame, field, client_left, client_top, client_w, client_h):
    out = frame.copy()
    rel_x = field["left"] - client_left
    rel_y = field["top"] - client_top
    cv2.rectangle(
        out,
        (rel_x, rel_y),
        (rel_x + field["width"], rel_y + field["height"]),
        (0, 0, 255),
        2,
    )
    cv2.putText(
        out,
        f"client {client_w}x{client_h}",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def _annotate_capture(capture_bgr, field):
    out = capture_bgr.copy()
    h, w = out.shape[:2]
    lines = [
        f"capture: {w}x{h}",
        f"screen left={field['left']} top={field['top']}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(
            out,
            line,
            (8, 22 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return out


def main():
    game_window_name = DEFAULT_GAME_WINDOW_NAME
    if len(sys.argv) > 1:
        game_window_name = sys.argv[1]

    hwnd = find_game_window(game_window_name)
    if hwnd is None:
        print(f"Window '{game_window_name}' not found. Open Graphwar and run again.")
        sys.exit(1)

    title = win32gui.GetWindowText(hwnd)
    print(f"Found window: '{title}' (hwnd={hwnd})")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.namedWindow(OVERVIEW_NAME, cv2.WINDOW_NORMAL)

    saved_margins = load_capture_margins()
    trackbar_keys = ("left", "top", "right", "bottom")
    margin_keys = ("margin_left", "margin_top", "margin_right", "margin_bottom")

    for name, margin_key in zip(trackbar_keys, margin_keys):
        cv2.createTrackbar(name, WINDOW_NAME, saved_margins[margin_key], TRACKBAR_MAX, _on_trackbar)

    legacy_hint = {"margin_left": 21, "margin_top": 52, "margin_right": 0, "margin_bottom": 0}
    print("Trackbars: inner margins from client edges (pixels).")
    print("Loaded margins:", saved_margins)
    print("Legacy hardcoded hint (for comparison):", legacy_hint)
    print("Keys: s = save + print field, r = reset margins, h = legacy hint, q/Esc = quit")

    with mss.mss() as sct:
        while True:
            hwnd = find_game_window(game_window_name)
            if hwnd is None:
                print("Graphwar window lost.")
                break

            client_left, client_top, client_w, client_h = get_client_screen_rect(hwnd)
            margins = _read_margins()

            try:
                field = get_capture_field(hwnd, margins)
            except ValueError as exc:
                blank = np.zeros((120, 640, 3), dtype=np.uint8)
                cv2.putText(
                    blank,
                    str(exc)[:70],
                    (8, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )
                cv2.imshow(WINDOW_NAME, blank)
                key = cv2.waitKey(100) & 0xFF
                if key in (ord("q"), 27):
                    break
                continue

            client_shot = np.array(
                sct.grab(
                    {
                        "left": client_left,
                        "top": client_top,
                        "width": client_w,
                        "height": client_h,
                    }
                )
            )
            client_bgr = cv2.cvtColor(client_shot, cv2.COLOR_BGRA2BGR)
            overview = _draw_overlay(client_bgr, field, client_left, client_top, client_w, client_h)

            capture_shot = np.array(sct.grab(field))
            capture_bgr = cv2.cvtColor(capture_shot, cv2.COLOR_BGRA2BGR)
            preview = _annotate_capture(capture_bgr, field)

            cv2.imshow(OVERVIEW_NAME, overview)
            cv2.imshow(WINDOW_NAME, preview)

            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                config_path = save_capture_margins(margins)
                print("\n--- current field (for mss.grab / GraphBot) ---")
                print(field)
                print("margins:", margins)
                print("client screen rect:", (client_left, client_top, client_w, client_h))
                print(f"saved to: {config_path}")
            if key == ord("r"):
                for name in ("left", "top", "right", "bottom"):
                    cv2.setTrackbarPos(name, WINDOW_NAME, 0)
            if key == ord("h"):
                cv2.setTrackbarPos("left", WINDOW_NAME, legacy_hint["margin_left"])
                cv2.setTrackbarPos("top", WINDOW_NAME, legacy_hint["margin_top"])
                cv2.setTrackbarPos("right", WINDOW_NAME, legacy_hint["margin_right"])
                cv2.setTrackbarPos("bottom", WINDOW_NAME, legacy_hint["margin_bottom"])

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
