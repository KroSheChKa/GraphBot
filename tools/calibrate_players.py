"""
Calibrate ALL player detection (teammates + enemies).

Separate from active_config.json — tuning glow must not break player Hough params.
All players are the same size: use one expected radius + tolerance.

  python tools/calibrate_players.py
  python tools/calibrate_players.py path/to/screenshot.png

Controls:
  s        — save to players_config.json
  r        — reset defaults
  q / Esc  — quit
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import mss
import numpy as np

from core.detection import (
    DEFAULT_PLAYERS_PARAMS,
    draw_players_overlay,
    find_active_player,
    find_all_players,
    load_active_params,
    load_players_params,
    radius_bounds,
    sanitize_players_params,
    save_players_params,
)
from core.window_capture import find_game_window, get_capture_field, load_capture_margins

WINDOW = "GraphBot — all players"
MASK_WINDOW = "player mask"

TRACKBARS = [
    ("gray_lo", "player_gray_low", 0, 255),
    ("gray_hi", "player_gray_high", 0, 255),
    ("gap_lo", "player_gray_gap_low", 0, 255),
    ("gap_hi", "player_gray_gap_high", 0, 255),
    ("p_blur", "player_blur", 3, 41),
    ("hough_p1", "hough_param1", 1, 300),
    ("hough_p2", "hough_param2", 1, 30),
    ("min_dist", "hough_min_dist", 1, 40),
    ("radius", "expected_radius", 4, 20),
    ("radius_tol", "radius_tolerance", 0, 6),
]


def _on_trackbar(_):
    pass


def _read_params():
    params = load_players_params()
    for trackbar_name, param_key, min_val, _ in TRACKBARS:
        params[param_key] = max(min_val, cv2.getTrackbarPos(trackbar_name, WINDOW))
    return sanitize_players_params(params)


def _setup_trackbars(initial):
    for trackbar_name, param_key, min_val, max_val in TRACKBARS:
        start = int(initial.get(param_key, DEFAULT_PLAYERS_PARAMS[param_key]))
        cv2.createTrackbar(trackbar_name, WINDOW, start, max_val, _on_trackbar)
        if start < min_val:
            cv2.setTrackbarPos(trackbar_name, WINDOW, min_val)


def _grab_frame(sct, image_path, capture_margins):
    if image_path:
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        return bgr

    hwnd = find_game_window()
    if hwnd is None:
        return None
    field = get_capture_field(hwnd, capture_margins)
    shot = np.array(sct.grab(field))
    return cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)


def main():
    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    capture_margins = load_capture_margins()
    initial = load_players_params()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.namedWindow(MASK_WINDOW, cv2.WINDOW_NORMAL)
    _setup_trackbars(initial)

    print("Calibrate ALL players (same size for everyone).")
    print("Orange = ours (left side)   Blue = enemy (right side)")
    print("Green = active player (from active_config.json + red glow)")
    print("radius + radius_tol -> Hough min/max radius")
    if image_path:
        print(f"Static image: {image_path}")
    else:
        print("Live capture from Graphwar.")
    print("Keys: s = save players_config.json, r = reset, q/Esc = quit")

    with mss.mss() as sct:
        static_frame = None
        if image_path:
            static_frame = _grab_frame(sct, image_path, capture_margins)

        while True:
            if static_frame is not None:
                bgr = static_frame
            else:
                bgr = _grab_frame(sct, None, capture_margins)
                if bgr is None:
                    blank = np.zeros((200, 640, 3), dtype=np.uint8)
                    cv2.putText(
                        blank,
                        "Graphwar window not found",
                        (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                    )
                    cv2.imshow(WINDOW, blank)
                    key = cv2.waitKey(200) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    continue

            field_width = bgr.shape[1]
            params = _read_params()
            min_r, max_r = radius_bounds(params)

            try:
                result = find_all_players(bgr, field_width, params)
                active_params = load_active_params()
                active_result = find_active_player(bgr, field_width, active_params, params)
            except cv2.error as exc:
                overlay = bgr.copy()
                cv2.putText(
                    overlay,
                    str(exc)[:80],
                    (8, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )
                cv2.imshow(WINDOW, overlay)
                key = cv2.waitKey(30 if static_frame is None else 0) & 0xFF
                if key in (ord("q"), 27):
                    break
                continue

            overlay = draw_players_overlay(
                bgr, result, field_width, active=active_result["active"]
            )
            if active_result["active"]:
                method = active_result["method"]
                cv2.putText(
                    overlay,
                    f"active via {method}",
                    (8, 66),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
            cv2.putText(
                overlay,
                f"Hough radius {min_r}-{max_r}  p2={params['hough_param2']}  blur={params['player_blur']}",
                (8, overlay.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(WINDOW, overlay)
            cv2.imshow(MASK_WINDOW, result["player_mask"])

            key = cv2.waitKey(30 if static_frame is None else 0) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                path = save_players_params(params)
                print(f"Saved: {path}")
                print(params)
                print(f"Hough radius range: {min_r}-{max_r}")
            if key == ord("r"):
                for trackbar_name, param_key, _, max_val in TRACKBARS:
                    default = int(DEFAULT_PLAYERS_PARAMS[param_key])
                    cv2.setTrackbarPos(trackbar_name, WINDOW, min(default, max_val))

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
