"""
Calibrate black obstacle (sphere) detection.

Black circles vary in size and can be nested. Player-sized dark blobs are
filtered out using players_config.json + inner brightness check.

  python tools/calibrate_obstacles.py
  python tools/calibrate_obstacles.py path/to/screenshot.png

Controls:
  s        — save to obstacles_config.json
  r        — reset defaults
  p        — toggle player overlay
  j        — toggle rejected candidates
  f        — toggle player filter (on/off)
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
    DEFAULT_OBSTACLES_PARAMS,
    draw_obstacles_overlay,
    find_all_obstacles,
    load_obstacles_params,
    load_players_params,
    sanitize_obstacles_params,
    save_obstacles_params,
)
from core.window_capture import find_game_window, get_capture_field, load_capture_margins

WINDOW = "GraphBot — black obstacles"
MASK_WINDOW = "obstacle hough mask"
DARK_WINDOW = "dark pixel mask"

TRACKBARS = [
    ("method", "detect_method", 0, 1),
    ("black_lo", "black_gray_low", 0, 255),
    ("black_hi", "black_gray_high", 0, 255),
    ("min_area", "min_blob_area", 1, 400),
    ("min_circ", "min_circularity", 0, 100),
    ("blob_close", "blob_close", 0, 8),
    ("blur", "black_blur", 3, 41),
    ("hough_p1", "hough_param1", 1, 300),
    ("hough_p2", "hough_param2", 1, 50),
    ("min_dist", "hough_min_dist", 1, 40),
    ("min_r", "min_radius", 1, 80),
    ("max_r", "max_radius", 2, 200),
    ("nest_sup", "suppress_nested", 0, 1),
    ("inner_max", "inner_gray_max", 0, 120),
    ("inner_marg", "inner_edge_margin", 0, 60),
    ("overlap", "player_overlap_reject", 0, 100),
    ("plr_dist", "player_center_dist", 0, 40),
    ("dedupe_d", "dedupe_center_dist", 0, 15),
    ("dedupe_r", "dedupe_radius_diff", 0, 10),
    ("plr_filter", "use_player_filter", 0, 1),
]


def _on_trackbar(_):
    pass


def _read_params():
    params = load_obstacles_params()
    for trackbar_name, param_key, min_val, _ in TRACKBARS:
        params[param_key] = max(min_val, cv2.getTrackbarPos(trackbar_name, WINDOW))
    return sanitize_obstacles_params(params)


def _setup_trackbars(initial):
    for trackbar_name, param_key, min_val, max_val in TRACKBARS:
        start = int(initial.get(param_key, DEFAULT_OBSTACLES_PARAMS[param_key]))
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
    initial = load_obstacles_params()
    show_players = True
    show_rejected = False

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.namedWindow(MASK_WINDOW, cv2.WINDOW_NORMAL)
    cv2.namedWindow(DARK_WINDOW, cv2.WINDOW_NORMAL)
    _setup_trackbars(initial)

    print("Calibrate BLACK obstacles (any size, nested OK).")
    print("method=1 blobs (1 circle per dark blob, recommended)")
    print("method=0 hough + nest_sup removes inner hits on same blob")
    print("Orange outline = players from players_config.json (for comparison).")
    print("Colored circles = accepted obstacles.")
    print("inner_max + inner_marg — reject if inside is not dark enough.")
    print("overlap / plr_dist — reject if looks like a player.")
    if image_path:
        print(f"Static image: {image_path}")
    else:
        print("Live capture from Graphwar.")
    print("Keys: s=save  r=reset  p=players  j=rejected  f=player filter  q=quit")

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
            players_params = load_players_params()
            filter_players = bool(params["use_player_filter"])

            try:
                result = find_all_obstacles(
                    bgr,
                    obstacles_params=params,
                    players_params=players_params,
                    filter_players=filter_players,
                )
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

            overlay = draw_obstacles_overlay(
                bgr,
                result,
                field_width=field_width,
                show_players=show_players,
                show_rejected=show_rejected,
            )
            filter_label = "ON" if filter_players else "OFF"
            mode_label = "blobs" if params["detect_method"] else "hough"
            cv2.putText(
                overlay,
                f"mode={mode_label}  player filter={filter_label}  radius {params['min_radius']}-{params['max_radius']}",
                (8, overlay.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(WINDOW, overlay)
            cv2.imshow(MASK_WINDOW, result["obstacle_mask"])
            cv2.imshow(DARK_WINDOW, result["dark_mask"])

            key = cv2.waitKey(30 if static_frame is None else 0) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                path = save_obstacles_params(params)
                print(f"Saved: {path}")
                print(params)
                print(f"Accepted obstacles: {len(result['obstacles'])}")
            if key == ord("r"):
                for trackbar_name, param_key, _, max_val in TRACKBARS:
                    default = int(DEFAULT_OBSTACLES_PARAMS[param_key])
                    cv2.setTrackbarPos(trackbar_name, WINDOW, min(default, max_val))
            if key == ord("p"):
                show_players = not show_players
                print(f"Player overlay: {'on' if show_players else 'off'}")
            if key == ord("j"):
                show_rejected = not show_rejected
                print(f"Rejected overlay: {'on' if show_rejected else 'off'}")
            if key == ord("f"):
                current = cv2.getTrackbarPos("plr_filter", WINDOW)
                cv2.setTrackbarPos("plr_filter", WINDOW, 1 - current)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
