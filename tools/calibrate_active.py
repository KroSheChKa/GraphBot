"""
Calibrate active-player detection (red glow + player circle).

Live capture from Graphwar, or a static screenshot:
  python tools/calibrate_active.py
  python tools/calibrate_active.py path/to/screenshot.png

Controls:
  s        — save params to active_config.json
  d        — save the enlarged active-player debug frame
  r        — reset to defaults
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
    DEFAULT_ACTIVE_PARAMS,
    draw_detection_overlay,
    find_active_player,
    load_active_params,
    load_players_params,
    sanitize_active_params,
    save_active_params,
)
from core.window_capture import find_game_window, get_capture_field, load_capture_margins

WINDOW = "GraphBot — active player"
MASK_WINDOW = "red glow mask"
PLAYER_MASK_WINDOW = "player mask"
DEBUG_WINDOW = "active player x10"
DEBUG_SCALE = 10

TRACKBARS = [
    ("red_excess", "red_excess_thresh", 0, 80),
    ("hsv_h1_lo", "hsv_h_low", 0, 180),
    ("hsv_h1_hi", "hsv_h_high", 0, 180),
    ("hsv_s_min", "hsv_s_min", 0, 255),
    ("hsv_v_min", "hsv_v_min", 0, 255),
    ("glow_blur", "glow_blur", 0, 15),
    ("glow_dilate", "glow_dilate", 0, 8),
    ("glow_area", "glow_min_area", 0, 500),
    ("match_dist", "match_max_dist", 5, 120),
    ("left_only", "left_side_only", 0, 1),
]


def _on_trackbar(_):
    pass


def _read_params():
    params = load_active_params()
    for trackbar_name, param_key, min_val, _ in TRACKBARS:
        value = cv2.getTrackbarPos(trackbar_name, WINDOW)
        params[param_key] = max(min_val, value)
    return sanitize_active_params(params)


def _setup_trackbars(initial):
    for trackbar_name, param_key, min_val, max_val in TRACKBARS:
        start = int(initial.get(param_key, DEFAULT_ACTIVE_PARAMS[param_key]))
        cv2.createTrackbar(trackbar_name, WINDOW, start, max_val, _on_trackbar)
        if start < min_val:
            cv2.setTrackbarPos(trackbar_name, WINDOW, min_val)


def _active_debug_frame(bgr, result, scale=DEBUG_SCALE):
    active = result.get("active")
    if not active:
        return np.zeros((240, 420, 3), dtype=np.uint8)

    cx, cy, radius = (float(value) for value in active)
    padding = max(12, int(round(radius * 3.5)))
    height, width = bgr.shape[:2]
    left = max(0, int(np.floor(cx - padding)))
    top = max(0, int(np.floor(cy - padding)))
    right = min(width, int(np.ceil(cx + padding + 1)))
    bottom = min(height, int(np.ceil(cy + padding + 1)))

    crop = bgr[top:bottom, left:right].copy()
    mask = result["glow_mask"][top:bottom, left:right]
    tint = np.zeros_like(crop)
    tint[:, :, 2] = mask
    crop = cv2.addWeighted(crop, 1.0, tint, 0.45, 0)
    enlarged = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    center = (int(round((cx - left) * scale)), int(round((cy - top) * scale)))
    cv2.circle(enlarged, center, int(round(radius * scale)), (0, 255, 0), 2)
    cv2.drawMarker(enlarged, center, (0, 255, 0), cv2.MARKER_CROSS, 18, 2)

    detection = result.get("active_detection") or {}
    source_center = detection.get("source_center")
    if source_center:
        sx, sy = source_center
        source = (int(round((sx - left) * scale)), int(round((sy - top) * scale)))
        cv2.drawMarker(enlarged, source, (0, 255, 255), cv2.MARKER_CROSS, 18, 2)

    ring_center = detection.get("ring_center")
    if ring_center:
        rx, ry = ring_center
        ring = (int(round((rx - left) * scale)), int(round((ry - top) * scale)))
        cv2.drawMarker(enlarged, ring, (255, 0, 255), cv2.MARKER_TILTED_CROSS, 18, 2)

    glow_center = result.get("glow_center")
    if glow_center:
        gx, gy = glow_center
        glow = (int(round((gx - left) * scale)), int(round((gy - top) * scale)))
        cv2.drawMarker(enlarged, glow, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 18, 2)

    confidence = float(detection.get("confidence", 0.0))
    uncertainty = detection.get("uncertainty_game", {}).get("x")
    label = f"green refined / yellow Hough / magenta ring  conf={confidence:.2f} err=±{float(uncertainty or 0):.3f}"
    cv2.putText(
        enlarged,
        label,
        (8, max(22, enlarged.shape[0] - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    return enlarged


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
    initial = load_active_params()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.namedWindow(MASK_WINDOW, cv2.WINDOW_NORMAL)
    cv2.namedWindow(PLAYER_MASK_WINDOW, cv2.WINDOW_NORMAL)
    cv2.namedWindow(DEBUG_WINDOW, cv2.WINDOW_NORMAL)
    _setup_trackbars(initial)

    print("Calibrate ACTIVE player (red glow on the left side).")
    print("Green circle = detected active. Red tint = glow mask.")
    print("Vertical green line = game x=0 (center axis).")
    if image_path:
        print(f"Static image: {image_path}")
    else:
        print("Live capture from Graphwar.")
    print("Keys: s = save active only, d = save enlarged debug, r = reset, q/Esc = quit")

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
            try:
                result = find_active_player(bgr, field_width, params, players_params)
            except cv2.error as exc:
                overlay = bgr.copy()
                cv2.putText(
                    overlay,
                    f"OpenCV: {exc}",
                    (8, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imshow(WINDOW, overlay)
                key = cv2.waitKey(30 if static_frame is None else 0) & 0xFF
                if key in (ord("q"), 27):
                    break
                continue
            overlay = draw_detection_overlay(bgr, result, field_width)
            debug_frame = _active_debug_frame(bgr, result)

            info = f"method={result['method']} glow_area={result['glow_area']}"
            if result["active"]:
                cx, cy, r = result["active"]
                info += f"  active=({cx},{cy}) r={r}"
                detection = result.get("active_detection") or {}
                uncertainty = detection.get("uncertainty_game", {}).get("x")
                info += (
                    f"  confidence={detection.get('confidence', 0):.2f}"
                    f"  error=±{float(uncertainty or 0):.3f}"
                )
            cv2.putText(
                overlay,
                info,
                (8, overlay.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(WINDOW, overlay)
            cv2.imshow(MASK_WINDOW, result["glow_mask"])
            cv2.imshow(PLAYER_MASK_WINDOW, result["player_mask"])
            cv2.imshow(DEBUG_WINDOW, debug_frame)

            key = cv2.waitKey(30 if static_frame is None else 0) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                path = save_active_params(params)
                print(f"Saved: {path}")
                print(params)
            if key == ord("d"):
                output_path = ROOT_DIR / "outputs" / "active_debug.png"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(output_path), debug_frame)
                print(f"Saved: {output_path}")
            if key == ord("r"):
                for trackbar_name, param_key, _, max_val in TRACKBARS:
                    default = int(DEFAULT_ACTIVE_PARAMS[param_key])
                    cv2.setTrackbarPos(trackbar_name, WINDOW, min(default, max_val))

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
