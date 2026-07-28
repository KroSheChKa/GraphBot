"""Interactive configurator for raster forbidden-zone detection.

Usage:
    python tools/calibrate_forbidden_mask.py
    python tools/calibrate_forbidden_mask.py path/to/field.png

The 2x2 dashboard shows source+overlay, raw black pixels, cleaned/safety mask,
and the exact occupancy grid that Dot mode will receive.

Keys:
    s        save config/forbidden_config.json
    d        export the current diagnostic bundle to outputs/
    Space    freeze/unfreeze live capture
    f        toggle player removal
    r        restore defaults
    q / Esc  quit
"""

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import mss
import numpy as np

from core.forbidden_mask import (
    DEFAULT_FORBIDDEN_PARAMS,
    build_forbidden_mask,
    load_forbidden_params,
    sanitize_forbidden_params,
    save_forbidden_params,
)
from core.window_capture import find_game_window, get_capture_field, load_capture_margins

WINDOW = "GraphBot — forbidden mask configurator"
PANEL_WIDTH = 600
PANEL_HEIGHT = 360

TRACKBARS = [
    ("black max", "black_gray_high", 0, 120),
    ("open kernel", "open_kernel", 0, 15),
    ("close kernel", "close_kernel", 0, 21),
    ("close iter", "close_iterations", 0, 6),
    ("min area", "min_component_area", 1, 3000),
    ("player pad", "player_padding", 0, 25),
    ("player filter", "use_player_filter", 0, 1),
    ("safety px", "safety_margin_px", 0, 25),
    ("grid cell px", "grid_cell_px", 1, 12),
    ("cell fill pct", "grid_min_fill_pct", 0, 100),
]


def _on_trackbar(_value):
    pass


def _setup_trackbars(initial):
    for label, key, minimum, maximum in TRACKBARS:
        value = max(minimum, min(maximum, int(initial[key])))
        cv2.createTrackbar(label, WINDOW, value, maximum, _on_trackbar)
        if value < minimum:
            cv2.setTrackbarPos(label, WINDOW, minimum)


def _read_params():
    values = DEFAULT_FORBIDDEN_PARAMS.copy()
    for label, key, minimum, _maximum in TRACKBARS:
        values[key] = max(minimum, cv2.getTrackbarPos(label, WINDOW))
    return sanitize_forbidden_params(values)


def _restore_defaults():
    for label, key, minimum, maximum in TRACKBARS:
        value = max(minimum, min(maximum, int(DEFAULT_FORBIDDEN_PARAMS[key])))
        cv2.setTrackbarPos(label, WINDOW, value)


def _grab_frame(sct, image_path, capture_margins):
    if image_path is not None:
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        return frame

    hwnd = find_game_window()
    if hwnd is None:
        return None
    field = get_capture_field(hwnd, capture_margins)
    shot = np.array(sct.grab(field))
    return cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)


def _add_label(image, label, color=(255, 255, 255)):
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (20, 20, 20), -1)
    cv2.putText(
        out,
        label,
        (9, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        color,
        1,
        cv2.LINE_AA,
    )
    return out


def _fit_panel(image, label, interpolation=cv2.INTER_AREA):
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    resized = cv2.resize(
        image,
        (PANEL_WIDTH, PANEL_HEIGHT),
        interpolation=interpolation,
    )
    return _add_label(resized, label)


def _source_overlay(bgr, result):
    overlay = bgr.copy()
    red = np.zeros_like(overlay)
    red[:, :, 2] = 255
    mask = result["safe_mask"]
    blended = cv2.addWeighted(overlay, 0.62, red, 0.38, 0)
    overlay[mask > 0] = blended[mask > 0]

    contours, _ = cv2.findContours(
        result["clean_mask"],
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(overlay, contours, -1, (0, 255, 80), 1)
    for cx, cy, radius in result["players"]:
        cv2.circle(overlay, (cx, cy), radius, (255, 190, 0), 1)

    stats = result["stats"]
    text = (
        f"areas={stats['components']}  players={stats['players_removed']}  "
        f"grid={stats['grid_forbidden_cells']}/{stats['grid_total_cells']}"
    )
    cv2.putText(
        overlay,
        text,
        (8, overlay.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return overlay


def _safe_mask_preview(result):
    clean = result["clean_mask"]
    safe = result["safe_mask"]
    preview = np.zeros((*safe.shape, 3), dtype=np.uint8)
    preview[safe > 0] = (20, 20, 150)
    preview[clean > 0] = (40, 220, 40)
    return preview


def _grid_preview(result):
    grid = result["grid"].astype(np.uint8) * 255
    preview = np.zeros((*grid.shape, 3), dtype=np.uint8)
    preview[grid > 0] = (30, 30, 255)
    return preview


def build_dashboard(bgr, result, frozen=False):
    source_label = "1. source + final forbidden area"
    if frozen:
        source_label += "  [FROZEN]"
    raw_label = (
        f"2. raw black pixels  <= {result['params']['black_gray_high']} gray"
    )
    safe_label = (
        "3. green=clean shape  red=safety margin "
        f"{result['params']['safety_margin_px']} px"
    )
    grid_label = (
        f"4. exact Dot grid  cell={result['params']['grid_cell_px']} px  "
        f"fill>={result['params']['grid_min_fill_pct']}%"
    )

    top = np.hstack(
        [
            _fit_panel(_source_overlay(bgr, result), source_label),
            _fit_panel(result["raw_mask"], raw_label),
        ]
    )
    bottom = np.hstack(
        [
            _fit_panel(_safe_mask_preview(result), safe_label),
            _fit_panel(
                _grid_preview(result),
                grid_label,
                interpolation=cv2.INTER_NEAREST,
            ),
        ]
    )
    return np.vstack([top, bottom])


def _save_diagnostics(bgr, result, dashboard):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = ROOT_DIR / "outputs" / f"forbidden-mask-{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    cv2.imwrite(str(output_dir / "01-source.png"), bgr)
    cv2.imwrite(str(output_dir / "02-raw-mask.png"), result["raw_mask"])
    cv2.imwrite(str(output_dir / "03-clean-mask.png"), result["clean_mask"])
    cv2.imwrite(str(output_dir / "04-safe-mask.png"), result["safe_mask"])
    cv2.imwrite(str(output_dir / "05-grid.png"), _grid_preview(result))
    cv2.imwrite(str(output_dir / "06-dashboard.png"), dashboard)
    with open(output_dir / "report.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "params": result["params"],
                "stats": result["stats"],
                "components": result["components"],
                "grid": result["grid_payload"],
            },
            file,
            indent=2,
        )
    return output_dir


def _missing_window_frame():
    blank = np.zeros((PANEL_HEIGHT * 2, PANEL_WIDTH * 2, 3), dtype=np.uint8)
    cv2.putText(
        blank,
        "Graphwar window not found",
        (300, 330),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (70, 70, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        blank,
        "Start Graphwar or pass a screenshot path.",
        (340, 375),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (190, 190, 190),
        1,
        cv2.LINE_AA,
    )
    return blank


def main():
    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    capture_margins = load_capture_margins()
    initial = load_forbidden_params()
    frozen_frame = None
    static_frame = None

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, PANEL_WIDTH * 2, PANEL_HEIGHT * 2)
    _setup_trackbars(initial)

    print("Raster forbidden-mask configurator.")
    print("Red overlay is the exact safety mask; panel 4 is the grid sent to Dot mode.")
    print("Overlapping/nested circles may merge: that is correct for collision checking.")
    print("Keys: s=save  d=diagnostic bundle  Space=freeze  f=players  r=reset  q=quit")
    if image_path:
        print(f"Static image: {image_path}")

    with mss.mss() as sct:
        if image_path is not None:
            static_frame = _grab_frame(sct, image_path, capture_margins)

        latest_frame = None
        latest_result = None
        latest_dashboard = None

        while True:
            if static_frame is not None:
                bgr = static_frame
            elif frozen_frame is not None:
                bgr = frozen_frame
            else:
                bgr = _grab_frame(sct, None, capture_margins)

            if bgr is None:
                cv2.imshow(WINDOW, _missing_window_frame())
            else:
                params = _read_params()
                result = build_forbidden_mask(bgr, params=params)
                dashboard = build_dashboard(
                    bgr,
                    result,
                    frozen=frozen_frame is not None,
                )
                cv2.imshow(WINDOW, dashboard)
                latest_frame = bgr
                latest_result = result
                latest_dashboard = dashboard

            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                params = _read_params()
                path = save_forbidden_params(params)
                print(f"Saved: {path}")
                print(params)
            elif key == ord("d") and latest_result is not None:
                path = _save_diagnostics(
                    latest_frame,
                    latest_result,
                    latest_dashboard,
                )
                print(f"Diagnostics: {path}")
            elif key == ord("r"):
                _restore_defaults()
            elif key == ord("f"):
                current = cv2.getTrackbarPos("player filter", WINDOW)
                cv2.setTrackbarPos("player filter", WINDOW, 1 - current)
            elif key == 32 and static_frame is None and latest_frame is not None:
                frozen_frame = None if frozen_frame is not None else latest_frame.copy()
                print("Live capture" if frozen_frame is None else "Frame frozen")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
