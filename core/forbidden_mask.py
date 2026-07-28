"""Raster forbidden-zone extraction for the web UI and Dot mode.

The detector intentionally does not reconstruct circles. Black pixels are the
source of truth; morphology removes thin graph/UI strokes, connected components
remove specks, and a safety dilation turns the result into a collision mask.
The mask is compressed to row runs for inexpensive transfer to JavaScript.
"""

import json
import math
from pathlib import Path

import cv2
import numpy as np

from core.detection import (
    detect_player_circles,
    load_players_params,
    sanitize_players_params,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "forbidden_config.json"

DEFAULT_FORBIDDEN_PARAMS = {
    "black_gray_high": 9,
    "open_kernel": 5,
    "close_kernel": 0,
    "close_iterations": 1,
    "min_component_area": 29,
    "player_padding": 1,
    "use_player_filter": 1,
    "safety_margin_px": 4,
    "grid_cell_px": 3,
    "grid_min_fill_pct": 0,
}


def _players_as_tuples(players):
    if players is None:
        return []
    return [
        (int(point[0]), int(point[1]), int(point[2]))
        for point in players[0]
    ]


def _odd_kernel(value, allow_zero=True):
    value = max(0 if allow_zero else 1, int(value))
    if value == 0 and allow_zero:
        return 0
    return value if value % 2 == 1 else value + 1


def sanitize_forbidden_params(params):
    p = {**DEFAULT_FORBIDDEN_PARAMS, **(params or {})}
    p["black_gray_high"] = max(0, min(120, int(p["black_gray_high"])))
    p["open_kernel"] = min(21, _odd_kernel(p["open_kernel"]))
    p["close_kernel"] = min(31, _odd_kernel(p["close_kernel"]))
    p["close_iterations"] = max(0, min(8, int(p["close_iterations"])))
    p["min_component_area"] = max(1, int(p["min_component_area"]))
    p["player_padding"] = max(0, min(30, int(p["player_padding"])))
    p["use_player_filter"] = 1 if int(p["use_player_filter"]) else 0
    p["safety_margin_px"] = max(0, min(40, int(p["safety_margin_px"])))
    p["grid_cell_px"] = max(1, min(20, int(p["grid_cell_px"])))
    p["grid_min_fill_pct"] = max(0, min(100, int(p["grid_min_fill_pct"])))
    return p


def load_forbidden_params(path=CONFIG_PATH):
    if not path.exists():
        return DEFAULT_FORBIDDEN_PARAMS.copy()
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    return sanitize_forbidden_params(data.get("forbidden", {}))


def save_forbidden_params(params, path=CONFIG_PATH):
    clean = sanitize_forbidden_params(params)
    payload = {"forbidden": {key: clean[key] for key in DEFAULT_FORBIDDEN_PARAMS}}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    return path


def _remove_player_regions(mask, players, padding):
    if players is None:
        return mask
    cleaned = mask.copy()
    for cx, cy, radius in _players_as_tuples(players):
        cv2.circle(
            cleaned,
            (cx, cy),
            max(1, int(radius) + int(padding)),
            0,
            -1,
        )
    return cleaned


def _morphology(mask, params):
    result = mask
    open_size = int(params["open_kernel"])
    if open_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)

    close_size = int(params["close_kernel"])
    close_iterations = int(params["close_iterations"])
    if close_size > 0 and close_iterations > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        result = cv2.morphologyEx(
            result,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=close_iterations,
        )
    return result


def _filter_components(mask, min_area):
    label_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )
    cleaned = np.zeros_like(mask)
    components = []
    for label in range(1, label_count):
        x, y, width, height, area = (int(value) for value in stats[label])
        if area < min_area:
            continue
        cleaned[labels == label] = 255
        components.append(
            {
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "area": area,
            }
        )
    return cleaned, components


def _dilate_safety(mask, margin_px):
    if margin_px <= 0:
        return mask.copy()
    size = margin_px * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(mask, kernel, iterations=1)


def mask_to_occupancy_grid(mask, cell_px, min_fill_pct=0):
    """Downsample a mask without losing isolated forbidden pixels by default."""
    height, width = mask.shape[:2]
    rows = math.ceil(height / cell_px)
    cols = math.ceil(width / cell_px)
    padded = np.zeros((rows * cell_px, cols * cell_px), dtype=np.uint8)
    padded[:height, :width] = mask
    blocks = padded.reshape(rows, cell_px, cols, cell_px)
    occupied_pixels = np.count_nonzero(blocks, axis=(1, 3))
    if min_fill_pct <= 0:
        grid = occupied_pixels > 0
    else:
        fill_pct = occupied_pixels * 100.0 / (cell_px * cell_px)
        grid = fill_pct >= min_fill_pct
    return grid


def encode_grid_runs(grid, image_width, image_height, cell_px):
    rows_rle = []
    for row in grid:
        runs = []
        start = None
        for index, occupied in enumerate(row):
            if occupied and start is None:
                start = index
            if start is not None and (not occupied or index == len(row) - 1):
                end = index + 1 if occupied and index == len(row) - 1 else index
                runs.append([start, end - start])
                start = None
        rows_rle.append(runs)

    return {
        "version": 1,
        "cols": int(grid.shape[1]),
        "rows": int(grid.shape[0]),
        "cell_px": int(cell_px),
        "image_width": int(image_width),
        "image_height": int(image_height),
        "x_min": -25,
        "x_max": 25,
        "y_min": -15,
        "y_max": 15,
        "rows_rle": rows_rle,
    }


def build_forbidden_mask(
    bgr,
    params=None,
    players=None,
    players_params=None,
):
    """Build raw, cleaned, safety and grid representations for one field."""
    params = sanitize_forbidden_params(params or load_forbidden_params())
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    raw_mask = cv2.inRange(gray, 0, int(params["black_gray_high"]))

    if params["use_player_filter"]:
        if players is None:
            players_params = sanitize_players_params(players_params or load_players_params())
            players, _ = detect_player_circles(gray, players_params)
        without_players = _remove_player_regions(
            raw_mask,
            players,
            params["player_padding"],
        )
    else:
        players = None
        without_players = raw_mask.copy()

    morphological = _morphology(without_players, params)
    clean_mask, components = _filter_components(
        morphological,
        int(params["min_component_area"]),
    )
    safe_mask = _dilate_safety(clean_mask, int(params["safety_margin_px"]))
    grid = mask_to_occupancy_grid(
        safe_mask,
        int(params["grid_cell_px"]),
        int(params["grid_min_fill_pct"]),
    )
    height, width = gray.shape[:2]
    grid_payload = encode_grid_runs(
        grid,
        width,
        height,
        int(params["grid_cell_px"]),
    )
    player_count = len(_players_as_tuples(players))
    stats = {
        "raw_pixels": int(np.count_nonzero(raw_mask)),
        "clean_pixels": int(np.count_nonzero(clean_mask)),
        "safe_pixels": int(np.count_nonzero(safe_mask)),
        "components": len(components),
        "players_removed": player_count if params["use_player_filter"] else 0,
        "grid_forbidden_cells": int(np.count_nonzero(grid)),
        "grid_total_cells": int(grid.size),
    }

    return {
        "raw_mask": raw_mask,
        "without_players_mask": without_players,
        "clean_mask": clean_mask,
        "safe_mask": safe_mask,
        "grid": grid,
        "grid_payload": grid_payload,
        "components": components,
        "players": _players_as_tuples(players),
        "stats": stats,
        "params": params,
    }
