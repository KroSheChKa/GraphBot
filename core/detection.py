"""
Player detection helpers for GraphBot.

Configs are split:
  active_config.json  — red glow / active player matching
  players_config.json — all player circles (same size for everyone)
"""

import json
import math
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
ACTIVE_CONFIG_PATH = CONFIG_DIR / "active_config.json"
PLAYERS_CONFIG_PATH = CONFIG_DIR / "players_config.json"
OBSTACLES_CONFIG_PATH = CONFIG_DIR / "obstacles_config.json"

DEFAULT_ACTIVE_PARAMS = {
    "red_excess_thresh": 18,
    "hsv_h_low": 0,
    "hsv_h_high": 12,
    "hsv_h_low2": 168,
    "hsv_h_high2": 180,
    "hsv_s_min": 30,
    "hsv_v_min": 40,
    "glow_blur": 5,
    "glow_dilate": 2,
    "glow_min_area": 80,
    "match_max_dist": 40,
    "left_side_only": 1,
}

DEFAULT_PLAYERS_PARAMS = {
    "player_gray_low": 50,
    "player_gray_high": 250,
    "player_gray_gap_low": 169,
    "player_gray_gap_high": 171,
    "player_blur": 23,
    "hough_param1": 150,
    "hough_param2": 10,
    "hough_min_dist": 10,
    "expected_radius": 8,
    "radius_tolerance": 1,
}

DEFAULT_OBSTACLES_PARAMS = {
    "detect_method": 1,
    "black_gray_low": 0,
    "black_gray_high": 25,
    "black_blur": 13,
    "hough_param1": 50,
    "hough_param2": 15,
    "hough_min_dist": 8,
    "min_radius": 2,
    "max_radius": 120,
    "min_blob_area": 40,
    "min_circularity": 55,
    "blob_close": 2,
    "suppress_nested": 1,
    "inner_gray_max": 45,
    "inner_edge_margin": 25,
    "player_overlap_reject": 35,
    "player_center_dist": 10,
    "dedupe_center_dist": 4,
    "dedupe_radius_diff": 3,
    "use_player_filter": 1,
}


def load_active_params(path=ACTIVE_CONFIG_PATH):
    if not path.exists():
        return DEFAULT_ACTIVE_PARAMS.copy()
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    params = DEFAULT_ACTIVE_PARAMS.copy()
    params.update(data.get("active", {}))
    return params


def save_active_params(params, path=ACTIVE_CONFIG_PATH):
    clean = sanitize_active_params(params)
    payload = {"active": {key: clean[key] for key in DEFAULT_ACTIVE_PARAMS}}
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    return path


def load_players_params(path=PLAYERS_CONFIG_PATH):
    if not path.exists():
        return DEFAULT_PLAYERS_PARAMS.copy()
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    params = DEFAULT_PLAYERS_PARAMS.copy()
    params.update(data.get("players", {}))
    return params


def save_players_params(params, path=PLAYERS_CONFIG_PATH):
    clean = sanitize_players_params(params)
    payload = {"players": {key: clean[key] for key in DEFAULT_PLAYERS_PARAMS}}
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    return path


def load_obstacles_params(path=OBSTACLES_CONFIG_PATH):
    if not path.exists():
        return DEFAULT_OBSTACLES_PARAMS.copy()
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    params = DEFAULT_OBSTACLES_PARAMS.copy()
    params.update(data.get("obstacles", {}))
    return params


def save_obstacles_params(params, path=OBSTACLES_CONFIG_PATH):
    clean = sanitize_obstacles_params(params)
    payload = {"obstacles": {key: clean[key] for key in DEFAULT_OBSTACLES_PARAMS}}
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    return path


def radius_bounds(params):
    expected = int(params["expected_radius"])
    tol = int(params["radius_tolerance"])
    min_r = max(1, expected - tol)
    max_r = max(min_r + 1, expected + tol)
    return min_r, max_r


def sanitize_active_params(params):
    p = {**DEFAULT_ACTIVE_PARAMS, **params}
    p["match_max_dist"] = max(1, int(p["match_max_dist"]))
    p["glow_min_area"] = max(0, int(p["glow_min_area"]))
    p["red_excess_thresh"] = max(0, int(p["red_excess_thresh"]))
    p["left_side_only"] = 1 if int(p["left_side_only"]) else 0
    return p


def sanitize_players_params(params):
    p = {**DEFAULT_PLAYERS_PARAMS, **params}

    p["hough_param1"] = max(1, int(p["hough_param1"]))
    p["hough_param2"] = max(1, int(p["hough_param2"]))
    p["hough_min_dist"] = max(1, int(p["hough_min_dist"]))

    blur = max(3, int(p["player_blur"]))
    p["player_blur"] = blur if blur % 2 == 1 else blur + 1

    p["expected_radius"] = max(1, int(p["expected_radius"]))
    p["radius_tolerance"] = max(0, int(p["radius_tolerance"]))

    gap_lo = int(p["player_gray_gap_low"])
    gap_hi = int(p["player_gray_gap_high"])
    if gap_lo >= gap_hi:
        gap_hi = min(255, gap_lo + 2)
    p["player_gray_gap_low"] = gap_lo
    p["player_gray_gap_high"] = gap_hi

    return p


def sanitize_obstacles_params(params):
    p = {**DEFAULT_OBSTACLES_PARAMS, **params}

    p["black_gray_low"] = max(0, min(255, int(p["black_gray_low"])))
    p["black_gray_high"] = max(p["black_gray_low"], min(255, int(p["black_gray_high"])))

    p["hough_param1"] = max(1, int(p["hough_param1"]))
    p["hough_param2"] = max(1, int(p["hough_param2"]))
    p["hough_min_dist"] = max(1, int(p["hough_min_dist"]))

    blur = max(3, int(p["black_blur"]))
    p["black_blur"] = blur if blur % 2 == 1 else blur + 1

    p["min_radius"] = max(1, int(p["min_radius"]))
    p["max_radius"] = max(p["min_radius"] + 1, int(p["max_radius"]))
    p["detect_method"] = 1 if int(p.get("detect_method", 1)) else 0
    p["min_blob_area"] = max(1, int(p["min_blob_area"]))
    p["min_circularity"] = max(0, min(100, int(p["min_circularity"])))
    p["blob_close"] = max(0, min(8, int(p["blob_close"])))
    p["suppress_nested"] = 1 if int(p.get("suppress_nested", 1)) else 0
    p["inner_gray_max"] = max(0, min(255, int(p["inner_gray_max"])))
    p["inner_edge_margin"] = max(0, min(80, int(p["inner_edge_margin"])))
    p["player_overlap_reject"] = max(0, min(100, int(p["player_overlap_reject"])))
    p["player_center_dist"] = max(0, int(p["player_center_dist"]))
    p["dedupe_center_dist"] = max(0, int(p["dedupe_center_dist"]))
    p["dedupe_radius_diff"] = max(0, int(p["dedupe_radius_diff"]))
    p["use_player_filter"] = 1 if int(p["use_player_filter"]) else 0
    return p


def sanitize_params(params):
    """Backward-compatible merge (used by calibrate_active during transition)."""
    return {**sanitize_players_params(params), **sanitize_active_params(params)}


def pixel_to_game_x(px, field_width):
    return -25 + px * 50 / field_width


def pixel_radius_to_game(r, field_width):
    return r * 50 / field_width


def classify_side(cx, r, field_width):
    """
    Left half of the screen is always our team; right half is enemy.
    Near the vertical center, account for circle radius (not just the center point).
    """
    center_x = field_width / 2

    if cx + r <= center_x:
        return "ours"
    if cx - r >= center_x:
        return "enemy"
    return "ours" if cx < center_x else "enemy"


def detect_red_glow_mask(bgr, params):
    b, g, r = cv2.split(bgr)
    red_excess = cv2.subtract(r, cv2.max(g, b))
    _, excess_mask = cv2.threshold(
        red_excess, int(params["red_excess_thresh"]), 255, cv2.THRESH_BINARY
    )

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower1 = np.array(
        [int(params["hsv_h_low"]), int(params["hsv_s_min"]), int(params["hsv_v_min"])],
        dtype=np.uint8,
    )
    upper1 = np.array([int(params["hsv_h_high"]), 255, 255], dtype=np.uint8)
    lower2 = np.array(
        [int(params["hsv_h_low2"]), int(params["hsv_s_min"]), int(params["hsv_v_min"])],
        dtype=np.uint8,
    )
    upper2 = np.array([int(params["hsv_h_high2"]), 255, 255], dtype=np.uint8)
    hsv_mask = cv2.bitwise_or(
        cv2.inRange(hsv, lower1, upper1),
        cv2.inRange(hsv, lower2, upper2),
    )

    mask = cv2.bitwise_or(excess_mask, hsv_mask)

    blur = int(params["glow_blur"])
    if blur > 0:
        k = blur if blur % 2 == 1 else blur + 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
        _, mask = cv2.threshold(mask, 40, 255, cv2.THRESH_BINARY)

    dilate = int(params["glow_dilate"])
    if dilate > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kernel, iterations=dilate)

    return mask


def glow_centroid(mask, min_area):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    best = None
    best_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area and area > best_area:
            best = contour
            best_area = area

    if best is None:
        return None, None

    moments = cv2.moments(best)
    if moments["m00"] == 0:
        return None, best_area

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    return (cx, cy), best_area


def filter_uniform_players(circles, params):
    """Keep circles matching the expected player radius (all players same size)."""
    if circles is None:
        return None

    min_r, max_r = radius_bounds(params)
    filtered = []
    for pt in circles[0]:
        cx, cy, r = int(pt[0]), int(pt[1]), int(pt[2])
        if min_r <= r <= max_r:
            filtered.append([cx, cy, r])

    if not filtered:
        return None

    return np.array([filtered], dtype=np.uint16)


def detect_player_circles(gray, params, filter_radius=True):
    params = sanitize_players_params(params)
    lower = int(params["player_gray_low"])
    upper = int(params["player_gray_high"])
    gap_low = int(params["player_gray_gap_low"])
    gap_high = int(params["player_gray_gap_high"])

    mask1 = cv2.inRange(gray, lower, gap_low)
    mask2 = cv2.inRange(gray, gap_high, upper)
    mask = cv2.bitwise_or(mask1, mask2)

    result = np.ones_like(gray) * 255
    result[mask == 255] = 0

    blur = int(params["player_blur"])
    result = cv2.GaussianBlur(result, (blur, blur), 0)

    min_r, max_r = radius_bounds(params)
    circles = cv2.HoughCircles(
        result,
        cv2.HOUGH_GRADIENT,
        1,
        minDist=int(params["hough_min_dist"]),
        param1=int(params["hough_param1"]),
        param2=int(params["hough_param2"]),
        minRadius=min_r,
        maxRadius=max_r,
    )

    if filter_radius:
        circles = filter_uniform_players(circles, params)

    if circles is None:
        return None, result

    return np.uint16(np.around(circles)), result


def find_all_players(bgr, field_width, players_params=None):
    players_params = sanitize_players_params(players_params or load_players_params())
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    players, player_mask = detect_player_circles(gray, players_params)

    ours = []
    enemies = []
    if players is not None:
        for pt in players[0]:
            cx, cy, r = int(pt[0]), int(pt[1]), int(pt[2])
            entry = (cx, cy, r)
            if classify_side(cx, r, field_width) == "enemy":
                enemies.append(entry)
            else:
                ours.append(entry)

    return {
        "players": players,
        "ours": ours,
        "enemies": enemies,
        "player_mask": player_mask,
        "gray": gray,
        "params": players_params,
    }


def find_active_player(bgr, field_width, active_params=None, players_params=None):
    active_params = sanitize_active_params(active_params or load_active_params())
    players_params = sanitize_players_params(players_params or load_players_params())

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    glow_mask = detect_red_glow_mask(bgr, active_params)
    glow_center, glow_area = glow_centroid(glow_mask, int(active_params["glow_min_area"]))

    players, player_mask = detect_player_circles(gray, players_params)
    center_x = field_width / 2

    candidates = []
    if players is not None:
        for pt in players[0]:
            cx, cy, r = int(pt[0]), int(pt[1]), int(pt[2])
            if active_params["left_side_only"] and cx > center_x + r:
                continue
            candidates.append((cx, cy, r))

    active = None
    method = "none"
    expected_r, _ = radius_bounds(players_params)

    if glow_center and candidates:
        gx, gy = glow_center
        max_dist = float(active_params["match_max_dist"])
        best = None
        best_score = float("inf")

        for cx, cy, r in candidates:
            overlap = glow_mask[
                max(0, cy - r) : cy + r,
                max(0, cx - r) : cx + r,
            ]
            overlap_ratio = np.count_nonzero(overlap) / overlap.size if overlap.size else 0
            dist = ((cx - gx) ** 2 + (cy - gy) ** 2) ** 0.5
            score = dist - overlap_ratio * 30
            if dist <= max_dist and score < best_score:
                best_score = score
                best = (cx, cy, r)

        if best:
            active = best
            method = "glow+player"

    if active is None and glow_center and candidates:
        gx, gy = glow_center
        nearest = min(candidates, key=lambda p: (p[0] - gx) ** 2 + (p[1] - gy) ** 2)
        active = nearest
        method = "glow+nearest"

    if active is None and glow_center and not candidates:
        active = (glow_center[0], glow_center[1], expected_r)
        method = "glow_only"

    if active is None and candidates:
        active = min(candidates, key=lambda p: p[0])
        method = "leftmost"

    return {
        "active": active,
        "method": method,
        "glow_center": glow_center,
        "glow_area": glow_area,
        "players": players,
        "candidates": candidates,
        "glow_mask": glow_mask,
        "player_mask": player_mask,
        "gray": gray,
    }


def draw_players_overlay(bgr, result, field_width, active=None):
    out = bgr.copy()
    h, w = out.shape[:2]
    center_x = int(field_width / 2)
    cv2.line(out, (center_x, 0), (center_x, h), (0, 180, 0), 1)

    active_tuple = tuple(active) if active else None
    radii = []
    if result["players"] is not None:
        for pt in result["players"][0]:
            cx, cy, r = int(pt[0]), int(pt[1]), int(pt[2])
            radii.append(r)
            player = (cx, cy, r)
            if active_tuple and player == active_tuple:
                continue
            side = classify_side(cx, r, field_width)
            color = (0, 120, 255) if side == "enemy" else (255, 200, 0)
            cv2.circle(out, (cx, cy), r, color, 2)
            cv2.circle(out, (cx, cy), 2, color, -1)
            cv2.putText(
                out,
                f"r={r}",
                (cx + r + 2, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA,
            )

    if active_tuple:
        cx, cy, r = active_tuple
        cv2.circle(out, (cx, cy), r + 4, (0, 255, 0), 2)
        cv2.circle(out, (cx, cy), 2, (0, 255, 0), -1)
        cv2.putText(
            out,
            "ACTIVE",
            (cx + r + 4, cy - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    enemy_n = len(result["enemies"])
    ours_n = len(result["ours"])
    total = enemy_n + ours_n
    radius_info = ""
    if radii:
        radius_info = f"  r={min(radii)}..{max(radii)} avg={sum(radii)/len(radii):.1f}"

    cv2.putText(
        out,
        f"players={total}  ours={ours_n}  enemy={enemy_n}{radius_info}",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        "green=active  orange=ours  blue=enemy  line=screen center",
        (8, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    return out


def draw_detection_overlay(bgr, result, field_width):
    out = bgr.copy()
    h, w = out.shape[:2]
    center_x = int(field_width / 2)

    cv2.line(out, (center_x, 0), (center_x, h), (0, 180, 0), 1)

    glow_tint = np.zeros_like(out)
    glow_tint[:, :, 2] = result["glow_mask"]
    out = cv2.addWeighted(out, 1.0, glow_tint, 0.35, 0)

    if result["glow_center"]:
        gx, gy = result["glow_center"]
        cv2.drawMarker(out, (gx, gy), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)

    if result["players"] is not None:
        for pt in result["players"][0]:
            cx, cy, r = int(pt[0]), int(pt[1]), int(pt[2])
            side = classify_side(cx, r, field_width)
            color = (0, 120, 255) if side == "enemy" else (255, 200, 0)
            if result["active"] and (cx, cy, r) == result["active"]:
                continue
            cv2.circle(out, (cx, cy), r, color, 1)
            cv2.circle(out, (cx, cy), 2, color, -1)

    if result["active"]:
        cx, cy, r = result["active"]
        cv2.circle(out, (cx, cy), r + 3, (0, 255, 0), 2)
        cv2.circle(out, (cx, cy), 2, (0, 255, 0), -1)
        game_x = pixel_to_game_x(cx, field_width)
        game_r = pixel_radius_to_game(r, field_width)
        label = f"ACTIVE ({result['method']}) x={game_x:.2f} r={game_r:.2f}"
        cv2.putText(
            out, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA
        )

    return out


def circle_overlap_ratio(c1, c2):
    """Intersection area / smaller circle area."""
    x1, y1, r1 = c1
    x2, y2, r2 = c2
    dist = math.hypot(x1 - x2, y1 - y2)
    if dist >= r1 + r2:
        return 0.0
    if dist <= abs(r1 - r2):
        return 1.0

    r_small, r_large = (r1, r2) if r1 <= r2 else (r2, r1)
    part1 = r_small ** 2 * math.acos(
        max(-1.0, min(1.0, (dist ** 2 + r_small ** 2 - r_large ** 2) / (2 * dist * r_small + 1e-9)))
    )
    part2 = r_large ** 2 * math.acos(
        max(-1.0, min(1.0, (dist ** 2 + r_large ** 2 - r_small ** 2) / (2 * dist * r_large + 1e-9)))
    )
    part3 = 0.5 * math.sqrt(
        max(
            0.0,
            (-dist + r_small + r_large)
            * (dist + r_small - r_large)
            * (dist - r_small + r_large)
            * (dist + r_small + r_large),
        )
    )
    intersection = part1 + part2 - part3
    smaller_area = math.pi * r_small ** 2
    return intersection / smaller_area if smaller_area else 0.0


def inner_mean_gray(gray, cx, cy, radius, edge_margin_pct):
    """Mean grayscale inside a circle, ignoring the outer edge ring."""
    h, w = gray.shape[:2]
    margin = max(0.0, min(0.8, edge_margin_pct / 100.0))
    inner_r = max(1, int(radius * (1.0 - margin)))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (int(cx), int(cy)), inner_r, 255, -1)
    pixels = gray[mask == 255]
    if pixels.size == 0:
        return 255.0
    return float(np.mean(pixels))


def dedupe_obstacle_circles(circles, params):
    if circles is None:
        return None

    center_dist = int(params["dedupe_center_dist"])
    radius_diff = int(params["dedupe_radius_diff"])
    kept = []

    for pt in sorted(circles[0], key=lambda p: p[2]):
        cx, cy, r = int(pt[0]), int(pt[1]), int(pt[2])
        duplicate = False
        for kx, ky, kr in kept:
            if math.hypot(cx - kx, cy - ky) < center_dist and abs(r - kr) <= radius_diff:
                duplicate = True
                break
        if not duplicate:
            kept.append([cx, cy, r])

    if not kept:
        return None
    return np.array([kept], dtype=np.uint16)


def _circles_from_points(points):
    if not points:
        return None
    return np.array([points], dtype=np.uint16)


def _build_dark_mask(gray, params):
    lower = int(params["black_gray_low"])
    upper = int(params["black_gray_high"])
    return cv2.inRange(gray, lower, upper)


def suppress_nested_same_blob(circles, dark_mask):
    """
    Drop smaller Hough hits that sit inside a larger circle on the same dark blob.
    Keeps real separate nested obstacles (white gap -> different connected component).
    """
    if circles is None or len(circles[0]) < 2:
        return circles

    _, labels = cv2.connectedComponents(dark_mask)
    h, w = labels.shape[:2]
    pts = sorted(
        [(int(p[0]), int(p[1]), int(p[2])) for p in circles[0]],
        key=lambda p: -p[2],
    )
    kept = []

    for cx, cy, r in pts:
        if not (0 <= cx < w and 0 <= cy < h):
            continue
        label = labels[cy, cx]
        if label == 0:
            continue

        dominated = False
        for kx, ky, kr in kept:
            if r >= kr:
                continue
            if math.hypot(cx - kx, cy - ky) + r <= kr + 2 and labels[ky, kx] == label:
                dominated = True
                break

        if not dominated:
            kept.append((cx, cy, r))

    return _circles_from_points([[x, y, r] for x, y, r in kept])


def _detect_obstacles_hough(gray, params, dark_mask):
    result = np.ones_like(gray) * 255
    result[dark_mask == 255] = 0

    blur = int(params["black_blur"])
    result = cv2.GaussianBlur(result, (blur, blur), 0)

    circles = cv2.HoughCircles(
        result,
        cv2.HOUGH_GRADIENT,
        1,
        minDist=int(params["hough_min_dist"]),
        param1=int(params["hough_param1"]),
        param2=int(params["hough_param2"]),
        minRadius=int(params["min_radius"]),
        maxRadius=int(params["max_radius"]),
    )

    circles = dedupe_obstacle_circles(circles, params)
    if circles is not None and params["suppress_nested"]:
        circles = suppress_nested_same_blob(circles, dark_mask)
    return circles, result


def _detect_obstacles_blobs(gray, params, dark_mask):
    close_iters = int(params["blob_close"])
    work = dark_mask
    if close_iters > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        work = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=close_iters)

    contours, _ = cv2.findContours(work, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = int(params["min_blob_area"])
    min_circ = int(params["min_circularity"]) / 100.0
    min_r = int(params["min_radius"])
    max_r = int(params["max_radius"])

    circles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue
        circ = 4 * math.pi * area / (peri * peri)
        if circ < min_circ:
            continue

        (cx, cy), r_enc = cv2.minEnclosingCircle(cnt)
        r_area = math.sqrt(area / math.pi)
        r = int(round((r_enc + r_area) / 2))
        if min_r <= r <= max_r:
            circles.append([int(cx), int(cy), r])

    preview = cv2.cvtColor(work, cv2.COLOR_GRAY2BGR)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(preview, [cnt], -1, (0, 255, 255), 1)

    return _circles_from_points(circles), preview


def detect_obstacle_circles(gray, params):
    params = sanitize_obstacles_params(params)
    dark_mask = _build_dark_mask(gray, params)

    if params["detect_method"] == 0:
        circles, obstacle_mask = _detect_obstacles_hough(gray, params, dark_mask)
    else:
        circles, obstacle_mask = _detect_obstacles_blobs(gray, params, dark_mask)

    return circles, obstacle_mask, dark_mask


def _players_as_tuples(players):
    if players is None:
        return []
    return [(int(pt[0]), int(pt[1]), int(pt[2])) for pt in players[0]]


def filter_obstacle_candidates(circles, gray, params, players=None):
    """Drop non-dark circles and circles that overlap detected players."""
    if circles is None:
        return [], []

    params = sanitize_obstacles_params(params)
    player_list = _players_as_tuples(players)
    overlap_thresh = params["player_overlap_reject"] / 100.0
    center_dist_thresh = int(params["player_center_dist"])
    inner_max = int(params["inner_gray_max"])
    use_player_filter = bool(params["use_player_filter"])

    accepted = []
    rejected = []

    for pt in circles[0]:
        cx, cy, r = int(pt[0]), int(pt[1]), int(pt[2])
        circle = (cx, cy, r)
        inner_mean = inner_mean_gray(gray, cx, cy, r, params["inner_edge_margin"])

        if inner_mean > inner_max:
            rejected.append({"circle": circle, "reason": f"bright inner={inner_mean:.0f}"})
            continue

        if use_player_filter and player_list:
            player_hit = None
            for player in player_list:
                px, py, pr = player
                overlap = circle_overlap_ratio(circle, player)
                center_dist = math.hypot(cx - px, cy - py)
                if overlap >= overlap_thresh:
                    player_hit = (player, f"overlap={overlap:.2f}")
                    break
                if center_dist <= pr + 2 and r <= pr + 4:
                    player_hit = (player, f"center dist={center_dist:.1f}")
                    break
                if center_dist <= center_dist_thresh and abs(r - pr) <= 3:
                    player_hit = (player, f"player-sized d={center_dist:.1f}")
                    break

            if player_hit:
                _, reason = player_hit
                rejected.append({"circle": circle, "reason": f"player {reason}"})
                continue

        accepted.append(circle)

    return accepted, rejected


def count_nested_pairs(circles):
    pairs = 0
    for i, (x1, y1, r1) in enumerate(circles):
        for x2, y2, r2 in circles[i + 1 :]:
            dist = math.hypot(x1 - x2, y1 - y2)
            if dist < abs(r1 - r2) * 0.35 and min(r1, r2) >= 2:
                pairs += 1
    return pairs


def find_all_obstacles(bgr, obstacles_params=None, players_params=None, filter_players=True):
    obstacles_params = sanitize_obstacles_params(obstacles_params or load_obstacles_params())
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    raw_circles, obstacle_mask, dark_mask = detect_obstacle_circles(gray, obstacles_params)

    players = None
    player_list = []
    if filter_players:
        players_params = sanitize_players_params(players_params or load_players_params())
        players, _ = detect_player_circles(gray, players_params)
        player_list = _players_as_tuples(players)

    accepted, rejected = filter_obstacle_candidates(
        raw_circles, gray, obstacles_params, players if filter_players else None
    )

    return {
        "obstacles": accepted,
        "rejected": rejected,
        "raw_count": 0 if raw_circles is None else len(raw_circles[0]),
        "nested_pairs": count_nested_pairs(accepted),
        "detect_method": obstacles_params["detect_method"],
        "players": player_list,
        "player_circles": players,
        "obstacle_mask": obstacle_mask,
        "dark_mask": dark_mask,
        "gray": gray,
        "params": obstacles_params,
    }


def draw_obstacles_overlay(bgr, result, field_width=None, show_players=True, show_rejected=False):
    out = bgr.copy()
    h, w = out.shape[:2]

    if field_width is not None:
        center_x = int(field_width / 2)
        cv2.line(out, (center_x, 0), (center_x, h), (0, 180, 0), 1)

    if show_players and result["players"]:
        for px, py, pr in result["players"]:
            cv2.circle(out, (px, py), pr, (255, 200, 0), 1)
            cv2.circle(out, (px, py), 2, (255, 200, 0), -1)

    if show_rejected:
        for item in result["rejected"]:
            cx, cy, r = item["circle"]
            cv2.circle(out, (cx, cy), r, (80, 80, 80), 1)
            cv2.putText(
                out,
                item["reason"][:18],
                (cx + r + 2, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (120, 120, 120),
                1,
                cv2.LINE_AA,
            )

    obstacles = sorted(result["obstacles"], key=lambda c: c[2])
    for idx, (cx, cy, r) in enumerate(obstacles):
        hue = (idx * 47) % 180
        color = cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        color = (int(color[0]), int(color[1]), int(color[2]))
        cv2.circle(out, (cx, cy), r, color, 2)
        cv2.circle(out, (cx, cy), 2, color, -1)
        cv2.putText(
            out,
            f"r={r}",
            (cx + r + 2, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )

    n = len(obstacles)
    rej = len(result["rejected"])
    nested = result["nested_pairs"]
    raw = result["raw_count"]
    method = "blobs" if result.get("detect_method", 1) else "hough"
    cv2.putText(
        out,
        f"obstacles={n}  raw={raw}  rejected={rej}  nested={nested}  mode={method}",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        "colored=obstacle  orange=player  gray=rejected (toggle p/r)",
        (8, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    return out
