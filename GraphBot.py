"""
   ____                     _      ____          _   
  / ___| _ __  __ _  _ __  | |__  | __ )   ___  | |_ 
 | |  _ | '__|/ _` || '_ \ | '_ \ |  _ \  / _ \ | __|
 | |_| || |  | (_| || |_) || | | || |_) || (_) || |_ 
  \____||_|   \__,_|| .__/ |_| |_||____/  \___/  \__|
                    |_|
                    
                            15
                            ▲ enemy
                            │   X
                            │  * *
                            │ *   *
                            │*     *
                            *      enemy        enemy
                           *│        X************X
-25 ──────────────────────*─│────────────────────────► 25
                         *  │
           KroSheChKa  **   │
                @******     │
                            │
                            │
                            │
                           -15

=========================================================
 GraphBot.py
=========================================================
Description:
    GraphBot is a tool that automatically plots straight-line trajectories from
    the active player to all enemies on the game field, enabling precise targeting.
    It also features a manual mode for creating custom paths by clicking on the
    field, perfect for avoiding obstacles like black spheres.

Author: KroSheChKa
Github: https://github.com/KroSheChKa/GraphBot
License: MIT License
Date: 2024-12-05
=========================================================

MIT License

Copyright (c) 2024 KroSheChKa

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
=========================================================
"""

# =========================================================
# Imports
# =========================================================
import cv2
import mss
import sys, ctypes
import time
import random
import re
import numpy as np
import win32api, win32con, win32gui
import pyperclip

from core.window_capture import find_game_window, get_capture_field, load_capture_margins
from core.detection import (
    find_active_player,
    find_all_obstacles,
    find_all_players,
    load_active_params,
    load_obstacles_params,
    load_players_params,
)
from core.avoidance import field_obstacles_to_game, DEFAULT_CLEARANCE
from core.pathfinding import astar_game, build_enemy_chain_astar, draw_path_on_field
from core.polynomial_planner import (
    draw_polynomial_curve_on_field,
    draw_polynomial_stats_overlay,
    search_best_polynomial,
)
from core.symbolic_ga_planner import (
    draw_symbolic_curve_on_field,
    draw_symbolic_stats_overlay,
    search_best_symbolic_ga,
)

# =========================================================
# Functions
# =========================================================
def is_key_pressed(key):
    return ctypes.windll.user32.GetAsyncKeyState(key) & 0x8000 != 0

# A function to print out the status of finishing of the program and exiting it
def safe_exit(exit_code):
    print(exit_codes[exit_code])
    sys.exit()

# A function to move the game window to a certain position with logs.
def move_window(window_title, target_x, target_y):
    hwnd = win32gui.FindWindow(None, window_title)
    if hwnd == 0:
        print(f"[window] not found: '{window_title}'")
        return None, False

    title = win32gui.GetWindowText(hwnd)
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    width = right - left
    height = bottom - top
    print(
        f"[window] found: '{title}' hwnd={hwnd} "
        f"rect=({left}, {top}, {right}, {bottom})"
    )

    if left == target_x and top == target_y:
        print(f"[window] not moved: already at ({target_x}, {target_y})")
        return hwnd, False

    try:
        win32gui.MoveWindow(hwnd, target_x, target_y, width, height, True)
    except Exception as exc:
        print(f"[window] move failed: {exc}")
        return hwnd, False

    new_left, new_top, new_right, new_bottom = win32gui.GetWindowRect(hwnd)
    print(
        "[window] moved: "
        f"({left}, {top}) -> ({new_left}, {new_top}), "
        f"size={new_right - new_left}x{new_bottom - new_top}"
    )
    return hwnd, True

# More accurate sleep function with ability to exit the
def sleep_key(sec):
    start_time = time.time()
    while True:
        if is_key_pressed(exit_key):
            safe_exit(0)
        
        current_time = time.time()
        elapsed_time = current_time - start_time

        if elapsed_time >= sec:
            break

# Function to detect the cords and radius of the black circles
def detect_black_circles(s_r):
    s_r = cv2.GaussianBlur(s_r, (3, 3), 0)

    lower_bound = 0
    upper_bound = 20

    mask = cv2.inRange(s_r, lower_bound, upper_bound)

    result = np.ones_like(s_r) * 255
    result[mask == 255] = 0
    result = cv2.GaussianBlur(result, (13, 13), 0)

    detected_circles = cv2.HoughCircles(result,
                    cv2.HOUGH_GRADIENT, 1, minDist= 15, param1 = 50,
                param2 = 25, minRadius = 1, maxRadius = 200)
    
    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))
        return detected_circles
    else:
        print('Probably there is no black circles. It might be a mistake')
        return None

def detect_players(s_r):
    # These the interval in grayscale to mask (delete unneeded details)
    lower_bound = 50
    upper_bound = 250

    mask1 = cv2.inRange(s_r, lower_bound, 169)
    mask2 = cv2.inRange(s_r, 171, upper_bound)
    mask = cv2.bitwise_or(mask1, mask2)
    
    result = np.ones_like(s_r) * 255
    result[mask == 255] = 0

    # Well, actually without blur func.GaussianBlur working pretty badly
    blur_rate = 23
    result = cv2.GaussianBlur(result, (blur_rate, blur_rate), 0)

    # A magic formula to get the circles
    detected_circles = cv2.HoughCircles(result,
        cv2.HOUGH_GRADIENT, 1, minDist= 10, param1 = 150,
        param2 = 10, minRadius = 4, maxRadius = 15)
    
    # cv2.imshow('GraphBot', result)
    # cv2.waitKey(1)

    if detected_circles is not None: 
        detected_circles = np.uint16(np.around(detected_circles))
        return detected_circles
    else:
        print('Probably there is no players. It might be a mistake')
        return None

def draw_circles(circles, screenshot_r):
    for pt in circles[0,:]:
            a, b, r = pt[0], pt[1], pt[2]
            cv2.circle(screenshot_r, (a, b), r, (100, 0, 0), 2)
            cv2.circle(screenshot_r, (a, b), 1, (255, 0, 0), 3)
    return screenshot_r


def draw_players_roles_overlay(screenshot_bgr, allies, enemies, active):
    """Draw allies/enemies/active with distinct colors."""
    out = screenshot_bgr

    # Enemies (right side): red
    for cx, cy, radius in enemies:
        cv2.circle(out, (int(cx), int(cy)), int(radius), (0, 70, 255), 2)
        cv2.circle(out, (int(cx), int(cy)), 2, (0, 70, 255), -1)

    # Allies (left side, except active): cyan/orange-like
    for cx, cy, radius in allies:
        cv2.circle(out, (int(cx), int(cy)), int(radius), (255, 200, 0), 2)
        cv2.circle(out, (int(cx), int(cy)), 2, (255, 200, 0), -1)

    # Active player: bright green with extra ring
    if active is not None:
        cx, cy, radius = active
        cv2.circle(out, (int(cx), int(cy)), int(radius) + 3, (0, 255, 0), 2)
        cv2.circle(out, (int(cx), int(cy)), 2, (0, 255, 0), -1)

    return out


def draw_obstacle_circles(screenshot_bgr, obstacles):
    for cx, cy, r in obstacles:
        cv2.circle(screenshot_bgr, (int(cx), int(cy)), int(r), (255, 0, 255), 1)
    return screenshot_bgr


def separate(players):
    active = players[0][0]
    good = []
    bad = []
    for i in players[0]:
        if i[2] > active[2]:
            active = i
        if i[0] > field['width'] / 2:
            bad.append(i[:2])
        else:
            if i[2] > active[2]:
                active = i
            good.append(i[:2])
    return good, bad, active[:2]

GAME_PRECISION = 5
VERTICAL_MAX_COEFF = 999
VERTICAL_MIN_EPS = 0.001
CLICK_LEFT_TOLERANCE = 0.08  # ~1 px in game coords — ignore barely-left clicks


def fmt_game(value):
    """Round game coordinates and formula coefficients for Graphwar."""
    return round(float(value), GAME_PRECISION)


def vertical_eps(y_from, y_to, max_coeff=VERTICAL_MAX_COEFF):
    """Pick eps so the steep segment coefficient stays within Graphwar limits."""
    dy = abs(y_to - y_from)
    if dy < 1e-9:
        return VERTICAL_MIN_EPS
    return max(VERTICAL_MIN_EPS, dy / (2 * max_coeff))


def field_to_game(field_x, field_y):
    game_x = -25 + field_x * 50 / field["width"]
    game_y = 15 - field_y * 50 / field["width"]
    return fmt_game(game_x), fmt_game(game_y)


def to_game_cords(cord_list):
    return [list(field_to_game(i[0], i[1])) for i in cord_list]


def field_radius_to_game(radius_px):
    return fmt_game(radius_px * 50 / field["width"])


def field_circle_to_game(circle):
    cx, cy, radius = circle
    gx, gy = field_to_game(cx, cy)
    gr = field_radius_to_game(radius)
    return (gx, gy, gr)


def split_players_for_auto(players):
    """Return (allies_left, enemies_right, active_left) in field pixels."""
    left = []
    right = []
    center_x = field["width"] / 2

    for pt in players[0]:
        cx, cy, radius = int(pt[0]), int(pt[1]), int(pt[2])
        if cx > center_x:
            right.append((cx, cy, radius))
        else:
            left.append((cx, cy, radius))

    if not left:
        return [], right, None

    active_idx = max(range(len(left)), key=lambda idx: left[idx][2])
    active = left[active_idx]
    allies = [pt for idx, pt in enumerate(left) if idx != active_idx]
    return allies, right, active


def collect_symbolic_scene(screenshot_bgr, players_params, active_params, obstacles_params):
    players_result = find_all_players(
        screenshot_bgr,
        field["width"],
        players_params=players_params,
    )
    active_result = find_active_player(
        screenshot_bgr,
        field["width"],
        active_params=active_params,
        players_params=players_params,
    )
    obstacle_result = find_all_obstacles(
        screenshot_bgr,
        obstacles_params=obstacles_params,
        players_params=players_params,
        filter_players=True,
    )

    enemies_field = sorted(
        [tuple(int(v) for v in circle) for circle in players_result["enemies"]],
        key=lambda circle: circle[0],
    )
    ours_field = [tuple(int(v) for v in circle) for circle in players_result["ours"]]

    active_circle = active_result["active"]
    if active_circle is not None:
        active_circle = tuple(int(v) for v in active_circle)
    elif ours_field:
        # Fallback to legacy heuristic if glow-based active was not found.
        active_circle = max(ours_field, key=lambda circle: circle[2])

    allies_field = []
    if active_circle is not None:
        skipped_active = False
        for circle in ours_field:
            if not skipped_active and circle == active_circle:
                skipped_active = True
                continue
            allies_field.append(circle)
    else:
        allies_field = ours_field

    obstacles_field = obstacle_result["obstacles"]

    active_norm = None
    if active_circle is not None:
        active_norm = field_to_game(active_circle[0], active_circle[1])

    enemy_circles_game = [field_circle_to_game(circle) for circle in enemies_field]
    allies_game = [field_circle_to_game(circle) for circle in allies_field]
    obstacles_game = field_obstacles_to_game(obstacles_field, field["width"])

    return {
        "players_result": players_result,
        "active_result": active_result,
        "obstacles_result": obstacle_result,
        "enemies_field": enemies_field,
        "allies_field": allies_field,
        "active_circle": active_circle,
        "obstacles_field": obstacles_field,
        "enemy_circles_game": enemy_circles_game,
        "allies_game": allies_game,
        "obstacles_game": obstacles_game,
        "active_norm": active_norm,
    }


def normalize_formula(text):
    """Collapse - - / -- / + + to + and mixed signs (+-, -+) to -."""
    s = str(text)
    while True:
        prev = s
        s = re.sub(r"-\s*-", "+", s)
        s = re.sub(r"\+\s*\+", "+", s)
        s = re.sub(r"\+\s*-", "-", s)
        s = re.sub(r"-\s*\+", "-", s)
        if s == prev:
            break
    return s


def direct_line(p1, p2):
    x1, y1 = fmt_game(p1[0]), fmt_game(p1[1])
    x2, y2 = fmt_game(p2[0]), fmt_game(p2[1])
    dx = x2 - x1
    if abs(dx) < 1e-12:
        dx = fmt_game(vertical_eps(y1, y2)) if y1 != y2 else VERTICAL_MIN_EPS
        x2 = fmt_game(x1 + dx)
    dist = fmt_game(-((y1 - y2) / 2) / dx)
    return f"{dist}*(abs(x - {x1}) - abs(x - {x2}))"


def process_clicks_to_waypoints(clicks):
    """
    Clicks in press order. First click = formula anchor (where the graph must
    already be when the formula starts). Active player is NOT included.
    """
    if not clicks:
        return []

    waypoints = []
    for field_x, field_y in clicks:
        game_x, game_y = field_to_game(field_x, field_y)
        if not waypoints:
            waypoints.append([game_x, game_y])
            continue

        prev_x, prev_y = waypoints[-1]

        if game_x < prev_x - CLICK_LEFT_TOLERANCE:
            y_target = game_y
            if abs(y_target - prev_y) < 1e-6:
                print("Click left at same height — skipped.")
                continue

            direction = "up" if y_target > prev_y else "down"
            eps = vertical_eps(prev_y, y_target)
            end_x = fmt_game(prev_x + eps)
            steepness = abs(y_target - prev_y) / (2 * eps)
            print(
                f"Click left of previous point -> vertical {direction} "
                f"at x={prev_x:.4f}, y {prev_y:.4f} -> {y_target:.4f} "
                f"(steepness~{steepness:.0f})"
            )
            waypoints.append([end_x, y_target])
        else:
            waypoints.append([game_x, game_y])

    return waypoints


def waypoints_to_formula(waypoints):
    parts = []
    for i in range(len(waypoints) - 1):
        parts.append(direct_line(tuple(waypoints[i]), tuple(waypoints[i + 1])))
    return normalize_formula(" + ".join(parts))


def is_click_in_field(screen_x, screen_y):
    """True if screen coordinates are inside the captured game field."""
    return (
        field["left"] <= screen_x < field["left"] + field["width"]
        and field["top"] <= screen_y < field["top"] + field["height"]
    )


def collect_clicks():
    print("Click on the game field (outside clicks ignored). F3 = start, F4 = done.")
    if not refresh_field():
        print("Graphwar window not found — cannot collect clicks.")
        return []

    clicks = []
    while not is_key_pressed(clicks_start):
        pass

    while not is_key_pressed(clicks_end):
        if is_key_pressed(left_mouse_key):
            screen_x, screen_y = win32gui.GetCursorPos()
            if not is_click_in_field(screen_x, screen_y):
                print(f"Ignored click outside field: screen ({screen_x}, {screen_y})")
                win32api.keybd_event(left_mouse_key, 0, win32con.KEYEVENTF_KEYUP, 0)
                continue

            field_x = screen_x - field["left"]
            field_y = screen_y - field["top"]
            print((field_x, field_y))
            clicks.append((field_x, field_y))
            win32api.keybd_event(left_mouse_key, 0, win32con.KEYEVENTF_KEYUP, 0)

    return clicks


def warn_no_players(context=""):
    prefix = f"{context}: " if context else ""
    print(
        f"{prefix}No players detected.\n"
        "  - Is Graphwar open and a round in progress?\n"
        "  - Is the capture region correct? (tools/preview_capture.py)\n"
        "  - Tune detection (tools/calibrate_players.py / tools/calibrate_active.py)"
    )

# Prevents unnesessary clipboard copying
def safe_copy(text, previous_text):
    if text != previous_text:
        pyperclip.copy(text)
        print("Safely copied!")

def setup():
    global field

    hwnd, _ = move_window(game_window_name, window_start_cords[0], window_start_cords[1])
    if hwnd is None:
        safe_exit(1)

    field = get_capture_field(hwnd, capture_margins)
    print(f"Capture region: {field}")
    print(f"Margins (client-relative): {capture_margins}")

    # Handling user mode input
    # 0 - automatic
    # 1 - clicks
    while not(is_key_pressed(exit_key)):
        print("Select the mode\n0 - automatic\n1 - clicks")
        mode = input().strip()
        if len(mode) == 1 and (mode == '0' or mode == '1'):
            mode = int(mode)
            break
        print("Incorrect input!\n")

    auto_planner = 0
    if mode == 0:
        while not(is_key_pressed(exit_key)):
            print(
                "Select auto planner\n"
                "0 - A* chain (current)\n"
                "1 - polynomial search (experimental)\n"
                "2 - symbolic GA (graphwar-like)"
            )
            planner = input().strip()
            if planner in ("0", "1", "2"):
                auto_planner = int(planner)
                break
            print("Incorrect input!\n")

    return mode, auto_planner

def refresh_field():
    global field

    hwnd = find_game_window(game_window_name)
    if hwnd is None:
        return False

    field = get_capture_field(hwnd, capture_margins)
    return True

def main():
    prev_text = ""
    prev_summary = ""
    poly_best_coeffs = None
    poly_rng = random.Random()
    poly_scene = None
    sym_best_genome = None
    sym_rng = random.Random()
    players_params = load_players_params()
    active_params = load_active_params()
    obstacles_params = load_obstacles_params()
    mss_ = mss.mss()
    if auto_planner == 0:
        print("Auto mode: F2 = quit. Planner: A*. Updates every ~1 s.")
    elif auto_planner == 1:
        print("Auto mode: F2 = quit. Planner: Polynomial Search. Updates every ~1 s.")
    else:
        print("Auto mode: F2 = quit. Planner: Symbolic GA. Updates every ~1 s.")
    while not(is_key_pressed(exit_key)):
        if not mode and auto_planner == 2:
            if not refresh_field():
                print("Graphwar window not found. Waiting...")
                sleep_key(0.5)
                continue

            screenshot = np.array(mss_.grab(field))
            screenshot_r = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)
            screenshot_bgr = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)

            scene = collect_symbolic_scene(
                screenshot_bgr,
                players_params=players_params,
                active_params=active_params,
                obstacles_params=obstacles_params,
            )
            screenshot_vis = cv2.cvtColor(screenshot_r, cv2.COLOR_GRAY2BGR)
            screenshot_vis = draw_players_roles_overlay(
                screenshot_vis,
                scene["allies_field"],
                scene["enemies_field"],
                scene["active_circle"],
            )
            screenshot_vis = draw_obstacle_circles(screenshot_vis, scene["obstacles_field"])

            if scene["players_result"]["players"] is None:
                warn_no_players("Auto mode")
                cv2.imshow("GraphBot", screenshot_vis)
                cv2.waitKey(1)
                sleep_key(0.5)
                continue

            if scene["active_norm"] is None:
                print(
                    "Auto mode: no active player detected "
                    f"(method={scene['active_result']['method']})."
                )
                cv2.imshow("GraphBot", screenshot_vis)
                cv2.waitKey(1)
                sleep_key(0.5)
                continue

            if not scene["enemy_circles_game"]:
                print("No enemies on the right side. Waiting...")
                cv2.imshow("GraphBot", screenshot_vis)
                cv2.waitKey(1)
                sleep_key(0.5)
                continue

            sym_result = search_best_symbolic_ga(
                x0=scene["active_norm"][0],
                y0=scene["active_norm"][1],
                enemies=scene["enemy_circles_game"],
                dangers=scene["obstacles_game"],
                allies=scene["allies_game"],
                prev_best=sym_best_genome,
                rng=sym_rng,
                step=0.05,
                budget_ms=320,
                max_evals=900,
                population_size=50,
            )
            sym_best_genome = sym_result["best_genome"]
            screenshot_vis = draw_symbolic_curve_on_field(
                screenshot_vis,
                sym_result["best_points"],
                field["width"],
                color=(0, 255, 0),
                thickness=2,
            )

            stats = sym_result["stats"]
            hits = len(sym_result["hit_enemy_ids"])
            stats_lines = [
                f"planner=SYM-GA  score={int(sym_result['best_score'])}  hits={hits}/{len(scene['enemy_circles_game'])}",
                f"evals={stats['evals']}  gen={stats['generations']}  time={stats['time_ms']}ms  ally_hits(best)={stats['ally_hits_best']}",
                f"reject danger={stats['rejects_danger']}  bounds={stats['rejects_bounds']}  invalid={stats['rejects_invalid']}",
            ]
            screenshot_vis = draw_symbolic_stats_overlay(screenshot_vis, stats_lines)

            players_count = len(scene["players_result"]["players"][0])
            summary = (
                f"players={players_count}  obstacles={len(scene['obstacles_field'])}  "
                f"enemies={hits}/{len(scene['enemy_circles_game'])}  "
                f"planner=SYM-GA  score={int(sym_result['best_score'])}  "
                f"active={scene['active_result']['method']}"
            )
            if summary != prev_summary:
                print(summary)
                prev_summary = summary

            formula = sym_result["formula"]
            if formula and formula != prev_text:
                print(formula)
                safe_copy(formula, prev_text)
                prev_text = formula

            cv2.imshow("GraphBot", screenshot_vis)
            cv2.waitKey(1)
            sleep_key(0.0)
            continue

        if not mode and auto_planner == 1:
            if poly_scene is None:
                if not refresh_field():
                    print("Graphwar window not found. Waiting...")
                    sleep_key(0.5)
                    continue

                screenshot = np.array(mss_.grab(field))
                screenshot_r = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)
                screenshot_bgr = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)

                players_cords = detect_players(screenshot_r)
                if players_cords is None:
                    warn_no_players("Auto mode")
                    cv2.imshow("GraphBot", screenshot_r)
                    cv2.waitKey(1)
                    sleep_key(0.5)
                    continue

                obstacle_result = find_all_obstacles(screenshot_bgr, obstacles_params)
                obstacles_field = obstacle_result["obstacles"]
                obstacles_game = field_obstacles_to_game(obstacles_field, field["width"])

                allies_field, enemies_field, active_circle = split_players_for_auto(players_cords)

                screenshot_base = cv2.cvtColor(screenshot_r, cv2.COLOR_GRAY2BGR)
                screenshot_base = draw_players_roles_overlay(
                    screenshot_base,
                    allies_field,
                    enemies_field,
                    active_circle,
                )
                screenshot_base = draw_obstacle_circles(screenshot_base, obstacles_field)

                if active_circle is None:
                    warn_no_players("Auto mode")
                    cv2.imshow("GraphBot", screenshot_base)
                    cv2.waitKey(1)
                    sleep_key(0.5)
                    continue

                if not enemies_field:
                    print("No enemies on the right side. Waiting...")
                    cv2.imshow("GraphBot", screenshot_base)
                    cv2.waitKey(1)
                    sleep_key(0.5)
                    continue

                active_norm = field_to_game(active_circle[0], active_circle[1])
                enemy_circles_game = sorted(
                    [field_circle_to_game(circle) for circle in enemies_field],
                    key=lambda circle: circle[0],
                )
                allies_game = [field_circle_to_game(circle) for circle in allies_field]
                poly_best_coeffs = None
                poly_scene = {
                    "screenshot_base": screenshot_base,
                    "players_count": len(players_cords[0]),
                    "obstacles_count": len(obstacles_field),
                    "active_norm": active_norm,
                    "enemy_circles_game": enemy_circles_game,
                    "allies_game": allies_game,
                    "obstacles_game": obstacles_game,
                }
                print("Polynomial planner: scene snapshot captured once and frozen.")

            screenshot_vis = poly_scene["screenshot_base"].copy()
            enemy_circles_game = poly_scene["enemy_circles_game"]
            allies_game = poly_scene["allies_game"]
            obstacles_game = poly_scene["obstacles_game"]
            active_norm = poly_scene["active_norm"]

            poly_result = search_best_polynomial(
                x0=active_norm[0],
                y0=active_norm[1],
                enemies=enemy_circles_game,
                dangers=obstacles_game,
                allies=allies_game,
                prev_best=poly_best_coeffs,
                rng=poly_rng,
                step=0.05,
                budget_ms=320,
                max_evals=900,
            )
            poly_best_coeffs = poly_result["best_coeffs"]

            if poly_result["current_points"]:
                screenshot_vis = draw_polynomial_curve_on_field(
                    screenshot_vis,
                    poly_result["current_points"],
                    field["width"],
                    color=(0, 220, 255),
                    thickness=1,
                )
            screenshot_vis = draw_polynomial_curve_on_field(
                screenshot_vis,
                poly_result["best_points"],
                field["width"],
                color=(0, 255, 0),
                thickness=2,
            )

            stats = poly_result["stats"]
            hits = len(poly_result["hit_enemy_ids"])
            danger_text = "9999+" if stats["rejects_danger"] >= 9999 else str(stats["rejects_danger"])
            bounds_text = "9999+" if stats["rejects_bounds"] >= 9999 else str(stats["rejects_bounds"])
            stats_lines = [
                f"planner=POLY  score={int(poly_result['best_score'])}  hits={hits}/{len(enemy_circles_game)}",
                    f"evals={stats['evals']}  time={stats['time_ms']}ms  ally_hits(best)={stats['ally_hits_best']}  near={stats['proximity_bonus_best']}",
                f"reject danger={danger_text}  bounds={bounds_text}  ally_penalized={stats['ally_penalty_hits']}",
            ]
            screenshot_vis = draw_polynomial_stats_overlay(screenshot_vis, stats_lines)

            summary = (
                f"players={poly_scene['players_count']}  obstacles={poly_scene['obstacles_count']}  "
                f"enemies={hits}/{len(enemy_circles_game)}  "
                f"planner=POLY  score={int(poly_result['best_score'])}"
            )
            if summary != prev_summary:
                print(summary)
                print(
                    "  evals="
                    f"{stats['evals']} reject_danger={danger_text} "
                    f"reject_bounds={bounds_text} ally_penalized={stats['ally_penalty_hits']}"
                )
                prev_summary = summary

            formula = poly_result["formula"]
            if formula and formula != prev_text:
                print(formula)
                safe_copy(formula, prev_text)
                prev_text = formula

            cv2.imshow("GraphBot", screenshot_vis)
            cv2.waitKey(1)
            sleep_key(0.0)
            continue

        if not refresh_field():
            print("Graphwar window not found. Waiting...")
            sleep_key(0.5)
            continue

        screenshot = np.array(mss_.grab(field))
        screenshot_r = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)
        screenshot_bgr = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)

        if not mode:
            players_cords = detect_players(screenshot_r)
            if players_cords is None:
                warn_no_players("Auto mode")
                cv2.imshow("GraphBot", screenshot_r)
                cv2.waitKey(1)
                sleep_key(0.5)
                continue

            obstacle_result = find_all_obstacles(screenshot_bgr, obstacles_params)
            obstacles_field = obstacle_result["obstacles"]
            obstacles_game = field_obstacles_to_game(obstacles_field, field["width"])

            allies_field, enemies_field, active_circle = split_players_for_auto(players_cords)

            screenshot_vis = cv2.cvtColor(screenshot_r, cv2.COLOR_GRAY2BGR)
            screenshot_vis = draw_players_roles_overlay(
                screenshot_vis,
                allies_field,
                enemies_field,
                active_circle,
            )
            screenshot_vis = draw_obstacle_circles(screenshot_vis, obstacles_field)

            if active_circle is None:
                warn_no_players("Auto mode")
                cv2.imshow("GraphBot", screenshot_vis)
                cv2.waitKey(1)
                sleep_key(0.5)
                continue

            if not enemies_field:
                print("No enemies on the right side. Waiting...")
                cv2.imshow("GraphBot", screenshot_vis)
                cv2.waitKey(1)
                sleep_key(0.5)
                continue

            active_norm = field_to_game(active_circle[0], active_circle[1])
            enemy_centers_norm = sorted(
                to_game_cords([(cx, cy) for cx, cy, _ in enemies_field]),
                key=lambda point: point[0],
            )

            path_waypoints, hit_enemies, skipped_enemies = build_enemy_chain_astar(
                enemy_centers_norm,
                obstacles_game,
                clearance=DEFAULT_CLEARANCE,
            )

            if enemy_centers_norm and astar_game(
                active_norm, tuple(enemy_centers_norm[0]), obstacles_game, clearance=DEFAULT_CLEARANCE
            ) is None:
                print("  note: active -> first formula point may need manual travel")

            if len(path_waypoints) < 2 and enemy_centers_norm:
                print("  A* failed — fallback: straight chain from 1st enemy")
                path_waypoints = [list(point) for point in enemy_centers_norm]

            screenshot_vis = draw_path_on_field(screenshot_vis, path_waypoints, field["width"])

            summary = (
                f"players={len(players_cords[0])}  obstacles={len(obstacles_field)}  "
                f"enemies={len(hit_enemies)}/{len(enemy_centers_norm)}  "
                f"waypoints={len(path_waypoints)}  planner=A*"
            )
            if summary != prev_summary:
                print(summary)
                if skipped_enemies:
                    print("  skipped (no detour):", skipped_enemies)
                prev_summary = summary

            formula = waypoints_to_formula(path_waypoints)
            if not formula:
                print("  no formula (need at least 2 waypoints)")
            elif formula != prev_text:
                print(formula)
                safe_copy(formula, prev_text)
                prev_text = formula

            cv2.imshow("GraphBot", screenshot_vis)
            cv2.waitKey(1)
            sleep_key(0.0 if auto_planner == 1 else 0.2)
        else:
            players_cords = detect_players(screenshot_r)
            if players_cords is None:
                warn_no_players("Click mode")
                cv2.imshow("GraphBot", screenshot_r)
                cv2.waitKey(1)
                sleep_key(0.5)
                continue

            _, _, active_player = separate(players_cords.tolist())

            clicks = collect_clicks()
            if not clicks:
                print("No clicks recorded. Press F3 to start, F4 when done.")
                continue

            print("Clicks (field px):", clicks)
            waypoints = process_clicks_to_waypoints(clicks)
            print("Formula anchor + path:", waypoints)
            formula = waypoints_to_formula(waypoints)
            print()
            print(formula)
            safe_copy(formula, prev_text)
            prev_text = formula

            safe_exit(0)

# =========================================================
# Main Program
# =========================================================
if __name__ == '__main__':
    left_mouse_key = 0x01
    start_key = 0x70 # f1
    exit_key = 0x71 # f2
    clicks_start = 0x72 # f3
    clicks_end = 0x73 # f4

    window_start_cords = (-7, 0)
    game_window_name = 'Graphwar'
    capture_margins = load_capture_margins()
    field = None
    exit_codes = {
        0: "Program has successfuly finished!",
        1: "No window with the game name has found :(\nMake sure Graphwar is running"
    }

    mode, auto_planner = setup()

    while not(is_key_pressed(start_key)):
        pass

    main()
