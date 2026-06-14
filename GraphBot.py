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
import numpy as np
import win32api, win32con, win32gui
import pyperclip

from window_capture import find_game_window, get_capture_field, load_capture_margins
from detection import find_all_obstacles, load_obstacles_params
from avoidance import field_obstacles_to_game, DEFAULT_CLEARANCE
from pathfinding import astar_game, build_enemy_chain_astar, draw_path_on_field

# =========================================================
# Functions
# =========================================================
def is_key_pressed(key):
    return ctypes.windll.user32.GetAsyncKeyState(key) & 0x8000 != 0

# A function to print out the status of finishing of the program and exiting it
def safe_exit(exit_code):
    print(exit_codes[exit_code])
    sys.exit()

# A function to move the game windown to a certain position
def move_window(window_title, x, y, width, height):
    # Trying to get the grapwar window
    hwnd = win32gui.FindWindow(None, window_title)

    # If this code exited on that moment, it means you need to pass
    # the first arg (name of the game window) as it typed in the 
    # upper part of the game screen 
    if hwnd == 0:
        safe_exit(1)
        
    # As function MoveWindow gets 6 args including the width and height
    # of the window, which we do not care, just set as it is
    rect = win32gui.GetWindowRect(hwnd)
    width = rect[2] - rect[0]
    height = rect[3] - rect[1]

    # win32gui.SetActiveWindow(hwnd)
    win32gui.MoveWindow(hwnd, x, y, width, height, True)
    print(f"Window '{window_title}' moved to ({x}, {y}).")

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


def direct_line(p1, p2):
    x1, y1 = fmt_game(p1[0]), fmt_game(p1[1])
    x2, y2 = fmt_game(p2[0]), fmt_game(p2[1])
    dx = x2 - x1
    if abs(dx) < 1e-12:
        dx = fmt_game(vertical_eps(y1, y2)) if y1 != y2 else VERTICAL_MIN_EPS
        x2 = fmt_game(x1 + dx)
    dist = fmt_game(-((y1 - y2) / 2) / dx)
    return f"{dist}*(abs(x - {x1}) - abs(x - {x2}))".replace("- -", "+ ")


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
    return " + ".join(parts).replace("+ -", "- ")


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
        "  - Is the capture region correct? (preview_capture.py)\n"
        "  - Tune detection (calibrate_players.py / calibrate_active.py)"
    )

# Prevents unnesessary clipboard copying
def safe_copy(text, previous_text):
    if text != previous_text:
        pyperclip.copy(text)
        print("Safely copied!")

def setup():
    global field

    hwnd = find_game_window(game_window_name)
    if hwnd is None:
        safe_exit(1)

    field = get_capture_field(hwnd, capture_margins)
    print(f"Capture region: {field}")
    print(f"Margins (client-relative): {capture_margins}")

    # Handling user mode input
    # 0 - usual detection
    # 1 - clicks
    while not(is_key_pressed(exit_key)):
        print("Select the mode\n0 - automatic (straight lines)\n1 - clicks")
        mode = input()
        if len(mode) == 1 and (mode == '0' or mode == '1'):
            mode = int(mode)
            break
        print("Incorrect input!\n")
    return mode

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
    mss_ = mss.mss()
    print("Auto mode: F2 = quit. Pathfinding: A*. Updates every ~1 s.")
    while not(is_key_pressed(exit_key)):
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

            obstacle_result = find_all_obstacles(screenshot_bgr, load_obstacles_params())
            obstacles_field = obstacle_result["obstacles"]
            obstacles_game = field_obstacles_to_game(obstacles_field, field["width"])

            screenshot_vis = cv2.cvtColor(screenshot_r, cv2.COLOR_GRAY2BGR)
            screenshot_vis = draw_circles(players_cords, screenshot_vis)
            screenshot_vis = draw_obstacle_circles(screenshot_vis, obstacles_field)

            good_guys, bad_guys, active_player = separate(players_cords.tolist())
            if not bad_guys:
                print("No enemies on the right side. Waiting...")
                cv2.imshow("GraphBot", screenshot_vis)
                cv2.waitKey(1)
                sleep_key(0.5)
                continue

            bad_guys_norm = sorted(to_game_cords(bad_guys), key=lambda x: x[0])
            active_norm = field_to_game(active_player[0], active_player[1])

            path_waypoints, hit_enemies, skipped_enemies = build_enemy_chain_astar(
                bad_guys_norm,
                obstacles_game,
                clearance=DEFAULT_CLEARANCE,
            )

            if bad_guys_norm and astar_game(
                active_norm, tuple(bad_guys_norm[0]), obstacles_game, clearance=DEFAULT_CLEARANCE
            ) is None:
                print("  note: active -> first formula point may need manual travel")

            if len(path_waypoints) < 2 and bad_guys_norm:
                print("  A* failed — fallback: straight chain from 1st enemy")
                path_waypoints = [list(p) for p in bad_guys_norm]

            screenshot_vis = draw_path_on_field(screenshot_vis, path_waypoints, field["width"])

            summary = (
                f"players={len(players_cords[0])}  obstacles={len(obstacles_field)}  "
                f"enemies={len(hit_enemies)}/{len(bad_guys_norm)}  "
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
            sleep_key(1.0)
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

    mode = setup()

    while not(is_key_pressed(start_key)):
        pass

    main()
