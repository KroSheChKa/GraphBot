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

def to_game_cords(cord_list):
    new_list = []
    for i in cord_list:
        x = -25 + i[0]*50/field['width']
        y = 15 - i[1]*50/field['width']
        new_list.append([x, y])
    return new_list

def direct_line(p1, p2): #x1 y1   x2 y2
    dist = ((p1[1]-p2[1])/2)/(p2[0]-p1[0]+0.00000001)
    # print(dist)
    return f'{-dist}*(abs(x - {p1[0]}) - abs(x - {p2[0]}))'.replace('- -', '+ ')

# A function to collect the cords of the clicks 
# + subsctract the offsets
def collect_clicks():
    print("Click wherever on the game screen")
    clicks = []
    while not is_key_pressed(clicks_start):
        pass
    while not is_key_pressed(clicks_end):
        if is_key_pressed(left_mouse_key):
            (x, y) = win32gui.GetCursorPos()
            print((x, y))
            clicks.append((x - field['left'], y - field['top']))
            win32api.keybd_event(left_mouse_key, 0, win32con.KEYEVENTF_KEYUP, 0)
    return clicks

# Prevents unnesessary clipboard copying
def safe_copy(text, previous_text):
    if text != previous_text:
        pyperclip.copy(text)
        print("Safely copied!")

def setup():
    # I set the x position of the windown to -7 due to gap between the
    # left border of the game window and left side of the screen 
    move_window(game_window_name, window_start_cords[0], window_start_cords[1], 100, 100)
    
    # Handling user mode input
    # 0 - usual detection
    # 1 - clicks
    while True:
        print("Select the mode\n0 - automatic (straight lines)\n1 - clicks")
        mode = input()
        if len(mode) == 1 and (mode == '0' or mode == '1'):
            mode = int(mode)
            break
        print("Incorrect input!\n")
    return mode

def main():
    prev_text = ""
    mss_ = mss.mss()
    while not(is_key_pressed(exit_key)):
        screenshot = np.array(mss_.grab(field))
        screenshot_r = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)

        if not mode:
            circles_cords = detect_black_circles(screenshot_r)
            
            if circles_cords is not None:
                screenshot_r = draw_circles(circles_cords, screenshot_r)
                # print(circles_cords)

            players_cords = detect_players(screenshot_r)
            if players_cords is not None:
                screenshot_r = draw_circles(players_cords, screenshot_r)
                print(players_cords)

            good_guys, bad_guys, active_player = separate(players_cords.tolist())
            good_guys_norm = sorted(to_game_cords(good_guys), key= lambda x:x[0])
            bad_guys_norm = sorted(to_game_cords(bad_guys), key= lambda x:x[0])
            active_norm = (-25 + active_player[0]*50/field['width'], 15 - active_player[1]*50/field['width'])

            print()
            print("Players detected:", len(players_cords[0]))
            print("Active player:", active_player)
            print("Normalized:", active_norm)
            print("Good guys:", good_guys)
            print("Normalized:", good_guys_norm)
            print("Bad guys:", bad_guys)
            print("Normalized:", bad_guys_norm)


            print(bad_guys_norm[0])
            print()
            print()
            a = [direct_line(active_norm, bad_guys_norm[0])]

            for i in range(1, len(bad_guys)):
                a.append(direct_line(bad_guys_norm[i-1], bad_guys_norm[i]))
            
            formula = ' + '.join(a).replace('+ -', '- ')
            print(formula)
            safe_copy(formula, prev_text)
            prev_text = formula

            cv2.imshow("GraphBot", screenshot_r)
            cv2.waitKey(1)
        else:
            clicks = collect_clicks()
            print(clicks)
            clicks_norm = sorted(to_game_cords(clicks), key= lambda x:x[0])
            print(clicks_norm)

            players_cords = detect_players(screenshot_r)
            _, _, active_player = separate(players_cords.tolist())
            print("Active player:", active_player)
            active_norm = (-25 + active_player[0]*50/field['width'], 15 - active_player[1]*50/field['width'])
            print("Active player norm:", active_norm)
            
            a = [direct_line(active_norm, clicks_norm[0])]
            # print("HERE", a, active_norm, clicks_norm[0])

            for i in range(1, len(clicks)):
                a.append(direct_line(clicks_norm[i-1], clicks_norm[i]))
            print()

            formula = ' + '.join(a).replace('+ -', '- ')
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

    window_start_cords = (-7, 7)
    game_window_name = 'Graphwar'
    exit_codes = {
        0: "Program has successfuly finished!",
        1: "No window with the game name has found :(\nMake sure Graphwar is running"
    }

    mode = setup()

    field = {'left': 14,
             'top': 52,
             'width': 772,
             'height': 452}
    
    while not(is_key_pressed(start_key)):
        pass
    
    main()
