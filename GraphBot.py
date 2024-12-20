import cv2
import mss
import sys, ctypes
import time
import numpy as np

start_key = 0x70 # f1
exit_key = 0x71 # f2

def is_key_pressed(key):
    return ctypes.windll.user32.GetAsyncKeyState(key) & 0x8000 != 0

def sleep_key(sec):
    start_time = time.time()
    while True:
        if is_key_pressed(exit_key):
            sys.exit()
        
        current_time = time.time()
        elapsed_time = current_time - start_time

        if elapsed_time >= sec:
            break


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
        print('Oops!')
        return None

def detect_players(s_r):
    lower_bound = 50
    upper_bound = 250

    mask1 = cv2.inRange(s_r, lower_bound, 169)
    mask2 = cv2.inRange(s_r, 171, upper_bound)
    mask = cv2.bitwise_or(mask1, mask2)

    # mask = cv2.inRange(s_r, lower_bound, upper_bound)

    result = np.ones_like(s_r) * 255
    result[mask == 255] = 0
    blur_rate = 23
    result = cv2.GaussianBlur(result, (blur_rate, blur_rate), 0)

    detected_circles = cv2.HoughCircles(result,  
        cv2.HOUGH_GRADIENT, 1, minDist= 10, param1 = 150, 
        param2 = 10, minRadius = 4, maxRadius = 15) 
    
    # cv2.imshow('GraphBot', result)
    # cv2.waitKey(1)

    if detected_circles is not None: 
        detected_circles = np.uint16(np.around(detected_circles))
        return detected_circles
    else:
        print('Oops!')
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
    print(dist)
    return f'- {dist}*(abs(x - {p1[0]}) - abs(x - {p2[0]}))'.replace('- -', '+')

def main():
    mss_ = mss.mss()
    while not(is_key_pressed(exit_key)):
        screenshot = np.array(mss_.grab(field))
        screenshot_r = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)

        circles_cords = detect_black_circles(screenshot_r)
        
        # if circles_cords is not None:
        #     screenshot_r = draw_circles(circles_cords, screenshot_r)
        #     print(circles_cords)

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
        a = direct_line(active_norm, bad_guys_norm[0])
        for i in range(1, len(bad_guys)):
            a += '+' + direct_line(bad_guys_norm[i-1], bad_guys_norm[i])
        print(a)


        cv2.imshow('GraphBot', screenshot_r)
        cv2.waitKey(1)

if __name__ == '__main__':

    field = {'left': 13,
             'top': 52,
             'width': 772,
             'height': 452}
    
    while not(is_key_pressed(start_key)):
        pass

    main()
