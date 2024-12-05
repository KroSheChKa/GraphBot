import cv2
import mss
import sys, ctypes
import time
import numpy

start_key = 0x70 # f1
exit_key = 0x71 # f2

def is_key_pressed(key):
    return ctypes.windll.user32.GetAsyncKeyState(key) & 0x8000 != 0

def sleep_key(sec):
    start_time = time.time()
    
    while True:
        # Key pressed during the loop? - exit the entire program
        if is_key_pressed(exit_key):
            sys.exit()
        
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        # If the time has run out, exit the loop
        if elapsed_time >= sec:
            break

def to_game_cords(x_y):
    print()
    new_x = -25 + x_y[0] / field['width'] * 50
    new_y = 15 - x_y[1] / field['height'] * 30
    return (new_x, new_y)

def main():
    # Reading the image
    active = cv2.imread(r'Assets\active.png', 0)
    active_w = active.shape[1]
    active_h = active.shape[0]
    
    enemy = cv2.imread(r'Assets\enemy.png', 0)
    enemy_w = enemy.shape[1]
    enemy_h = enemy.shape[0]

    mss_ = mss.mss()

    active_threshold = 0.72
    enemy_threshold = 0.64

    while not(is_key_pressed(exit_key)):
        screenshot = numpy.array(mss_.grab(field))
        screenshot_r = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)

        active_match = cv2.matchTemplate(screenshot_r, active, cv2.TM_CCOEFF_NORMED)
        _, active_max_val, _, active_max_loc = cv2.minMaxLoc(active_match)
        active_max_loc = (active_max_loc[0] + active_w/2, active_max_loc[1] + active_h/2)
        print(active_max_val, active_max_loc, to_game_cords(active_max_loc))

        enemy_match = cv2.matchTemplate(screenshot_r, enemy, cv2.TM_CCOEFF_NORMED)
        enemy_yloc, enemy_xloc = numpy.where(enemy_match > enemy_threshold)

        rectangles = []
        for (x, y) in zip(enemy_xloc, enemy_yloc):
            rectangles.append([x, y, enemy_w, enemy_h])
            rectangles.append([x, y, enemy_w, enemy_h])
        
        rectangles, _ = cv2.groupRectangles(rectangles, 1, 0.3)
        print(rectangles)

        enemies = []
        for i in range(len(rectangles)):
            x_y = to_game_cords((float(rectangles[i][0]) + enemy_w/2, float(rectangles[i][1]) + enemy_h/2))
            enemies.append(x_y)
            # min_x = min(min_x, x_y[0])
        
        print(enemies)

        # if max_val > threshold:
        #     print('YES')

        cv2.imshow('TimberBot', screenshot_r)
        cv2.waitKey(1)

if __name__ == '__main__':

    field = {'left': 13,
             'top': 52,
             'width': 772,
             'height': 452}
    
    while not(is_key_pressed(start_key)):
        pass

    # Run the code
    main()
    