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
        if is_key_pressed(exit_key):
            sys.exit()
        
        current_time = time.time()
        elapsed_time = current_time - start_time

        if elapsed_time >= sec:
            break



def to_game_cords(a):
    temp = a / field['height'] * 30
    return temp

def get_enemies(mss_, threshold, enemy, w, h):
    screenshot = numpy.array(mss_.grab(field))
    screenshot_r = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)

    enemy_match = cv2.matchTemplate(screenshot_r, enemy, cv2.TM_CCOEFF_NORMED)
    enemy_yloc, enemy_xloc = numpy.where(enemy_match > threshold)

    rectangles = []
    for (x, y) in zip(enemy_xloc, enemy_yloc):
        rectangles.append([x, y, w, h])
        rectangles.append([x, y, w, h])
    
    rectangles, _ = cv2.groupRectangles(rectangles, 1, 0.3)
    print(rectangles)
    enemies = []
    for i in range(len(rectangles)):
        # x = to_game_cords(float(rectangles[i][0]) + w/2)
        # y = 15 - to_game_cords(float(rectangles[i][1]) + h/2)
        enemies.append((w/2+float(rectangles[i][0])-1, h/2+float(rectangles[i][1])-1))
    return enemies

def detect_black_circles(s_r):
    s_r = cv2.GaussianBlur(s_r, (3, 3), 0)

    lower_bound = 0
    upper_bound = 20

    # Создаем маску: пиксели в диапазоне [240, 255] будут оставлены, остальные заменим на белый
    mask = cv2.inRange(s_r, lower_bound, upper_bound)

    result = numpy.ones_like(s_r) * 255  # Изначально делаем все белыми
    result[mask == 255] = 0  # Пиксели, попадающие в маску, делаем черными
    result = cv2.GaussianBlur(result, (13, 13), 0)

    # Apply Hough transform on the blurred image. 
    detected_circles = cv2.HoughCircles(result,  
                    cv2.HOUGH_GRADIENT, 1, minDist= 15, param1 = 230, 
                param2 = 28, minRadius = 1, maxRadius = 200) 
    
    if detected_circles is not None: 
        detected_circles = numpy.uint16(numpy.around(detected_circles)) 
        return detected_circles
    else:
        print('Oops!')
        return None
    
def draw_circles(detected_circles, screenshot_r):
    for pt in detected_circles[0,:]: 
            a, b, r = pt[0], pt[1], pt[2] 
            cv2.circle(screenshot_r, (a, b), r, (100, 0, 0), 2) 
            cv2.circle(screenshot_r, (a, b), 1, (255, 0, 0), 3) 
    return screenshot_r

def draw_circles1(detected_circles, screenshot_r):
    r = 10
    for pt in detected_circles: 
            a, b = int(pt[0]), int(pt[1])
            print(a, b, r)
            cv2.circle(screenshot_r, (a, b), r, (150, 0, 0), 2) 
    return screenshot_r

def main():
    active = cv2.imread(r'Assets\active.png', 0)
    active_w = active.shape[1]
    active_h = active.shape[0]
    
    enemy = cv2.imread(r'Assets\enemy.png', 0)
    enemy_w = enemy.shape[1]
    enemy_h = enemy.shape[0]

    mss_ = mss.mss()

    active_threshold = 0.72
    enemy_threshold = 0.65

    while not(is_key_pressed(exit_key)):
        screenshot = numpy.array(mss_.grab(field))
        screenshot_r = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)

        active_match = cv2.matchTemplate(screenshot_r, active, cv2.TM_CCOEFF_NORMED)
        _, active_max_val, _, active_max_loc = cv2.minMaxLoc(active_match)
        active_max_loc = (active_max_loc[0] + active_w/2, active_max_loc[1] + active_h/2)
        print(active_max_val, active_max_loc)

        circles_cords = detect_black_circles(screenshot_r)
        if circles_cords is not None:
            screenshot_r = draw_circles(circles_cords, screenshot_r)
            print(circles_cords)

        # enemies = get_enemies(mss_, enemy_threshold, enemy, enemy_w, enemy_h)
        # print(enemies)
        # screenshot_r = draw_circles1(enemies, screenshot_r)
        

        cv2.imshow('GraphBot', screenshot_r)
        cv2.waitKey(1)

if __name__ == '__main__':

    field = {'left': 13,
             'top': 52,
             'width': 772,
             'height': 452}

    r_field = {'left': 399,
             'top': 52,
             'width': 386,
             'height': 452}

    l_field = {'left': 13,
             'top': 52,
             'width': 386,
             'height': 452}
    
    while not(is_key_pressed(start_key)):
        pass

    main()
