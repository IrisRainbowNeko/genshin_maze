import pickle
import numpy as np
import cv2
import op
import winsound
import keyboard
import pyautogui as pag
import time
from maze_generate import Wall
from utils import *

flag=[True]

def stop(x):
    if x.event_type == 'down' and x.name == 'u':
        flag[0]=False

if __name__ == '__main__':

    keyboard.hook(stop)

    img_rot=cv2.imread('./rot_edge.png', -1)

    winsound.Beep(600, 500)
    keyboard.wait(hotkey='r')
    min_x, min_y = pag.position()

    winsound.Beep(600, 500)
    keyboard.wait(hotkey='r')
    max_x, max_y = pag.position()

    dx=(max_x-min_x)/(30-1)
    dy=(max_y-min_y)/(30-1)

    with open(r"maze1.pkl", "rb") as f:
        walls=pickle.load(f)['maze']

    winsound.Beep(600, 500)
    keyboard.wait(hotkey='r')
    winsound.Beep(800, 500)

    time.sleep(0.3)

    for wall in walls:
        if not flag[0]:
            break

        cx, cy, rot=wall.get_center()
        op.click((674, 1273))
        time.sleep(0.2)

        im_cap=cap(region=[2397,346,129,752])
        gray = cv2.cvtColor(im_cap, cv2.COLOR_RGB2GRAY)
        edge_output = cv2.Canny(gray, 50, 150)
        pos=match_img(edge_output, img_rot)

        pos_y=346+pos[-1]

        for _ in range(rot):
            op.click((2459,pos_y))
            #time.sleep(0.1)

        op.move((min_x+dx*14+dx*0.1, min_y+dy*14+dy*0.1))
        #time.sleep(0.1)
        op.press((min_x+dx*14+dx*0.1, min_y+dy*14+dy*0.1))
        #time.sleep(0.1)
        op.move((min_x+dx*cx+dx*0.1, min_y+dy*cy+dy*0.1))
        #time.sleep(0.1)
        op.release((min_x + dx * cx+dx*0.1, min_y + dy * cy+dy*0.1))
        #time.sleep(0.1)

        #time.sleep(0.2)
        op.click((2458, 1038))
        #time.sleep(0.1)