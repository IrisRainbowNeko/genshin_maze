import pickle
import numpy as np
import cv2
import op
import winsound
import keyboard
import pyautogui as pag
import time
from maze_generate import *
from utils import *

flag=[True]

def stop(x):
    if x.event_type == 'down' and x.name == 'u':
        flag[0]=False

def parse_args():
    def str2size(v: str):
        return tuple(int(x) for x in v.strip().split('x'))

    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--size', default=(40,40), type=str2size)
    parser.add_argument('--name', default='maze3', type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    cap_init()

    args = parse_args()
    #sx,sy=14,14

    keyboard.hook(stop)

    img_rot=cv2.imread('./rot_edge.png', -1)

    winsound.Beep(600, 500)
    keyboard.wait(hotkey='r')
    min_x, min_y = pag.position()

    winsound.Beep(600, 500)
    keyboard.wait(hotkey='r')
    max_x, max_y = pag.position()

    dx=(max_x-min_x)/(args.size[0]-1)
    dy=(max_y-min_y)/(args.size[0]-1)

    winsound.Beep(700, 500)
    keyboard.wait(hotkey='r')
    sx, sy = pag.position()
    sx = round((sx-min_x)/dx)
    sy = round((sy-min_y)/dy)

    with open(f"{args.name}.pkl", "rb") as f:
        data=pickle.load(f)
        walls=data['wall']
        coins=data['coin']

    winsound.Beep(600, 500)
    keyboard.wait(hotkey='r')
    winsound.Beep(800, 500)

    time.sleep(0.3)

    wall_center_idx=None
    for i, wall in enumerate(walls):
        rect = wall.get_rect()
        for y in range(rect[1], rect[3]+1):
            for x in range(rect[0], rect[2]+1):
                if x==sx and y==sy:
                    wall_center_idx=i
                    break
    if wall_center_idx is not None:
        wall_center=walls.pop(wall_center_idx)
        walls.append(wall_center)

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

        op.move((min_x+dx*sx+dx*0.1, min_y+dy*sy+dy*0.1))
        op.press((min_x+dx*sx+dx*0.1, min_y+dy*sy+dy*0.1))
        op.move((min_x+dx*cx+dx*0.1, min_y+dy*cy+dy*0.1))
        op.release((min_x + dx * cx+dx*0.1, min_y + dy * cy+dy*0.1))

        op.click((2458, 1038))

    #coins
    op.click((167, 1114))

    for coin in coins:
        if not flag[0]:
            break

        cx, cy=coin.x+1, coin.y+1
        op.click((410, 1273))
        #time.sleep(0.2)

        op.move((min_x+dx*sx+dx*0.1, min_y+dy*sy-dy*0))
        op.press((min_x+dx*sx+dx*0.1, min_y+dy*sy-dy*0))
        op.move((min_x+dx*cx+dx*0.1, min_y+dy*cy-dy*0))
        op.release((min_x + dx * cx+dx*0.1, min_y + dy * cy-dy*0))

        op.click((2458, 976))