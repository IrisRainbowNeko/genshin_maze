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
    parser.add_argument('--size', default=(40,40,25-5), type=str2size)
    parser.add_argument('--name', default='maze3D_2', type=str)

    args = parser.parse_args()
    return args

def move_to(sx, sy, ex, ey):
    op.move((sx, sy))
    op.press((sx, sy))
    op.move((ex, ey))
    op.release((ex, ey))

def get_sc_rate(z, z_max):
    dz=0
    for i in range(1,z+1):
        dz+=np.cos(np.arctan((20.5*np.sqrt(3)-i)/20.5))

    dz_max = 0
    for i in range(1,z_max+1):
        dz_max+=np.cos(np.arctan((20.5*np.sqrt(3)-i)/20.5))
    return dz/dz_max

if __name__ == '__main__':
    cap_init()

    args = parse_args()
    #sx,sy=14,14

    keyboard.hook(stop)

    img_rot=cv2.imread('./rot_edge.png', -1)
    img_x=cv2.imread('./x_edge.png', -1)

    with open(f"{args.name}.pkl", "rb") as f:
        data=pickle.load(f)
        walls=data['wall']
        coins=data['coin']
        print(len(coins))
        maze=data['maze']

    #底层坐标范围
    winsound.Beep(600, 500)
    keyboard.wait(hotkey='r')
    min_x_b, min_y_b = pag.position()

    winsound.Beep(600, 500)
    keyboard.wait(hotkey='r')
    max_x_b, max_y_b = pag.position()

    dx=(max_x_b-min_x_b)/(args.size[0])
    dy=(max_y_b-min_y_b)/(args.size[1])

    #底层放置中心坐标
    winsound.Beep(800, 500)
    keyboard.wait(hotkey='r')
    sx, sy = pag.position()
    sx = round((sx-min_x_b)/dx)
    sy = round((sy-min_y_b)/dy)

    #顶层坐标范围
    winsound.Beep(700, 500)
    keyboard.wait(hotkey='r')
    min_x_t, min_y_t = pag.position()

    winsound.Beep(700, 500)
    keyboard.wait(hotkey='r')
    max_x_t, max_y_t = pag.position()

    vx_d = min_x_b - min_x_t
    vy_d = min_y_b - min_y_t

    sc_rate_x = (max_x_t - min_x_t) / (max_x_b - min_x_b)-1
    sc_rate_y = (max_y_t - min_y_t) / (max_y_b - min_y_b)-1


    p_sx, p_sy=min_x_b+dx*sx+dx/2, min_y_b+dy*sy+dy/2

    #按高度顺序划分
    walls_dict={}
    for wall in walls:
        p1, p2 = wall.get_cube()
        z=p1[2]
        if p1[2] not in walls_dict:
            walls_dict[p1[2]]=[]
        if sx in range(p1[0], p2[0] + 1) and sy in range(p1[1], p2[1] + 1):
            continue
        if sx in range(p1[0], p2[0] + 1) and sy+1 in range(p1[1], p2[1] + 1):
            continue
        walls_dict[p1[2]].append(wall)

    '''walls_dict_tmp = walls_dict
    walls_dict = {}
    for k in sorted(walls_dict_tmp):
        walls_dict[k]=walls_dict_tmp[k]'''

    coins_dict = {}
    for coin in coins:
        z = coin[2]
        if z not in coins_dict:
            coins_dict[z] = []
        if sx in range(coin[0], coin[0] + 3) and sy in range(coin[1], coin[1] + 3):
            continue
        if sx in range(coin[0], coin[0] + 3) and sy + 1 in range(coin[1], coin[1] + 3):
            continue
        coins_dict[z].append(coin)

    winsound.Beep(600, 500)
    keyboard.wait(hotkey='r')
    winsound.Beep(1000, 500)

    time.sleep(0.3)

    '''wall_center_idx=None
    for i, wall in enumerate(walls):
        p1, p2 = wall.get_cube()
        if sx in range(p1[0], p2[0]+1) and sy in range(p1[1], p2[1]+1) and 0==p1[2]:
            wall_center_idx=i
            break
    if wall_center_idx is not None:
        wall_center=walls.pop(wall_center_idx)
        walls.append(wall_center)'''

    last_wz=0
    for wz in range(args.size[2]+5):
        if not flag[0]:
            break

        wall_list =walls_dict[wz] if wz in walls_dict else []
        coin_list =coins_dict[wz] if wz in coins_dict else []

        #sc_rate = wz / args.size[2]
        sc_rate = get_sc_rate(wz, args.size[2])
        min_x_sc, min_y_sc = min_x_b - vx_d * sc_rate, min_y_b - vy_d * sc_rate
        dx_sc, dy_sc = dx * (sc_rate * sc_rate_x + 1), dy * (sc_rate * sc_rate_y + 1)

        op.click((383, 1114))

        sy_off = int(wz>=13)

        if wz!=last_wz:
            last_sc_rate = get_sc_rate(last_wz, args.size[2])
            last_min_x_sc, last_min_y_sc = min_x_b - vx_d * last_sc_rate, min_y_b - vy_d * last_sc_rate
            last_dx_sc, last_dy_sc = dx * (last_sc_rate * sc_rate_x + 1), dy * (last_sc_rate * sc_rate_y + 1)

            op.click((674, 1273))
            move_to(last_min_x_sc + last_dx_sc * sx + last_dx_sc / 2, last_min_y_sc + last_dy_sc * (sy+sy_off) + last_dy_sc / 2,
                    last_min_x_sc + last_dx_sc * sx + last_dx_sc / 2, last_min_y_sc + last_dy_sc * 0 + last_dy_sc / 2)
            op.type_key('left shift')
            move_to(last_min_x_sc + last_dx_sc * sx + last_dx_sc / 2, last_min_y_sc + last_dy_sc * 0 + last_dy_sc / 2,
                    min_x_sc + dx_sc * sx + dx_sc / 2, min_y_sc + dy_sc * 0 + dy_sc / 2)
            op.type_key('left shift')

            time.sleep(0.1)
            im_cap = cap(region=[2397, 346, 129, 752])
            gray = cv2.cvtColor(im_cap, cv2.COLOR_RGB2GRAY)
            edge_output = cv2.Canny(gray, 50, 150)
            pos = match_img(edge_output, img_x)
            pos_y = 346 + pos[-1] + 30
            op.click((2458, pos_y))

        sy_off = int(wz >= 12)

        for wall in wall_list:
            if not flag[0]:
                break

            p1, p2 = wall.get_cube()

            cx, cy = (p1[:2] + p2[:2]) / 2
            if p1[2]==p2[2]: #平台
                op.click((410, 1273)) #位置2
            else:
                op.click((674, 1273))
                if p1[0] == p2[0]:  # 旋转
                    time.sleep(0.2)
                    im_cap = cap(region=[2397, 346, 129, 752])
                    gray = cv2.cvtColor(im_cap, cv2.COLOR_RGB2GRAY)
                    edge_output = cv2.Canny(gray, 50, 150)
                    pos = match_img(edge_output, img_rot)
                    pos_y = 346 + pos[-1] + 30
                    op.click((2458, pos_y))

            move_to(min_x_sc + dx_sc * sx + dx_sc / 2, min_y_sc + dy_sc * (sy+sy_off) + dy_sc / 2,
                    min_x_sc + dx_sc * cx + dx_sc / 2, min_y_sc + dy_sc * cy + dy_sc / 2)

            op.click((2458, 1038))

        # coins
        op.click((167, 1114))

        for coin in coin_list:
            if not flag[0]:
                break

            cx, cy = coin[0] + 1, coin[1] + 0
            op.click((410, 1273))
            # time.sleep(0.2)

            move_to(min_x_sc + dx_sc * sx + dx_sc / 2, min_y_sc + dy_sc * sy + dy_sc / 2,
                    min_x_sc + dx_sc * cx + dx_sc / 2, min_y_sc + dy_sc * cy + dy_sc / 2)

            op.click((2458, 976))

        last_wz = wz

    #coins
    '''op.click((167, 1114))

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

        op.click((2458, 976))'''