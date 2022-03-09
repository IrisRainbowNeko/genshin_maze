import numpy as np
import random
from typing import List
from copy import deepcopy
import cv2

import networkx as nx
import argparse
import pickle
from utils import *

class Pos:
    def __init__(self,x: int,y: int):
        self.data=np.array([x,y])

    @property
    def x(self):
        return self.data[0]

    @property
    def y(self):
        return self.data[1]

    def __add__(self, other):
        return Pos(self.x+other.x, self.y+other.y)

    def __repr__(self):
        return f'Pos({self.x},{self.y})'

    def __eq__(self, other):
        return (self.data==other.data).all()

    def __hash__(self):
        return hash(tuple(self.data.tolist()))

class Wall(Pos):
    def __init__(self, x: int, y: int, len: int, dir: int):
        #super().__init__(x, y)
        self.data=np.array([x,y,len,dir])

    @property
    def len(self):
        return self.data[2]

    @property
    def dir(self):
        return self.data[3]

    def get_rect(self):
        if self.dir==0: #上
            return [self.x, self.y-(self.len-1), self.x, self.y]
        elif self.dir==1: #右
            return [self.x, self.y, self.x+(self.len-1), self.y]
        elif self.dir==2: #下
            return [self.x, self.y, self.x, self.y+(self.len-1)]
        elif self.dir==3: #左
            return [self.x-(self.len-1), self.y, self.x, self.y]

    def get_center(self):
        rect=self.get_rect()
        return (rect[0]+rect[2])//2, (rect[1]+rect[3])//2, 1-self.dir%2

    def __repr__(self):
        return f'Wall({self.x},{self.y} - {self.len},{self.dir})'

class Cube(Pos):
    rot_mats_dirs=[
        np.eye(3),
        rotate_mat(az, -90),
        rotate_mat(az, -180),
        rotate_mat(az, -270),
        rotate_mat(ax, 90),
        rotate_mat(ax, -90),
    ]

    rot_mats_rots = [
        np.eye(3),
        rotate_mat(ay, 90),
        rotate_mat(ay, 180),
        rotate_mat(ay, 270),
    ]

    def __init__(self, x: int, y: int, z: int, l: int, w: int, h: int, dir: int, rot: int):
        self.data=np.array([x,y,z, l,w,h, dir,rot], dtype=int)

    @property
    def z(self):
        return self.data[2]

    @property
    def l(self):
        return self.data[3]

    @property
    def w(self):
        return self.data[4]

    @property
    def h(self):
        return self.data[5]

    @property
    def dir(self):
        return self.data[6]

    @property
    def rot(self):
        return self.data[7]

    def get_cube(self): #朝上或向前 rot为0, dir:[↑→↓←上下]
        # dir=0,rot=0 向前朝下
        p1, p0=self.data[:3], np.array((self.l-1, -self.w+1, -self.h+1))
        #p2 = np.dot(np.dot(p0[np.newaxis,:], Cube.rot_mats_dirs[self.dir]), Cube.rot_mats_rots[self.rot])[:]+p1
        p2 = np.dot(np.dot(p0[np.newaxis,:], Cube.rot_mats_rots[self.rot]), Cube.rot_mats_dirs[self.dir])[:]+p1

        p1, p2 = np.minimum(p1, p2), np.maximum(p1, p2)
        return np.round(p1[0]).astype(int), np.round(p2[0]).astype(int)

    def get_center(self):
        p1, p2=self.get_cube()
        return (p1+p2)/2

class MazeBase:
    def __init__(self, size):
        self.size = np.array(size)

        self.wall_list = []
        self.placed_wall_list = []
        self.placed_coin_list = []

    def can_place(self, wall):
        return True

    def place_wall(self, wall):
        pass

    def get_next_states_list(self, wall):
        pass

    def generate(self):
        pass

    def generate_coins(self):
        pass

    def draw(self, size):
        pass

class Maze2D(MazeBase):
    def __init__(self, size=(20, 20), wall_len=5):
        super().__init__(size)
        self.wall_len=wall_len
        self.maze = np.zeros(size[::-1], dtype=np.uint8)
        self.groups = np.zeros(size[::-1], dtype=np.uint8)
        self.groups_data=[0]

        self.dir_zero = [3,0,1,2]

    def can_place(self, wall):
        rect=np.array(wall.get_rect())

        if (rect[:2]<0).any() or (rect[2:]>=self.size).any():
            return False

        pads = np.array([-1, -1, 1, 1])
        pads[:2]=np.maximum(rect[:2]+pads[:2], 0)-rect[:2]
        pads[2:]=np.minimum(rect[2:]+pads[2:], self.size-1)-rect[2:]
        pads[self.dir_zero[wall.dir]]=0

        rect+=pads
        return not self.maze[rect[1]:rect[3]+1, rect[0]:rect[2]+1].any()

    def place_wall(self, wall):
        rect = np.array(wall.get_rect())

        #成环检测
        if wall.dir==0:
            px, py = rect[0], rect[3]+1
        elif wall.dir == 1:
            px, py = rect[0]-1, rect[1]
        elif wall.dir == 2:
            px, py = rect[0], rect[1]-1
        elif wall.dir == 3:
            px, py = rect[2]+1, rect[3]

        if px not in range(0,self.size[0]) or py not in range(0,self.size[1]) or self.groups[py, px]==0:
            self.groups[rect[1]:rect[3] + 1, rect[0]:rect[2] + 1] = len(self.groups_data)
            self.groups_data.append(0)
        else:
            if self.groups_data[self.groups[py, px]]==1:
                return
            else:
                self.groups[rect[1]:rect[3] + 1, rect[0]:rect[2] + 1] = self.groups[py, px]
                if (rect[:2]<=0).any() or (rect[2:]>=self.size-1).any():
                    self.groups_data[self.groups[py, px]] = 1

        self.maze[rect[1]:rect[3]+1, rect[0]:rect[2]+1]=1
        self.placed_wall_list.append(wall)

    def get_next_states_list(self, wall):
        wall_list=[]
        def add_dirs(x, y):
            for d in range(4):
                wall_list.append(Wall(x, y, self.wall_len, d))

        rect = wall.get_rect()
        if rect[0]==rect[2]:
            for y in range(rect[1], rect[3]+1):
                add_dirs(rect[0]-1, y)
                add_dirs(rect[0]+1, y)
            add_dirs(rect[0], rect[1]-1)
            add_dirs(rect[0], rect[3]+1)
        else:
            for x in range(rect[0], rect[2]+1):
                add_dirs(x, rect[1]-1)
                add_dirs(x, rect[1]+1)
            add_dirs(rect[0]-1, rect[1])
            add_dirs(rect[2]+1, rect[1])
        wall_list=list(filter(lambda w: self.can_place(w), wall_list))
        return wall_list

    def generate(self):
        def inter_gene():
            while len(self.wall_list)>0:
                wall=self.wall_list.pop(random.randint(0,len(self.wall_list)-1))
                if self.can_place(wall):
                    self.place_wall(wall)
                    self.wall_list.extend(self.get_next_states_list(wall))
        for i in range(3):
            self.wall_list.append(Wall(np.random.randint(0,self.size[0]-1), np.random.randint(0,self.size[1]-1), self.wall_len, np.random.randint(0, 4)))
        inter_gene()
        for i in range(100):
            self.wall_list.append(Wall(np.random.randint(0,self.size[0]-1), np.random.randint(0,self.size[1]-1), self.wall_len, np.random.randint(0, 4)))
        inter_gene()

        #self.maze[0,:]=1
        #self.maze[:,0]=1
        #self.maze[-1, :] = 1
        #self.maze[:, -1] = 1

    def generate_coins(self):
        available_list=[]
        for y in range(1,self.size[1]-2):
            for x in range(1,self.size[0]-2):
                if (self.maze[y:y+2, x:x+2]==0).all():
                    for coin in available_list:
                        if not (coin.x>=x+2 or coin.x+2<=x or coin.y>=y+2 or coin.y+2<=y):
                            break
                    else:
                        available_list.append(Pos(x,y))

        self.placed_coin_list=random.sample(available_list, min(10, len(available_list)))

    def draw(self, size=(400,400)):
        img=np.stack([self.maze,self.maze,self.maze], axis=-1)

        yellow=np.array([0,1,1])
        for coin in self.placed_coin_list:
            img[coin.y:coin.y+2,coin.x:coin.x+2]=yellow

        img=img*255
        img=cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)

        return img

class Maze3D(MazeBase):
    def __init__(self, size=(40, 40, 18), wall_size=(1, 5, 5)):
        super().__init__(size)
        self.wall_size=wall_size
        self.maze = np.zeros(size, dtype=np.uint8)

        self.dir_zero=[4,0,1,3,2,5]

    def can_place(self, wall: Cube):
        p1, p2 = wall.get_cube()

        if (p1 < 0).any() or (p2 >= self.size).any():
            return False

        pads=np.array([-1,-1,-4, 1,1,4])
        pads[:3] = np.maximum(p1 + pads[:3], 0) - p1
        pads[3:] = np.minimum(p2 + pads[3:], self.size - 1) - p2
        pads[self.dir_zero[wall.dir]]=0
        if p1[2]!=p2[2]:
            pads[2] = 0
            pads[5] = 0

        p1 += pads[:3]
        p2 += pads[3:]
        return not self.maze[p1[0]:p2[0]+1, p1[1]:p2[1]+1, p1[2]:p2[2]+1].any()

    def place_wall(self, wall):
        p1, p2 = wall.get_cube()
        self.maze[p1[0]:p2[0]+1, p1[1]:p2[1]+1, p1[2]:p2[2]+1]=1
        self.placed_wall_list.append(wall)

    def get_next_states_list(self, wall: Cube):
        wall_list = []

        def add_dirs(x, y, z, dir):
            for r in range(4):
                wall_list.append(Cube(x, y, z, *self.wall_size, dir, r))

        p1, p2 = wall.get_cube()
        if p1[0] == p2[0]:
            for y in range(p1[1], p2[1] + 1):
                for z in range(p1[2], p2[2] + 1):
                    add_dirs(p1[0]-1, y, z, 3)
                    add_dirs(p1[0]+1, y, z, 1)
        elif p1[1] == p2[1]:
            for x in range(p1[0], p2[0] + 1):
                for z in range(p1[2], p2[2] + 1):
                    add_dirs(x, p1[1]-1, z, 0)
                    add_dirs(x, p1[1]+1, z, 2)
        elif p1[2] == p2[2]:
            for x in range(p1[0], p2[0] + 1):
                for y in range(p1[1], p2[1] + 1):
                    add_dirs(x, y, p1[2]-1, 4)
                    add_dirs(x, y, p1[2]+1, 5)
        wall_list = list(filter(lambda w: self.can_place(w), wall_list))
        return wall_list

    def generate(self):
        def inter_gene():
            while len(self.wall_list)>0:
                wall=self.wall_list.pop(random.randint(0,len(self.wall_list)-1))
                if self.can_place(wall):
                    self.place_wall(wall)
                    self.wall_list.extend(self.get_next_states_list(wall))
        for i in range(3):
            self.wall_list.append(Cube(np.random.randint(0,self.size[0]-1), np.random.randint(0,self.size[1]-1), np.random.randint(0,self.size[2]-1), *self.wall_size, np.random.randint(0, 6), np.random.randint(0, 4)))
        inter_gene()
        for i in range(1000):
            self.wall_list.append(Cube(np.random.randint(0,self.size[0]-1), np.random.randint(0,self.size[1]-1), np.random.randint(0,self.size[2]-1), *self.wall_size, np.random.randint(0, 6), np.random.randint(0, 4)))
        inter_gene()

    def draw(self, size):

        def get_cube_list(p1, p2):
            p2=p2+1
            return [p1[0],p2[0], p1[1],p2[1], p1[2],p2[2]]

        cube_actors=[vtk_cube(bounds=get_cube_list(*wall.get_cube())) for wall in self.placed_wall_list]
        print(len(cube_actors))

        renderer = vtk.vtkRenderer()
        renderer.SetBackground(0.0, 0.0, 0.0)  # 背景只有一个所以是Set()
        for cube_actor in cube_actors:
            renderer.AddActor(cube_actor)  # 因为actor有可能为多个所以是add()

        # 5. 显示渲染窗口
        render_window = vtk.vtkRenderWindow()
        render_window.SetWindowName("My First Cube")
        render_window.SetSize(800, 800)
        render_window.AddRenderer(renderer)  # 渲染也会有可能有多个渲染把他们一起显示
        # 6. 创建交互控键（可以用鼠标拖来拖去看三维模型）
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(render_window)
        interactor.Initialize()
        render_window.Render()
        interactor.Start()

def find_longest_path(maze, num=10):
    poses = np.where(maze.maze==0)

    G = nx.Graph()
    ds=[Pos(-1,0), Pos(1,0), Pos(0,-1), Pos(0,1)]
    for y in range(maze.size[1]):
        for x in range(maze.size[0]):
            if maze.maze[y,x]:
                pos=Pos(x,y)
                data=tuple(filter(lambda p:p.x>=0 and p.y>=0 and p.x<maze.size[0] and p.y<maze.size[1] and not maze.maze[p.y,p.x], [pos+d for d in ds]))
                for p in data:
                    G.add_edge(pos, p)
            #data[pos]=tuple(filter(lambda p:p.x>=0 and p.y>=0 and p.x<maze.size[0] and p.y<maze.size[1], [pos+d for d in ds]))

    p_st = None
    p_end = None
    plen=0
    path_num=0

    for i in range(num):
        idx=np.random.randint(0, len(poses[0])-1, (2,))
        p1=Pos(poses[1][idx[0]], poses[0][idx[0]])
        p2=Pos(poses[1][idx[1]], poses[0][idx[1]])
        try:
            path = nx.shortest_path(G, p1, p2)
            path_num+=1
            if(len(path)>plen):
                p_st=p1
                p_end=p2
        except Exception as e:
            pass
    return p_st, p_end, path_num/num

def parse_args():
    def str2size(v: str):
        return tuple(int(x) for x in v.strip().split('x'))

    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--size', default=(40,40,40), type=str2size)
    parser.add_argument('--name', default='maze4', type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args=parse_args()

    maze=Maze3D(size=args.size)
    #maze.maze[12:12+5, :5]=1
    maze.generate()
    maze.draw(1)
    #maze.generate_coins()
    #se=find_longest_path(maze, num=1000)
    #print(se)
    #img=maze.draw()
    #cv2.imshow('aa', img)
    #cv2.waitKey()

    with open(f"{args.name}.pkl", "wb") as f:
        pickle.dump({'wall': maze.placed_wall_list, 'coin':maze.placed_coin_list}, f)