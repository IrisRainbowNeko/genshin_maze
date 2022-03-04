import numpy as np
import random
from typing import List
from copy import deepcopy
import cv2

import networkx as nx
import argparse
import pickle

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

class MazePP:
    def __init__(self, size=(20,20), wall_len=5):
        self.size=size
        self.wall_len=wall_len
        self.maze = np.zeros(size[::-1], dtype=np.uint8)

        self.wall_list=[]
        self.placed_wall_list=[]
        self.placed_coin_list=[]

    def can_place(self, wall):
        rect=wall.get_rect()
        if wall.dir%2==0:
            if wall.dir==0:
                if rect[1]>0:
                    rect[1]-=1
            else:
                if rect[3]<self.size[1]-1:
                    rect[3]+=1

            if rect[0]>0:
                rect[0]-=1
            if rect[2] < self.size[0] - 1:
                rect[2] += 1

            for y in range(rect[1], rect[3]+1):
                for x in range(rect[0], rect[2]+1):
                    if y<0 or x<0 or y>=self.size[1] or x>=self.size[0] or self.maze[y, x]:
                        return False
            #if (wall.dir==0 and wall.y-wall.len>=0 and self.maze[wall.y-wall.len, x]) or (wall.dir==2 and wall.y+wall.len<self.size[1]-1 and self.maze[wall.y+wall.len, x]):
            #    return False
        else:
            if wall.dir==1:
                if rect[2] < self.size[0] - 1:
                    rect[2]+=1
            else:
                if rect[0] > 0:
                    rect[0]-=1

            if rect[1] > 0:
                rect[1] -= 1
            if rect[3] < self.size[1] - 1:
                rect[3] += 1

            for y in range(rect[1], rect[3]+1):
                for x in range(rect[0], rect[2]+1):
                    if y<0 or x<0 or y>=self.size[1] or x>=self.size[0] or self.maze[y, x]:
                        return False
            #if (wall.dir==3 and wall.x-wall.len>=0 and self.maze[y, wall.x-wall.len]) or (wall.dir==1 and wall.x+wall.len<self.size[0]-1 and self.maze[y, wall.x+wall.len]):
            #    return False
        return True

    def place_wall(self, wall):
        rect = wall.get_rect()
        self.maze[rect[1]:rect[3]+1, rect[0]:rect[2]+1]=1
        self.placed_wall_list.append(wall)

    def get_next_states_list(self, wall):
        wall_list=[]
        rect = wall.get_rect()
        if rect[0]==rect[2]:
            for y in range(rect[1], rect[3]+1):
                for d in range(4):
                    wall_list.append(Wall(rect[0]-1, y, self.wall_len, d))
                    wall_list.append(Wall(rect[0]+1, y, self.wall_len, d))
        else:
            for x in range(rect[0], rect[2]+1):
                for d in range(4):
                    wall_list.append(Wall(x, rect[1]-1, self.wall_len, d))
                    wall_list.append(Wall(x, rect[1]+1, self.wall_len, d))
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

    def get_unit_pos_list(self) -> List[Pos]:
        raw = list(range(0, self.size[0]*self.size[1]))
        raw = np.array(random.sample(raw, 10))
        X = np.mod(raw, self.size[0])
        Y = raw // self.size[0]

        #X, Y = np.meshgrid(x, y)
        return [Pos(*pos) for pos in zip(X.reshape(-1).tolist(),Y.reshape(-1).tolist())]

    def draw(self, size=(400,400)):
        img=np.stack([self.maze,self.maze,self.maze], axis=-1)

        yellow=np.array([0,1,1])
        for coin in self.placed_coin_list:
            img[coin.y:coin.y+2,coin.x:coin.x+2]=yellow

        img=img*255
        img=cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)

        return img

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
    parser.add_argument('--size', default=(40,40), type=str2size)
    parser.add_argument('--name', default='maze4', type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args=parse_args()

    maze=MazePP(size=args.size)
    #maze.maze[12:12+5, :5]=1
    maze.generate()
    maze.generate_coins()
    se=find_longest_path(maze, num=1000)
    print(se)
    img=maze.draw()
    cv2.imshow('aa', img)
    cv2.waitKey()

    with open(f"{args.name}.pkl", "wb") as f:
        pickle.dump({'wall': maze.placed_wall_list, 'coin':maze.placed_coin_list}, f)