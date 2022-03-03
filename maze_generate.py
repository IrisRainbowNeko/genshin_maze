import numpy as np
import random
from typing import List
from copy import deepcopy
import cv2

import networkx as nx

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
        return hash((self.x, self.y))

class Maze:
    def __init__(self, size=(20,20)):
        self.size=size

        self.maze=np.zeros(size[::-1], dtype=np.uint8)

        self.unit_list=self.get_unit_pos_list()
        self.check_list=[]

        for pos in self.unit_list:
            self.maze[pos.y, pos.x]=1
        self.maze_unit = deepcopy(self.maze)

    def random_add(self, src: List, dst: List):
        dst.append(src.pop(random.randint(0,len(src)-1)))

    def get_unit_pos_list(self) -> List[Pos]:
        x = np.arange(1, self.size[0], 2)
        y = np.arange(1, self.size[1], 2)

        X, Y = np.meshgrid(x, y)
        return [Pos(*pos) for pos in zip(X.reshape(-1).tolist(),Y.reshape(-1).tolist())]

    def find_nearest_unit(self, unit):
        unit_list=[]
        row_idx=np.where(self.maze_unit[unit.y, :])[0]
        left=row_idx[row_idx<unit.x]
        if left.size>0:
            unit_list.append(Pos(np.max(left), unit.y))
        right=row_idx[row_idx>unit.x]
        if right.size>0:
            unit_list.append(Pos(np.min(right), unit.y))

        col_idx = np.where(self.maze_unit[:, unit.x])[0]
        top = col_idx[col_idx < unit.y]
        if len(top.shape)>0 and top.shape[0]>0:
            unit_list.append(Pos(unit.x, np.max(top)))
        bottom = col_idx[col_idx > unit.y]
        if len(bottom.shape)>0 and bottom.shape[0]>0:
            unit_list.append(Pos(unit.x, np.min(bottom)))

        return unit_list

    def remove_walls(self, ua, ub):
        xl,xh=min(ua.x, ub.x), max(ua.x, ub.x)
        yl,yh=min(ua.y, ub.y), max(ua.y, ub.y)
        if xl==xh:
            self.maze[yl+1:yh, xl:xh+1] = 1
        else:
            self.maze[yl:yh+1, xl+1:xh] = 1

    def generate(self):
        while len(self.unit_list)>0:
            self.random_add(self.unit_list, self.check_list)
            while len(self.check_list)>0:
                unit = random.choice(self.check_list)
                around_units = [item for item in self.find_nearest_unit(unit) if item in self.unit_list]
                if len(around_units)>0:
                    o_unit = random.choice(around_units)
                    self.remove_walls(unit, o_unit)
                    self.check_list.append(o_unit)
                    self.unit_list.remove(o_unit)
                else:
                    self.check_list.remove(unit)

    def draw(self):
        img=cv2.resize(self.maze*255, (400,400), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('a', img)
        cv2.waitKey()


class MazePP(Maze):
    def get_unit_pos_list(self) -> List[Pos]:
        raw = list(range(0, self.size[0]*self.size[1]))
        raw = np.array(random.sample(raw, 10))
        X = np.mod(raw, self.size[0])
        Y = raw // self.size[0]

        #X, Y = np.meshgrid(x, y)
        return [Pos(*pos) for pos in zip(X.reshape(-1).tolist(),Y.reshape(-1).tolist())]

def find_longest_path(maze, num=10):
    poses = np.where(maze.maze)

    G = nx.Graph()
    ds=[Pos(-1,0), Pos(1,0), Pos(0,-1), Pos(0,1)]
    for y in range(maze.size[1]):
        for x in range(maze.size[0]):
            if maze.maze[y,x]:
                pos=Pos(x,y)
                data=tuple(filter(lambda p:p.x>=0 and p.y>=0 and p.x<maze.size[0] and p.y<maze.size[1] and maze.maze[p.y,p.x], [pos+d for d in ds]))
                for p in data:
                    G.add_edge(pos, p)
            #data[pos]=tuple(filter(lambda p:p.x>=0 and p.y>=0 and p.x<maze.size[0] and p.y<maze.size[1], [pos+d for d in ds]))

    p_st = None
    p_end = None
    plen=0

    for i in range(num):
        idx=np.random.randint(0, len(poses[0])-1, (2,))
        p1=Pos(poses[1][idx[0]], poses[0][idx[0]])
        p2=Pos(poses[1][idx[1]], poses[0][idx[1]])
        path = nx.shortest_path(G, p1, p2)

        if(len(path)>plen):
            p_st=p1
            p_end=p2
    return p_st, p_end



if __name__ == '__main__':
    maze=Maze(size=(40,40))
    maze.generate()
    print(find_longest_path(maze, num=50))
    maze.draw()