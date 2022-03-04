import numpy as np
import random
from typing import List
from copy import deepcopy
import cv2

import networkx as nx
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

class Maze:
    def __init__(self, size=(20,20)):
        self.size=size

        self.maze=np.zeros(size[::-1], dtype=np.uint8)

        self.unit_list=self.get_unit_pos_list()
        self.check_list=[]


        self.maze_unit = deepcopy(self.maze)
        for pos in self.unit_list:
            self.maze_unit[pos.y, pos.x]=1

    def random_add(self, src: List, dst: List):
        dst.append(src.pop(random.randint(0,len(src)-1)))

    def get_unit_pos_list(self) -> List[Pos]:
        x = np.arange(0, self.size[0], 2)
        y = np.arange(0, self.size[1], 2)

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
            self.maze[yl:yh+1, xl:xh+1] = 1
        else:
            self.maze[yl:yh+1, xl:xh+1] = 1

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

class Maze2(Maze):
    def find_nearest_unit(self, unit):
        d_list=[Pos(-4, 0), Pos(4, 0), Pos(0, -4), Pos(0, 4)]

        unit_list=[]
        for d in d_list:
            dpos=unit+d
            if dpos.x>=0 and dpos.y>=0 and dpos.x<self.size[0] and dpos.y<self.size[1]:
                flag=False
                if unit.y==dpos.y:
                    for y in range(unit.y, dpos.y+1):
                        for x in range(min(unit.x,dpos.x)+1, max(unit.x,dpos.x) ):
                            if self.maze[y, x]:
                                flag=True
                                break
                        if flag:
                            break
                else:
                    for y in range(min(unit.y,dpos.y)+1, max(unit.y,dpos.y) ):
                        for x in range(unit.x, dpos.x + 1 ):
                            if self.maze[y, x]:
                                flag=True
                                break
                        if flag:
                            break
                if not flag:
                    unit_list.append(dpos)

        return unit_list

class MazePP:
    def __init__(self, size=(20,20), wall_len=5):
        self.size=size
        self.wall_len=wall_len
        self.maze = np.zeros(size[::-1], dtype=np.uint8)

        self.wall_list=[]
        self.placed_wall_list=[]

    def can_place(self, wall):
        rect=wall.get_rect()
        if wall.dir%2==0:
            if wall.dir==0:
                rect[1]-=1
            else:
                rect[3]+=1
            for y in range(rect[1], rect[3]+1):
                for x in range(rect[0]-1, rect[2]+2):
                    if y<0 or x<0 or y>=self.size[1] or x>=self.size[0] or self.maze[y, x]:
                        return False
            #if (wall.dir==0 and wall.y-wall.len>=0 and self.maze[wall.y-wall.len, x]) or (wall.dir==2 and wall.y+wall.len<self.size[1]-1 and self.maze[wall.y+wall.len, x]):
            #    return False
        else:
            if wall.dir==1:
                rect[2]+=1
            else:
                rect[0]-=1
            for y in range(rect[1]-1, rect[3]+2):
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

        self.maze[0,:]=1
        self.maze[:,0]=1
        self.maze[-1, :] = 1
        self.maze[:, -1] = 1

    def get_unit_pos_list(self) -> List[Pos]:
        raw = list(range(0, self.size[0]*self.size[1]))
        raw = np.array(random.sample(raw, 10))
        X = np.mod(raw, self.size[0])
        Y = raw // self.size[0]

        #X, Y = np.meshgrid(x, y)
        return [Pos(*pos) for pos in zip(X.reshape(-1).tolist(),Y.reshape(-1).tolist())]

    def draw(self, size=(400,400)):
        img=self.maze*255
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
        try:
            path = nx.shortest_path(G, p1, p2)

            if(len(path)>plen):
                p_st=p1
                p_end=p2
        except Exception as e:
            pass
    return p_st, p_end



if __name__ == '__main__':
    maze=MazePP(size=(30,30))
    maze.maze[12:12+5, :5]=1
    maze.generate()
    se=find_longest_path(maze, num=500)
    print(se)
    img=maze.draw()
    cv2.imshow('aa', img)
    cv2.waitKey()

    with open(r"maze1.pkl", "wb") as f:
        pickle.dump({'maze': maze.placed_wall_list, 'se': se}, f)