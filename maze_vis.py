from maze_generate import *
import cv2
import imageio
import numpy as np

with open(r"maze1.pkl", "rb") as f:
    walls = pickle.load(f)['maze']

maze=MazePP(size=(30,30))

fps = 60.0  # 视频帧率
size = (800, 800)  # 需要转为视频的图片的尺寸
video_out = imageio.get_writer('maze1.mp4', mode='I', fps=60, codec='libx264', bitrate='6M')

for i,wall in enumerate(walls):
    maze.place_wall(wall)
    img=maze.draw(size)
    print(f'{i}/{len(walls)}')
    for u in range(3):
        video_out.append_data(img)

for u in range(40):
    video_out.append_data(img)

video_out.close()