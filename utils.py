import time
import argparse

import cv2
import numpy as np
import scipy
#import scipy.linalg
from scipy.spatial.transform import Rotation as R
import win32api, win32con, win32gui, win32ui
from pathlib import Path
import yaml

import vtk

gvars=argparse.Namespace()

def cap_init():
    CONFIG_PATH = Path("./config.yaml")
    assert CONFIG_PATH.is_file()

    with open(CONFIG_PATH, encoding='utf-8') as f:
        result = yaml.safe_load(f)
        gvars.DEFAULT_MONITOR_WIDTH = result.get("windows").get("monitor_width")
        gvars.DEFAULT_MONITOR_HEIGHT = result.get("windows").get("monitor_height")
        gvars.WINDOW_NAME = result.get("game").get("window_name")

    hwnd = win32gui.FindWindow(None, gvars.WINDOW_NAME)
    gvars.genshin_window_rect = win32gui.GetWindowRect(hwnd)

def cap(region=None ,fmt='RGB'):
    if region is not None:
        left, top, w, h = region
        # w = x2 - left + 1
        # h = y2 - top + 1
    else:
        w = gvars.DEFAULT_MONITOR_WIDTH  # set this
        h = gvars.DEFAULT_MONITOR_HEIGHT  # set this
        left = 0
        top = 0

    hwnd = win32gui.FindWindow(None, gvars.WINDOW_NAME)
    # hwnd = win32gui.GetDesktopWindow()
    wDC = win32gui.GetWindowDC(hwnd)
    dcObj = win32ui.CreateDCFromHandle(wDC)
    cDC = dcObj.CreateCompatibleDC()
    dataBitMap = win32ui.CreateBitmap()

    dataBitMap.CreateCompatibleBitmap(dcObj, w, h)

    cDC.SelectObject(dataBitMap)
    cDC.BitBlt((0, 0), (w, h), dcObj, (left, top), win32con.SRCCOPY)
    # dataBitMap.SaveBitmapFile(cDC, bmpfilenamename)
    signedIntsArray = dataBitMap.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype="uint8")
    img.shape = (h, w, 4)

    # Free Resources
    dcObj.DeleteDC()
    cDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, wDC)
    win32gui.DeleteObject(dataBitMap.GetHandle())
    
    if fmt == 'BGR':
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGBA2BGR)
    if fmt == 'RGB':
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGBA2RGB)
    else:
        raise ValueError('Cannot indetify this fmt')

def match_img(img, target, type=cv2.TM_CCOEFF):
    h, w = target.shape[:2]
    res = cv2.matchTemplate(img, target, type)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if type in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        return (
            *min_loc,
            min_loc[0] + w,
            min_loc[1] + h,
            min_loc[0] + w // 2,
            min_loc[1] + h // 2,
        )
    else:
        return (
            *max_loc,
            max_loc[0] + w,
            max_loc[1] + h,
            max_loc[0] + w // 2,
            max_loc[1] + h // 2,
        )


#ax, ay, az = [1,0,0], [0,1,0], [0,0,1]
ax, ay, az='x','y','z'
T_world=0
T_self=1

#旋转矩阵 欧拉角
def rotate_mat(axis, deg):
    r = R.from_euler(axis, deg, degrees=True)
    return r.as_matrix()
    #rot_matrix = scipy.linalg.expm(np.cross(np.eye(3), axis / scipy.linalg.norm(axis) * np.deg2rad(deg)))
    #return rot_matrix

def rotate_mat_steps(*data, type=T_self):
    rot_matrix = rotate_mat(*data[0])
    if type==T_world:
        for item in data[1:]:
            rot_matrix=np.dot(rotate_mat(*item), rot_matrix)
    else:
        for item in data[1:]:
            rot_matrix=np.dot(rot_matrix, rotate_mat(*item))
    return rot_matrix

def vtk_cube(center=(0., 0., 0.), x_length=1.0, y_length=1.0, z_length=1.0, bounds=None):
    """Create a cube.

    It's possible to specify either the center and side lengths or just
    the bounds of the cube. If ``bounds`` are given, all other arguments are
    ignored.

    Parameters
    ----------
    center : np.ndarray or list
        Center in [x, y, z].

    x_length : float
        length of the cube in the x-direction.

    y_length : float
        length of the cube in the y-direction.

    z_length : float
        length of the cube in the z-direction.

    bounds : np.ndarray or list
        Specify the bounding box of the cube. If given, all other arguments are
        ignored. ``(xMin,xMax, yMin,yMax, zMin,zMax)``

    """
    cube = vtk.vtkCubeSource()
    if bounds is not None:
        if np.array(bounds).size != 6:
            raise TypeError('Bounds must be given as length 6 tuple: (xMin,xMax, yMin,yMax, zMin,zMax)')
        cube.SetBounds(bounds)
    else:
        cube.SetCenter(center)
        cube.SetXLength(x_length)
        cube.SetYLength(y_length)
        cube.SetZLength(z_length)
    cube.Update()

    cube_mapper = vtk.vtkPolyDataMapper()
    cube_mapper.SetInputData(cube.GetOutput())
    # 3. 根据2创建执行单元
    cube_actor = vtk.vtkActor()
    cube_actor.SetMapper(cube_mapper)
    cube_actor.GetProperty().SetColor(1.0, 1.0, 1.0)
    cube_actor.GetProperty().SetOpacity(0.1)
    return cube_actor