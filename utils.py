import time
import argparse

import cv2
import numpy as np
import win32api, win32con, win32gui, win32ui
from pathlib import Path
import yaml

CONFIG_PATH = Path("./config.yaml")
assert CONFIG_PATH.is_file()


with open(CONFIG_PATH, encoding='utf-8') as f:
    result = yaml.safe_load(f)
    DEFAULT_MONITOR_WIDTH = result.get("windows").get("monitor_width")
    DEFAULT_MONITOR_HEIGHT = result.get("windows").get("monitor_height")
    WINDOW_NAME = result.get("game").get("window_name")

gvars=argparse.Namespace()
hwnd = win32gui.FindWindow(None, WINDOW_NAME)
gvars.genshin_window_rect = win32gui.GetWindowRect(hwnd)

def cap(region=None ,fmt='RGB'):
    if region is not None:
        left, top, w, h = region
        # w = x2 - left + 1
        # h = y2 - top + 1
    else:
        w = DEFAULT_MONITOR_WIDTH  # set this
        h = DEFAULT_MONITOR_HEIGHT  # set this
        left = 0
        top = 0

    hwnd = win32gui.FindWindow(None, WINDOW_NAME)
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
