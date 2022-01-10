#! /usr/bin/env python
#  -*- coding: utf-8 -*-
import sys
import os
import func
import cv2
import numpy as np
import matplotlib.pyplot as plt

try:
    import Tkinter as tk
except ImportError:
    import tkinter as tk

try:
    import ttk
    py3 = False
except ImportError:
    import tkinter.ttk as ttk
    py3 = True

def init(top, gui, *args, **kwargs):
    global w, top_level, root
    w = gui
    top_level = top
    root = top

def onBtnModifyClick_1():
    # print('GUI_support.onBtnModifyClick_1')
    # sys.stdout.flush()
    global IMG_L
    global img_0
    global pre_img
    global nxt_img
    global text_get
    img_list = os.listdir('./img')
    # 取得./img中影像列表
    text_get = w.TEntry1.get()
    # 取得GUI輸入(此處為影像名稱)
    IMG_L = func._pick(img_list, text_get, './img')
    # 選取影像之檔名並加載影像
    pre_img = func._gray(IMG_L[0])
    nxt_img = func._gray(IMG_L[1])
    # 從BGR轉至RGB
    img_0 = cv2.cvtColor(IMG_L[0], cv2.COLOR_BGR2RGB)
    func._Pos(img_0, text_get)
    # 生成影像用以供使用者標記目標點
    img_0 = func._PlotPos(img_0, text_get)
    plt.imshow(img_0)
    plt.show() 
    # 畫上選取點


def onBtnModifyClick_2():
    global img_1
    img_1 = cv2.cvtColor(IMG_L[1], cv2.COLOR_BGR2RGB)
    src = np.load('./npy/loc_' + text_get + '.npy')
    # 讀取儲存的選取點座標
    for idx in range(2):
        print('\nPoint {}'.format(idx+1))
        pt_x, pt_y = src[idx, 0], src[idx, 1]
        PX, PY = func._LKflow(pre_img, nxt_img, pt_x, pt_y, func.lk_params)
        # Lucas-Kanade Flow
        img_1 = func._plot(img_1, PX, PY)
        # 畫出每個iteration的標點
    plt.imshow(img_1)
    plt.show()

def onBtnModifyClick_3():
    fn = text_get + '_res.png'
    fn0 = text_get + '_init.png'
    sP = './res'
    cv2.imwrite(os.path.join(sP, fn), cv2.cvtColor(img_1, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(sP, fn0), cv2.cvtColor(img_0, cv2.COLOR_RGB2BGR))
    # 將結果轉回BGR使用cv2儲存
    print('\nSaved')

def destroy_window():
    # Function which closes the window.
    global top_level
    top_level.destroy()
    top_level = None

if __name__ == '__main__':
    import GUI
    GUI.vp_start_gui()





