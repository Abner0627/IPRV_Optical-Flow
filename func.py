import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# %%
def _pick(L, ty, path):
    L_ = [cv2.imread(os.path.join(path, i)) for i in L if i.split('_')[0]==ty]
    # 輸入影像
    return L_

def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def _Pos(img, idx):
    def on_press(event):
        L.append(np.array([int(event.xdata), int(event.ydata)]))
        # 紀錄點選的座標點
        if len(L)>=2: 
            plt.close()
            # 當點選次數大於等於2時，關閉視窗
        np.save('./npy/loc_' + idx + '.npy', np.array(L))
        # 儲存紀錄座標點
    fig = plt.figure()
    plt.imshow(img, animated= True)
    L = []
    fig.canvas.mpl_connect('button_press_event', on_press)
    # 用動態圖的形式產生介面供使用者點選目標點
    plt.show() 

def _PlotPos(img, idx):
    img_c = np.copy(img)
    src = np.load('./npy/loc_' + idx + '.npy')
    # 輸入儲存的選取座標
    print('Choose point 1: ({}, {})'.format(src[0, 0], src[0, 1]))
    print('Choose point 2: ({}, {})'.format(src[1, 0], src[1, 1]))
    cv2.circle(img_c, (src[0, 0], src[0, 1]), 3, (0, 38, 255), -1)
    cv2.circle(img_c, (src[1, 0], src[1, 1]), 3, (0, 38, 255), -1)
    # 畫上座標點
    return img_c

# def _flow(pre_img, nxt_img, pt_x, pt_y, param, init_flow=None):
#     XL, YL = [0], [0]
#     PX, PY = [pt_x], [pt_y]
#     flow = init_flow
#     ep = 1000
#     i=0
#     while ep>1e-2:
#         if i==0:
#             fg = 0
#         else:
#             fg = cv2.OPTFLOW_USE_INITIAL_FLOW
#         flow = cv2.calcOpticalFlowFarneback(pre_img, nxt_img, flow=flow, flags=fg, **param)
        
#         XL.append(flow[pt_y, pt_x, 0])
#         YL.append(flow[pt_y, pt_x, 1])
#         PX.append(int(pt_x + flow[pt_y, pt_x, 0]))
#         PY.append(int(pt_y + flow[pt_y, pt_x, 1]))
#         print('iter:{}, ep:{}\nu = {:.4f}, v = {:.4f}'.format(i, ep, XL[i], YL[i]))
#         print('x = {:.4f}, y = {:.4f}'.format(PX[i], PY[i]))
#         print('======================')
#         i+=1
#         if i>0:
#             ep = np.sum(np.abs(XL[i-1] - XL[i])) + np.sum(np.abs(YL[i-1] - YL[i]))
#     return PX, PY

def _LKflow(pre_img, nxt_img, pt_x, pt_y, lk_params):
    p0 = np.array([[pt_x, pt_y]]).astype(np.float32)
    i = 0
    PX, PY = [pt_x], [pt_y]
    XL, YL = [], []
    ep = 1e3
    # 初始化各參數
    while ep>1e-2:
        if i==0:
            p1, _, _ = cv2.calcOpticalFlowPyrLK(pre_img, nxt_img, p0, None, **lk_params)
        else:
            p1, _, _ = cv2.calcOpticalFlowPyrLK(pre_img, nxt_img, p0, p1, flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **lk_params)
        # 用迴圈計算每個iteration的輸出座標
        PX.append(p1[0][0])
        PY.append(p1[0][1])
        XL.append(PX[i] - PX[i+1])
        YL.append(PY[i] - PY[i+1])
        # 紀錄輸出座標與位移向量
        if i>0:
            ep = np.sum(np.abs(XL[i-1] - XL[i])) + np.sum(np.abs(YL[i-1] - YL[i])) 
            # 與前一個iteration位移向量之差值，
            # 當差值<0.01時則停止迴圈
        print('iter:{}, ep:{}\nu = {:.4f}, v = {:.4f}'.format(i, ep, XL[i], YL[i]))
        print('x = {:.4f}, y = {:.4f}'.format(PX[i+1], PY[i+1]))
        print('======================')    
        i+=1    
    return PX, PY    

def _plot(img, PX, PY):
    PX = np.array(PX).astype(np.int)
    PY = np.array(PY).astype(np.int)
    for j in range(len(PX)):
        if j!=0:
            cv2.line(img, (PX[j-1], PY[j-1]), (PX[j], PY[j]), (250, 5, 216), 2)
    for k in range(len(PX)):
        if k==0:
            c = (0, 38, 255)
        elif k==len(PX)-1:
            c = (182, 255, 0)
        else:
            c = (255, 0, 0)
        cv2.circle(img, (PX[k], PY[k]), 3, c, -1) 
    # 依每個iteration輸出的座標畫上標點
    return img

# param = dict(pyr_scale=0.8,
#             levels=25,
#             iterations=1,
#             winsize=5,
#             poly_n=5,
#             poly_sigma=1.1)

lk_params = dict(winSize  = (15, 15),
                 maxLevel = 3,
                 criteria = (cv2.TERM_CRITERIA_COUNT, 1, 0.03))            