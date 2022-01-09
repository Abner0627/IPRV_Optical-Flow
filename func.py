import cv2
import os
import numpy as np

# %%
def _pick(L, ty, path):
    L_ = [cv2.imread(os.path.join(path, i)) for i in L if i.split('_')[0]==ty]
    return L_

def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def _flow(pre_img, nxt_img, pt_x, pt_y, param, init_flow=None):
    XL, YL = [0], [0]
    PX, PY = [pt_x], [pt_y]
    flow = init_flow
    ep = 1000
    i=0
    while ep>1e-2:
        if i==0:
            fg = 0
        else:
            fg = cv2.OPTFLOW_USE_INITIAL_FLOW
        flow = cv2.calcOpticalFlowFarneback(pre_img, nxt_img, flow=flow, flags=fg, **param)
        
        XL.append(flow[pt_y, pt_x, 0])
        YL.append(flow[pt_y, pt_x, 1])
        PX.append(int(pt_x + flow[pt_y, pt_x, 0]))
        PY.append(int(pt_y + flow[pt_y, pt_x, 1]))
        print('iter:{}, ep:{}\nu = {:.4f}, v = {:.4f}'.format(i, ep, XL[i], YL[i]))
        print('x = {:.4f}, y = {:.4f}'.format(PX[i], PY[i]))
        print('======================')
        i+=1
        if i>0:
            ep = np.sum(np.abs(XL[i-1] - XL[i])) + np.sum(np.abs(YL[i-1] - YL[i]))
    return PX, PY, i

def _plot(img, PX, PY):
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
        cv2.circle(img,(PX[k], PY[k]), 3, c, -1)    
    return img

param = dict(pyr_scale=0.8,
            levels=25,
            iterations=1,
            winsize=5,
            poly_n=5,
            poly_sigma=1.1)