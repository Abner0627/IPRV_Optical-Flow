# IPRV_Optical-Flow
NCKU Image Processing and Robot Vision course homework

## 專案目錄結構
```
Project
│   GUI.py
|   GUI_support.py
│   func.py
│   requirements.txt  
│   README.md      
│   ...    
└───img   
│   │   Cup_0.Jpg
|   |   Cup_1.Jpg
│   │   ...
└───result   
│   │   Cup_init.png
│   │   Cup_res.png
|   |   ...
└───npy  
│   │   loc_Cup.npy
|   |   ...
└───ipynb 
```

## 前置工作
### 作業說明
* 目標\
透過Lucas-Kanade Flow偵測兩張影像之間的光流，\
並標註每個iteration的輸出座標。

### 環境
* python 3.8
* Win 11

### 使用方式
1. 進入專案資料夾\
`cd [path/to/this/project]` 

2. 使用`pip install -r requirements.txt`安裝所需套件

3. 將欲處理的影像放入`./img`中\
   所有檔案的命名規則如下：\
   欲偵測影像：`<影像名稱>_<編號>.<副檔名>`\
   示意如下：\
   ![Imgur](https://i.imgur.com/AjEHsrJ.png)
4. 執行主程式開啟GUI\
`python GUI.py`   
使用介面介紹如下：
![Imgur](https://i.imgur.com/A3fBprm.png)
(1) 輸入影像的名稱，此作業共有"Cup"與"Pillow"兩類\
(2) 加載上述影像\
(3) 計算每個iteration之Lucas-Kanade Flow的輸出點\
(4) 儲存影像\
另外在terminal上會依序print出目前流程的資訊，詳見以下操作影片：\
[![Imgur](https://i.imgur.com/0r4JQVj.png)](https://www.youtube.com/watch?v=K2AEwcPKR8I)

## 程式碼說明
此處依上述GUI介面按鍵標號((2)~(4))進行說明，並省略GUI介面設計之介紹
### 輸入影像並供使用者選取座標點
```py
# GUI_support.py
def onBtnModifyClick_1():
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
```
```py
# func.py
def _pick(L, ty, path):
    L_ = [cv2.imread(os.path.join(path, i)) \
        for i in L if i.split('_')[0]==ty]
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
```
### Lucas-Kanade Flow
```py
# GUI_support.py
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
```
```py
# func.py
def _LKflow(pre_img, nxt_img, pt_x, pt_y, lk_params):
    p0 = np.array([[pt_x, pt_y]]).astype(np.float32)
    i = 0
    PX, PY = [pt_x], [pt_y]
    XL, YL = [], []
    ep = 1e3
    # 初始化各參數
    while ep>1e-2:
        if i==0:
            p1, _, _ = cv2.calcOpticalFlowPyrLK(pre_img, nxt_img, p0, \
                        None, **lk_params)
        else:
            p1, _, _ = cv2.calcOpticalFlowPyrLK(pre_img, nxt_img, p0, \
                        p1, flags=cv2.OPTFLOW_USE_INITIAL_FLOW, \
                        **lk_params)
        # 用迴圈計算每個iteration的輸出座標
        PX.append(p1[0][0])
        PY.append(p1[0][1])
        XL.append(PX[i] - PX[i+1])
        YL.append(PY[i] - PY[i+1])
        # 紀錄輸出座標與位移向量
        if i>0:
            ep = np.sum(np.abs(XL[i-1] - XL[i])) + \
                 np.sum(np.abs(YL[i-1] - YL[i])) 
            # 與前一個iteration位移向量之差值，
            # 當差值<0.01時則停止迴圈
        print('iter:{}, ep:{}\nu = {:.4f}, v = {:.4f}'\
                .format(i, ep, XL[i], YL[i]))
        print('x = {:.4f}, y = {:.4f}'.format(PX[i+1], PY[i+1]))
        print('======================')    
        i+=1    
    return PX, PY    

def _plot(img, PX, PY):
    PX = np.array(PX).astype(np.int)
    PY = np.array(PY).astype(np.int)
    for j in range(len(PX)):
        if j!=0:
            cv2.line(img, (PX[j-1], PY[j-1]), (PX[j], PY[j]), \
                (250, 5, 216), 2)
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
```
其iteration位移向量之差值的計算如下：
![Imgur](https://i.imgur.com/jClPxK4.png)
### 儲存影像
```py
# GUI_support.py
def onBtnModifyClick_3():
    fn = text_get + '_res.png'
    fn0 = text_get + '_init.png'
    sP = './res'
    cv2.imwrite(os.path.join(sP, fn), \
        cv2.cvtColor(img_1, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(sP, fn0), \
        cv2.cvtColor(img_0, cv2.COLOR_RGB2BGR))
    # 將結果轉回BGR使用cv2儲存
    print('\nSaved')
```

## 結果展示
### Cup
#### Cup_init
![Imgur](https://i.imgur.com/tjcJT6K.png)
#### Cup_res
![Imgur](https://i.imgur.com/bbVAnq5.png)

### Pillow
#### Pillow_init
![Imgur](https://i.imgur.com/QjhylIa.png)
#### Pillow_res
![Imgur](https://i.imgur.com/zxaF64M.png)

