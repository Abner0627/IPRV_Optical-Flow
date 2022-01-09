# %%
import cv2
import os
import func

# %%
path = './img'
sP = './res'
img_ty = 'Pillow'
IMG_L = func._pick(os.listdir(path), img_ty, path)
# (u,v) = optical_flow(IMG_L[0], IMG_L[1], 16, tau=1e-2)

img_shape = IMG_L[1].shape[:2]
pre_img = func._gray(IMG_L[0])
nxt_img = func._gray(IMG_L[1])

# %%
# pt_x, pt_y = 250, 325
img = cv2.cvtColor(IMG_L[0], cv2.COLOR_BGR2RGB)
for pt_x, pt_y in zip([250, 400], [325, 270]):
    print('\nChoose point: ({}, {})'.format(pt_x, pt_y))
    PX, PY = func._flow(pre_img, nxt_img, pt_x, pt_y, func.param, init_flow=None)
    img = func._plot(img, PX, PY)

import matplotlib.pyplot as plt
fig = plt.figure()
plt.imshow(img)
plt.show()

# %%
# fn = img_ty + '_res.png'
# cv2.imwrite(os.path.join(sP, fn), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


