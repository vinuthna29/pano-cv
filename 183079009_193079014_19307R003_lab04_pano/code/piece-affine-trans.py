import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import PiecewiseAffineTransform, warp
from skimage import data
count=0
positions=[]
def click_event(event, x, y, flags, params):
    global count,positions
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates of the peak values of sinusoid i.e max and min
        print(x, ' ', y)
        positions.append([x,y])
        count+=1
        cv2.imshow('image', img)

img = cv2.imread('../data/piece/brick.png')
print(img.shape) # x,y coordinates size is obtained
cv2.imshow('image', img)

cv2.setMouseCallback('image',click_event)
while(True):
    cv2.imshow('image',img)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break
    elif count==2:
        #cv2.waitKey(0)
        break
        
[h,a]=np.array(positions[0])-np.array(positions[1])
print(h,a)
A=a/2  # amplitude of sinusoid
n=img.shape[1]/(2*h) # no. of cycles
print(n,A)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

rows, cols = img.shape[0], img.shape[1]

src_cols = np.linspace(0, cols, 20)
src_rows = np.linspace(0, rows, 10)
src_rows, src_cols = np.meshgrid(src_rows, src_cols)
src = np.dstack([src_cols.flat, src_rows.flat])[0]

# add sinusoidal oscillation to row coordinates
dst_rows = src[:, 1] - np.sin(np.linspace(0, 2* n * np.pi, src.shape[0])) * A
dst_cols = src[:, 0]
dst_rows *= np.round(n,1)
dst_rows -= np.round(n,1) * 50
dst = np.vstack([dst_cols, dst_rows]).T

tform = PiecewiseAffineTransform()
tform.estimate(dst,src)

out_rows = img.shape[0] - np.round(n,1) * 50
out_cols = cols
out = warp(img, tform, output_shape=(out_rows, out_cols))

#fig, ax = plt.subplots()
cv2.imshow('out',out)
out = cv2.convertScaleAbs(out, alpha=(255.0))
cv2.imwrite('../results/piece-affine-results/piece-affine.jpg', out)
cv2.waitKey(0)
cv2.destroyAllWindows()
#fig, ax = plt.subplots()
#ax.plot(tform.inverse(src)[:, 0], tform.inverse(src)[:, 1],'.b')
#ax.plot(tform(src)[:, 0], tform(src)[:, 1],'.r')
#ax.axis((0, out_cols, 2*out_rows, -out_rows))
#plt.show()
