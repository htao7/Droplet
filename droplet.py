import cv2
from matplotlib import pyplot as plt
import numpy as np
import math

MIN_SIZE = 500

def Seg(cont,single_area,canvas_size):
    for conti in cont:
        canvas = np.zeros(canvas_size, np.uint8)
        if cv2.contourArea(conti) > 0.5 * single_area:
            cv2.drawContours(canvas, [conti], 0, 255, cv2.FILLED)
            dist = cv2.distanceTransform(canvas, cv2.DIST_L2, 5)
            _, cores = cv2.threshold(dist, 0.8 * dist.max(), 255, cv2.THRESH_BINARY)
            cores = np.uint8(cores)
            # cv2.imshow('cores',cores)
            # cv2.waitKey(0)
            _, cores_cont, _ = cv2.findContours(cores, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for corei in cores_cont:
                M = cv2.moments(corei)
                if (M['m00'] == 0):
                    continue
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                cv2.circle(canvas,(cx,cy),2,1,-1)

                rou_list = []
                for angle in range(1,360):
                    rou = 2
                    theta = math.radians(angle)
                    while (canvas[int(cy + rou * math.cos(theta))][int(cx + rou * math.sin(theta))] != 0):
                        rou +=1
                    rou_list.extend([rou])

                fft_val = np.fft.fft(rou_list)

                plt.subplot(211),plt.imshow(canvas)
                plt.subplot(212),plt.plot(range(1,360),fft_val)
                plt.show()







img0 = cv2.imread('1.png')
im = np.copy(img0)
im2 = np.copy(img0)
img = cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY)
# img = cv2.bilateralFilter(img,9,50,50)
# img = cv2.GaussianBlur(img,(5,5),0)
img = cv2.medianBlur(img,5)
img_edge = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,5,2)
img_sharpened = cv2.bitwise_and(img,img_edge)

# plt.hist(img.ravel(),256,[0,255])
# plt.show()

_,img_binary = cv2.threshold(img_sharpened,120,255,cv2.THRESH_BINARY_INV)

_,cont,_= cv2.findContours(img_binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)



cont_num = 0
area_list = []
for conti in cont:
    if cv2.contourArea(conti) > MIN_SIZE:
        area_list.extend([cv2.contourArea(conti)])
        cont_num += 1
single_area = np.median(area_list)

droplets = Seg(cont,single_area,img0.shape[0:2])

cont_convex = [cv2.convexHull(conti) for conti in cont]
cv2.drawContours(im, cont_convex, -1, (0,0,255), 1)
cv2.drawContours(im2, cont, -1, (255,0,0), 1)


cv2.imshow('img_sharpened',img_binary)
# cv2.imshow('cont',im2)
# cv2.imshow('cont_convex',im)


cv2.waitKey(0)

