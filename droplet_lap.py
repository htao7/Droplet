import cv2
from matplotlib import pyplot as plt
import numpy as np
import math

MIN_SIZE = 300
FONT = cv2.FONT_HERSHEY_SIMPLEX


def Seg(cont,canvas_size):
    for conti in cont:
        canvas = np.zeros(canvas_size, np.uint8)
        cv2.drawContours(canvas, [conti], 0, 255, cv2.FILLED)
        dist = cv2.distanceTransform(canvas, cv2.DIST_L2, 5)
        _, cores = cv2.threshold(dist, 0.8 * dist.max(), 255, cv2.THRESH_BINARY)
        cores = np.uint8(cores)
        _, cores_cont, _ = cv2.findContours(cores, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cluster_num = len(cores_cont)
        M = cv2.moments(conti)
        centroid = (int(M['m10'] / M['m00']),int(M['m01'] / M['m00']))
        cv2.putText(im_cluster,str(cluster_num),centroid,FONT,0.5,(255,0,0),1,cv2.LINE_AA)


def Cal(cont,img):
    canvas_size = img.shape
    _,img = cv2.threshold(img,80,255,cv2.THRESH_BINARY)

    cv2.imshow('a',img)

    for conti in cont:
        canvas = np.zeros(canvas_size, np.uint8)
        cv2.drawContours(canvas, [conti], 0, 255, cv2.FILLED)
        white_area = np.sum(cv2.bitwise_and(img,canvas)) / 255
        black_area = cv2.contourArea(conti) - white_area
        M = cv2.moments(conti)
        centroid = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
        cv2.putText(im_single, "%.2f" % (white_area / black_area), centroid, FONT, 0.5, (255, 0, 0), 1, cv2.LINE_AA)



img0 = cv2.imread('1.png')
im_single = np.copy(img0)
im_cluster = np.copy(img0)
img = cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY)
# img = cv2.bilateralFilter(img,9,50,50)
# img = cv2.GaussianBlur(img,(5,5),0)
img = cv2.medianBlur(img,5)
img_edge = cv2.Laplacian(img,cv2.CV_8U,ksize=5)



# plt.hist(img.ravel(),256,[0,255])
# plt.show()

# img_sharpened = cv2.add(cv2.bitwise_not(img),img_edge)
_,img_thresh = cv2.threshold(img_edge,150,255,cv2.THRESH_BINARY)
kernel = np.ones((3,3),np.uint8)
img_thresh = cv2.morphologyEx(img_thresh,cv2.MORPH_CLOSE,kernel)
_,cont,_= cv2.findContours(img_thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

# cv2.imshow('a',img_thresh)



area_list = []
droplets = []
for conti in cont:
    hull_area = cv2.contourArea(cv2.convexHull(conti))
    if hull_area > MIN_SIZE:
        area_list.append(hull_area)
        droplets.append(conti)
single_area = np.median(area_list)

# print (droplets)


droplets_single = []
droplets_cluster = []
for conti in droplets:
    hull_area = cv2.contourArea(cv2.convexHull(conti))
    if hull_area < 2 * single_area:
        droplets_single.append(conti)
    else:
        droplets_cluster.append(conti)


droplets_single = [cv2.convexHull(d) for d in droplets_single]
cv2.drawContours(im_cluster, droplets_cluster, -1, (0,0,255), 1)
cv2.drawContours(im_single, droplets_single, -1, (0,0,255), 1)

Seg(droplets_cluster,img0.shape[0:2])
Cal(droplets_single,img)


# cv2.imshow('img_sharpened',img_binary)
cv2.imshow('singles',im_single)
cv2.imshow('cluster',im_cluster)

cv2.imwrite('single.png',im_single)
cv2.imwrite('cluster.png',im_cluster)


cv2.waitKey(0)

