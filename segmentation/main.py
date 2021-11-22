import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('water.png')

cv2.imshow('orig', img)


gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,70,255,cv2.THRESH_BINARY_INV)



# noise removal
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
opening = cv2.morphologyEx(thresh,cv2.MORPH_GRADIENT,kernel)

cv2.imshow('noise', opening)

op = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
cv2.imshow('gray', gray)

# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the region of unknown with zero
markers[unknown==255] = 0

markers2 = cv2.watershed(img, markers)
img[markers2 == -1] = [255,0,0]


markers1 = cv2.watershed(op, markers)
op[markers1 == -1] = [255,0,0]

cv2.imshow('gragient result', op)
cv2.imshow('result', img)

cv2.waitKey()
