import cv2
import matplotlib.pyplot as plt
import numpy as np
import utils

img = cv2.imread('skeleton_orig.jpg', 0)

cv2.imshow('orig', img)
# диагональная маска
kernel = np.ones((3, 3))
kernel[1,1] = -8

laplassian_64bit = cv2.filter2D(img, cv2.CV_64F, kernel)
cv2.imshow('laplassian', utils.standart(laplassian_64bit))

gradcor_64bit = utils.gradation_cor(laplassian_64bit, 255)
cv2.imshow('gradcor', utils.standart(gradcor_64bit))
utils.save_image(utils.standart(gradcor_64bit), 'second_step.jpg')

cv2.imshow('b', utils.standart(np.int64(img) - 1 * laplassian_64bit))
utils.save_image(utils.standart(np.int64(img) - 1 * laplassian_64bit), 'third_step.jpg')


sobel = utils.sobel_XY(img)
cv2.imshow('sobel', sobel)
utils.save_image(sobel, 'sobel_step.jpg')


sobel_smoothed = utils.linear_smooth(sobel, method='average', width=5, height=5)
cv2.imshow('smooth', sobel_smoothed)
utils.save_image(sobel_smoothed, 'smooth_step.jpg')


laplacian_sobel = cv2.bitwise_and(utils.standart(np.int64(img) - 1 * laplassian_64bit), sobel_smoothed)
cv2.imshow('laplac_smooth', laplacian_sobel)
utils.save_image(laplacian_sobel, 'lapl_sobel_step.jpg')

pred_final = utils.standart(np.int64(img) + laplacian_sobel)
cv2.imshow('pred_final', pred_final)
utils.save_image(pred_final, 'pred_final_step.jpg')


pred_final = utils.standart(np.int64(img) + laplacian_sobel)
cv2.imshow('pred_final', pred_final)
utils.save_image(pred_final, 'pred_final_step.jpg')


final = utils.gamm(pred_final, 255, 0.5)
cv2.imshow('final', final)
utils.save_image(final, 'final_step.jpg')

cv2.waitKey()