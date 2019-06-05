import cv2
import numpy as np
import random

# 读取图像
img = cv2.imread('lena.jpg')
# 获取图像的信息
imginfo = img.shape
# 获取图形的高度
height = imginfo[0]
# 获取图像的宽度
width = imginfo[1]
cv2.imshow('src', img)
# 将图像一直显示
# cv2.waitKey(0)

'''
使用边缘检测的方法，先用canny算子
1. 先做灰度处理
2. 用高斯滤波，去燥
3. 条用canny方法

'''
# 将RGB图像转换成灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 高斯滤波，两个方法、
# GaussIanBlur有两个参数，第二个是模板大小
imgG = cv2.GaussianBlur(gray, (3,3),0)
#装载数据, 第一个参数是专递的图像数据，第二个是图片的门限，
#  图片经过卷积运算之后如果这个点大于门限值，就认为是边缘点，否者就是非边缘点，
dst = cv2.Canny(imgG, 50, 50)
cv2.imshow('dst',dst)
cv2.waitKey(0)
