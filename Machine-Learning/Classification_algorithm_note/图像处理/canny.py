
import cv2
# canny算子
img = cv2.imread('lena.jpg', 0)
# 用高斯滤波处理图像，去除噪声
blur = cv2.GaussianBlur(img, (3, 3), 0)
# 50是最小阈值,150是最大阈值
canny = cv2.Canny(blur, 60, 200)
cv2.imshow('canny', canny)
cv2.waitKey(0)
cv2.destroyAllWindows()


