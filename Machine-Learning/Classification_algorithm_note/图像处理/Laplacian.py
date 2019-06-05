

import cv2

# 拉普拉斯算子
img = cv2.imread('lena.jpg', 0)
laplacian = cv2.Laplacian(img, cv2.CV_16S, ksize=3)
dst = cv2.convertScaleAbs(laplacian)
cv2.imshow('img',img)
cv2.imshow('laplacian', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
