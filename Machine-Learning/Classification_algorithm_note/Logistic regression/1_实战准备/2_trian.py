# -*- coding:UTF-8 -*-

import matplotlib.pyplot as plt
import numpy as np


def loadDataSet():
    # 创建数据列表，用来放入我们的数据
    dataMat = []
    # 该列表就是用来存放我们的标签的
    labelMat = []
    # 读取我们的数据集txt文件
    fr = open('testSet.txt')
    # 在读取文件数据集时，是按行读取
    for line in fr.readlines():
        # 将换行符去掉，既是去回车，放入列表
        lineArr = line.strip().split()
        # 向我们的数据列表中添加数据
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        # 同上，就是向标签列表中，添加标签
        labelMat.append(int(lineArr[2]))
    # 关闭数据集文件
    fr.close()
    # 返回我们的数据集列表和标签列表
    return dataMat, labelMat

# 定义sigmiod函数
def sigmoid(inx):
    return 1.0/(1 + np.exp(-inx))


'''
下面就是我们的梯度上升算法的实现
gradAscent
'''
def gradAscent(dataMatIn, classLabels):
    # 将数据集转换为numpy的mat
    dataMatrix = np.mat(dataMatIn)
    # 将标签转换成numpy的mat。并且进行转置
    # transpose 就是表示转置
    labelMat = np.mat(classLabels).transpose()
    # 返回dataMatrix的大小。m为行数,n为列数。
    m, n = np.shape(dataMatrix)
    # 移动步长,也就是学习速率,控制更新的幅度。
    alpha = 0.001
    # 最大迭代次数
    maxCycles = 500
    # 权重的初始化为0
    weights = np.ones((n,1))
    for k in range(maxCycles):
        # h就是代表z的含义，梯度上升矢量化公式
        h = sigmoid(dataMatrix * weights)
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
    # 将矩阵转换为数组，返回权重数组
    # 这里的最优权重数组就是所求的最佳的解
    return weights.getA()


if __name__ =='__main__':
    dataMat, labelMat = loadDataSet()
    matrix = gradAscent(dataMat,labelMat)
    print(matrix)
