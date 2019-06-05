# -*- coding:UTF-8 -*-

import matplotlib.pyplot as plt
import numpy as np

"""
函数说明:读取数据

Parameters:
    fileName - 文件名
Returns:
    dataMat - 数据矩阵
    labelMat - 数据标签
    
 这里的对数据集的处理，和前面都是大同小异，没有什么区别
 可以把前面的直接拿过来用就行了   
"""

def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    # 逐行读取数据，
    for line in fr.readlines():
        # 滤除空格等
        lineArr = line.strip().split('\t')
        # 添加数据
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        # 添加标签
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

"""
函数说明:数据可视化

Parameters:
    dataMat - 数据矩阵
    labelMat - 数据标签
"""
def showDataSet(dataMat, labelMat):
    # 因为在SVM中，将数据集最终的分类为-1,1 两类，这就是不同于逻辑回归的0,1
    # 表示正样本数据集
    data_plus = []
    # 表示负样本数据集
    data_minus = []
    # 按行读取dataMat矩阵中的数据
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    # 将表示正样本的数据的列表转换为有顺序的numpy矩阵
    data_plus_np = np.array(data_plus)
    # 将表示正样本的数据的列表转换为有顺序的numpy矩阵
    data_minus_np = np.array(data_minus)
    # 这就是绘制正样本的散点图
    # scatter方法就是表示的散点图
    # transpose 就是做的是矩阵的转置，就是将原数据组的一列，变成现在的一行
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1])
    #绘制负样本的散点图
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1])
    plt.show()

if __name__ == '__main__':
    dataMat, labelMat = loadDataSet('testSet.txt')
    showDataSet(dataMat, labelMat)