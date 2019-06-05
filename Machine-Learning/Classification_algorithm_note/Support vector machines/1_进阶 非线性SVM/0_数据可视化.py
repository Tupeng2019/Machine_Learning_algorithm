# -*-coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np


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



def showDataSet(dataMat, labelMat):

    data_plus = []        #正样本
    data_minus = []       #负样本
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)      #转换为numpy矩阵
    data_minus_np = np.array(data_minus)     #转换为numpy矩阵
    # 绘制正样本散点图
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1])
    # 绘制负样本散点图
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1])
    plt.show()

if __name__ == '__main__':
    # 读取数据集，并且实现实例化
    dataArr,labelArr = loadDataSet('testSetRBF.txt')
    showDataSet(dataArr, labelArr)