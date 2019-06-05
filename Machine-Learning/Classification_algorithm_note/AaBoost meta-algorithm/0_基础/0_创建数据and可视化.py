# -*-coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

"""
创建单层决策树的数据集
Parameters:
    无
Returns:
    dataMat - 数据矩阵
    classLabels - 数据标签
"""
def loadSimpData():
    # 该数据集就是根据书上的写的，《machine Learning in action》
    datMat = np.matrix([[ 1. ,  2.1],
        [ 1.5,  1.6],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels

"""
数据可视化
        dataMat - 数据矩阵
        labelMat - 数据标签
"""
def showDataSet(dataMat, labelMat):

    data_plus = []        #正样本
    data_minus = []      #负样本
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    # 转换为numpy矩阵
    data_plus_np = np.array(data_plus)
    # 同上，转换为numpy矩阵
    data_minus_np = np.array(data_minus)
    # 绘制正样本散点图
    '''
    如果我们要对散点图进行设置
    就是对scatter函数的参数进行不同设置
    在这里有一篇博客可以进行参考
    https://blog.csdn.net/qiu931110/article/details/68130199
    '''
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1],c= 'b')
    # 绘制负样本散点图
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1],marker= 's',c = 'r')
    plt.show()

if __name__ == '__main__':
    dataArr,classLabels = loadSimpData()
    showDataSet(dataArr,classLabels)
