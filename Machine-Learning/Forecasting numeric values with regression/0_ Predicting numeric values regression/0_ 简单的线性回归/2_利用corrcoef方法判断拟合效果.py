# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

"""
函数说明:加载数据
Parameters:
    fileName - 文件名
Returns:
    xArr - x数据集
    yArr - y数据集

"""

def loadDataSet(fileName):
    '''
    Python 将文本文件的内容读入可以操作的字符串变量非常容易。文件对象提供了三个
    “读”方法： .read()、.readline() 和 .readlines()。每种方法可以接受一个变量
    以限制每次读取的数据量，但它们通常不使用变量。 .read() 每次读取整个文件，它
    通常用于将文件内容放到一个字符串变量中。然而 .read() 生成文件内容最直接的字
    符串表示，但对于连续的面向行的处理，它却是不必要的，并且如果文件大于可用内存
    ，则不可能实现这种处理。
     readline()就是每一次只读一行
     readlines() 自动将文件内容分析成一个行的列表

    '''
    numFeat = len(open(fileName).readline().split('\t')) - 1
    xArr = []; yArr = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        xArr.append(lineArr)
        yArr.append(float(curLine[-1]))
    return xArr, yArr

"""
函数说明:计算简单回归系数w
    Parameters:
        xArr - x数据集
        yArr - y数据集
    Returns:
        ws - 回归系数
"""
def standRegres(xArr,yArr):
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    # 根据文中推导的公式计算回归系数
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:
        print("矩阵为奇异矩阵,不能求逆")
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws

if __name__ == '__main__':
    # 加载数据集实现实例化
    xArr, yArr = loadDataSet('ex0.txt')
    #调用函数计算回归系数
    ws = standRegres(xArr, yArr)
    # 创建相应的矩阵
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    yHat = xMat * ws
    # 调用corrcoef方法，判断拟合的效果
    print(np.corrcoef(yHat.T, yMat))

'''
对角线上的数据是1.0，因为yMat和自己的匹配是完美的，
而YHat和yMat的相关系数为0.98。
(YHat 和yMat的相关系数越接近1越好）
'''