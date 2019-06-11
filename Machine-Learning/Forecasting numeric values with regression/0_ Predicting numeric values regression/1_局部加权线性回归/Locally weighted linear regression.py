# -*- coding:utf-8 -*-
from matplotlib.font_manager import FontProperties
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
函数说明:使用局部加权线性回归计算回归系数w
    Parameters:
        testPoint - 测试样本点
        xArr - x数据集
        yArr - y数据集
        k - 高斯核的k,自定义参数
    Returns:
        ws - 回归系数

"""
def lwlr(testPoint, xArr, yArr, k = 1.0):

    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    # 创建权重对角矩阵
    weights = np.mat(np.eye((m)))
    # 遍历数据集计算每一个样本的权重
    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = np.exp(diffMat * diffMat.T/(-2.0 * k**2))
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0.0:
        print("矩阵为奇异矩阵,不能求逆")
        return
    # 计算回归系数
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

"""
函数说明:局部加权线性回归测试
    Parameters:
        testArr - 测试数据集
        xArr - x数据集
        yArr - y数据集
        k - 高斯核的k,自定义参数
    Returns:
        ws - 回归系数
"""
def lwlrTest(testArr, xArr, yArr, k=1.0):
    # 计算测试数据集大小
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    # 对每一个测试集上面的样本点都进行预测
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat


"""
函数说明:绘制多条局部加权回归曲线

"""
def plotlwlrRegression():
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    # 加载数据集
    xArr, yArr = loadDataSet('ex0.txt')
    # 根据局部加权线性回归计算yHat
    # 下面三个都是，只是高斯核不一样
    yHat_1 = lwlrTest(xArr, xArr, yArr, 1.0)
    yHat_2 = lwlrTest(xArr, xArr, yArr, 0.01)
    yHat_3 = lwlrTest(xArr, xArr, yArr, 0.003)
    xMat = np.mat(xArr)  # 创建xMat矩阵
    yMat = np.mat(yArr)  # 创建yMat矩阵
    # 排序，并返回索引值
    # argsort()函数是将x中的元素从小到大排列，
    # 提取其对应的index(索引)，然后输出到y
    srtInd = xMat[:, 1].argsort(0)
    xSort = xMat[srtInd][:, 0, :]
    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=False, sharey=False, figsize=(10, 8))
    axs[0].plot(xSort[:, 1], yHat_1[srtInd], c='red')  # 绘制回归曲线
    axs[1].plot(xSort[:, 1], yHat_2[srtInd], c='red')  # 绘制回归曲线
    axs[2].plot(xSort[:, 1], yHat_3[srtInd], c='red')  # 绘制回归曲线
    # 绘制样本点
    axs[0].scatter(xMat[:, 1].flatten().A[0], yMat.flatten().A[0], s=20, c='blue', alpha=.5)
    axs[1].scatter(xMat[:, 1].flatten().A[0], yMat.flatten().A[0], s=20, c='blue', alpha=.5)
    axs[2].scatter(xMat[:, 1].flatten().A[0], yMat.flatten().A[0], s=20, c='blue', alpha=.5)
    # 设置标题,x轴label,y轴label
    axs0_title_text = axs[0].set_title(u'局部加权回归曲线,k=1.0', FontProperties=font)
    axs1_title_text = axs[1].set_title(u'局部加权回归曲线,k=0.01', FontProperties=font)
    axs2_title_text = axs[2].set_title(u'局部加权回归曲线,k=0.003', FontProperties=font)
    plt.setp(axs0_title_text, size=8, weight='bold', color='red')
    plt.setp(axs1_title_text, size=8, weight='bold', color='red')
    plt.setp(axs2_title_text, size=8, weight='bold', color='red')
    plt.xlabel('X')
    plt.show()


if __name__ == '__main__':
    plotlwlrRegression()
'''
当k越小，拟合效果越好。但是当k过小，会出现过拟合的情况，
例如k等于0.003的时候
'''