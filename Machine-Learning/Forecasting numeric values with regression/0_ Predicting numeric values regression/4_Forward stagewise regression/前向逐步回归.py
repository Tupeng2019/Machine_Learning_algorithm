'''
前向逐步线性回归算法属于一种贪心算法，即每一步都尽可能减少误差。
我们计算回归系数，不再是通过公式计算，而是通过每次微调各个回归系数，
然后计算预测误差。那个使误差最小的一组回归系数，就是我们需要的最佳回归系数
'''
# -*-coding:utf-8 -*-
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
    # 这就按行读取，因为数据集中有一列是y值，所以会见减去一个1
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
函数说明:数据标准化
    Parameters:
        xMat - x数据集
        yMat - y数据集
    Returns:
        inxMat - 标准化后的x数据集
        inyMat - 标准化后的y数据集

"""
def regularize(xMat, yMat):
    # 数据拷贝
    inxMat = xMat.copy()
    inyMat = yMat.copy()
    # 进行行与行的操作，求平均值
    yMean = np.mean(yMat, 0)  #
    # 数据减去均值
    inyMat = yMat - yMean
    # 求均值
    inMeans = np.mean(inxMat, 0)
    # 求方差
    inVar = np.var(inxMat, 0)
    # 数据先取均值除以方差，从而实现标准化
    inxMat = (inxMat - inMeans) / inVar
    return inxMat, inyMat

"""
函数说明:计算平方误差
    Parameters:
        yArr - 预测值
        yHatArr - 真实值
"""
def rssError(yArr, yHatArr):
    return ((yArr - yHatArr) ** 2).sum()


"""
函数说明:前向逐步线性回归
    Parameters:
        xArr - x输入数据
        yArr - y预测数据
        eps - 每次迭代需要调整的步长,the step size to take at esch itweation
        numIt - 迭代次数
    Returns:
        returnMat - numIt次迭代的回归系数矩阵
"""
def stageWise(xArr, yArr, eps=0.01, numIt=100):
    # 读取数据集
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    # 调用函数，使其数据标准化，既是0均值，1方差
    xMat, yMat = regularize(xMat, yMat)
    # 获取XMat剧组的行列数据
    m, n = np.shape(xMat)
    # 初始化numIt迭代的回归系数矩阵
    returnMat = np.zeros((numIt, n))
    # 初始化回归系数矩阵
    ws = np.zeros((n, 1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    # 遍历迭代次数
    for i in range(numIt):
        # print(ws.T)                                                                    #打印当前回归系数矩阵
        lowestError = float('inf');  # 正无穷
        # 遍历每个特征的回归系数
        for j in range(n):  #
            for sign in [-1, 1]:
                wsTest = ws.copy()
                # 微调回归系数
                wsTest[j] += eps * sign
                # 计算预测值
                yTest = xMat * wsTest
                # 利用公式计算平方误差
                rssE = rssError(yMat.A, yTest.A)
                # 如果误差很小，则更新当亲的最佳回归系数
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        # 记录所有迭代次数后的回归系数矩阵
        returnMat[i, :] = ws.T
    return returnMat

"""
函数说明:绘制岭回归系数矩阵
"""
def plotstageWiseMat():
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    xArr, yArr = loadDataSet('abalone.txt')
    returnMat = stageWise(xArr, yArr, 0.005, 1000)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(returnMat)
    ax_title_text = ax.set_title(u'前向逐步回归:迭代次数与回归系数的关系', FontProperties=font)
    ax_xlabel_text = ax.set_xlabel(u'迭代次数', FontProperties=font)
    ax_ylabel_text = ax.set_ylabel(u'回归系数', FontProperties=font)
    plt.setp(ax_title_text, size=15, weight='bold', color='red')
    plt.setp(ax_xlabel_text, size=10, weight='bold', color='black')
    plt.setp(ax_ylabel_text, size=10, weight='bold', color='black')
    plt.show()


if __name__ == '__main__':
    plotstageWiseMat()

'''
我们打印了迭代次数与回归系数的关系曲线。可以看到，有些系数从始至终都是约为0的，
这说明它们不对目标造成任何影响，也就是说这些特征很可能是不需要的。逐步线性回归
算法的优点在于它可以帮助人们理解有的模型并做出改进。当构建了一个模型后，可以运
行该算法找出重要的特征，这样就有可能及时停止对那些不重要特征的收集。

'''

'''
缩减方法（逐步线性回归或岭回归），就是将一些系数缩减成很小的值或者直接缩减为0。
这样做，就增大了模型的偏差（减少了一些特征的权重），通过把一些特征的回归系数缩
减到0，同时也就减少了模型的复杂度。消除了多余的特征之后，模型更容易理解，同时也
降低了预测误差。但是当缩减过于严厉的时候，就会出现过拟合的现象，即用训练集预测
结果很好，用测试集预测就糟糕很多。

'''



































