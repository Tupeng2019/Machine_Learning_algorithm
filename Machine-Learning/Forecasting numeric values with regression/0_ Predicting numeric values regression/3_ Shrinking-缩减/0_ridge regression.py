'''
如果数据的特征比样本点还多应该怎么办？很显然，
此时我们不能再使用上文的方法进行计算了，因为矩阵X不是满秩矩阵，
非满秩矩阵在求逆时会出现问题。为了解决这个问题，
统计学家引入岭回归（ridge regression）的概念。
'''
'''
Shrinking coefficients to understand our data
 为了使用岭回归和缩减技术，首先需要对特征做标准化处理。因为，
 我们需要使每个维度特征具有相同的重要性。本文使用的标准化处理比较简单，
 就是将所有特征都减去各自的均值并除以方差。
'''
# -*-coding:utf-8 -*-
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np


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
    函数说明:岭回归
    Parameters:
        xMat - x数据集
        yMat - y数据集
        lam - 缩减系数
    Returns:
        ws - 回归系数
"""
def ridgeRegres(xMat, yMat, lam = 0.2):
    xTx = xMat.T * xMat
    denom = xTx + np.eye(np.shape(xMat)[1]) * lam
    if np.linalg.det(denom) == 0.0:
        print("矩阵为奇异矩阵,不能转置")
        return
    ws = denom.I * (xMat.T * yMat)
    return ws

"""
函数说明:岭回归测试
    Parameters:
        xMat - x数据集
        yMat - y数据集
    Returns:
        wMat - 回归系数矩阵
"""
# to test this over a number of lambda values
def ridgeTest(xArr, yArr):
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    # 数据标准化
    # 行与行的数据操作，mean就是求y均值的
    yMean = np.mean(yMat, axis = 0)
    # 用原数据减去均值
    yMat = yMat - yMean
    # 求x的均值
    xMeans = np.mean(xMat, axis = 0)
    # 求方差
    xVar = np.var(xMat, axis = 0)
    # 数据减去均值除以方差实现标准化
    xMat = (xMat - xMeans) / xVar
    # 30个不同的lambda测试
    numTestPts = 30
    # 初始化回归系数矩阵
    wMat = np.zeros((numTestPts, np.shape(xMat)[1]))
    # 改变lambda计算回归系数
    for i in range(numTestPts):
        # #lambda以e的指数变化，最初是一个非常小的数，
        ws = ridgeRegres(xMat, yMat, np.exp(i - 10))
        # 计算回归系数矩阵
        wMat[i, :] = ws.T
    return wMat

"""
函数说明:绘制岭回归系数矩阵
"""
def plotwMat():
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    abX, abY = loadDataSet('abalone.txt')
    redgeWeights = ridgeTest(abX, abY)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(redgeWeights)
    ax_title_text = ax.set_title(u'log(lambada)与回归系数的关系', FontProperties = font)
    ax_xlabel_text = ax.set_xlabel(u'log(lambada)', FontProperties = font)
    ax_ylabel_text = ax.set_ylabel(u'回归系数', FontProperties = font)
    plt.setp(ax_title_text, size = 20, weight = 'bold', color = 'red')
    plt.setp(ax_xlabel_text, size = 10, weight = 'bold', color = 'black')
    plt.setp(ax_ylabel_text, size = 10, weight = 'bold', color = 'black')
    plt.show()
if __name__ == '__main__':
    plotwMat()


    '''
    图绘制了回归系数与log(λ)的关系。在最左边，即λ最小时，可以得到所有系数的
    原始值（与线性回归一致）；而在右边，系数全部缩减成0；在中间部分的某个位置，
    将会得到最好的预测结果。想要得到最佳的λ参数，可以使用交叉验证的方式获得，
    文章的后面会继续讲解。
    '''