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
函数说明:计算回归系数w
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

"""
误差大小评价函数
    Parameters:
        yArr - 真实数据
        yHatArr - 预测数据
    Returns:
        误差大小
"""
# describing the error of our estimate
def rssError(yArr, yHatArr):
    return ((yArr - yHatArr) **2).sum()



if __name__ == '__main__':
    abX, abY = loadDataSet('abalone.txt')
    #print(abX)
    #print("***" * 80)
    print('训练集与测试集相同:局部加权线性回归,核k的大小对预测的影响:')
    yHat01 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
    yHat1 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
    yHat10 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)
    print('k=0.1时,误差大小为:',rssError(abY[0:99], yHat01.T))
    print('k=1  时,误差大小为:',rssError(abY[0:99], yHat1.T))
    print('k=10 时,误差大小为:',rssError(abY[0:99], yHat10.T))
    print('')
    print('训练集与测试集不同:局部加权线性回归,核k的大小是越小越好吗？更换数据集,测试结果如下:')
    yHat01 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)
    yHat1 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)
    yHat10 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)
    print('k=0.1时,误差大小为:',rssError(abY[100:199], yHat01.T))
    print('k=1  时,误差大小为:',rssError(abY[100:199], yHat1.T))
    print('k=10 时,误差大小为:',rssError(abY[100:199], yHat10.T))
    print('')
    print('训练集与测试集不同:简单的线性归回与k=1时的局部加权线性回归对比:')
    print('k=1时,误差大小为:', rssError(abY[100:199], yHat1.T))
    # 就是简单线性回归-线性回归系数
    ws = standRegres(abX[0:99], abY[0:99])
    # 表示简单线性回归的预测值
    yHat = np.mat(abX[100:199]) * ws
    print('简单的线性回归误差大小:', rssError(abY[100:199], yHat.T.A))

'''
可以看到，当k=0.1时，训练集误差小，但是应用于新的数据集之后，误差反而变大了。
这就是经常说道的过拟合现象。我们训练的模型，我们要保证测试集准确率高，这样训
练出的模型才可以应用于新的数据，也就是要加强模型的普适性。可以看到，当k=1时，
局部加权线性回归和简单的线性回归得到的效果差不多。这也表明一点，必须在未知数据
上比较效果才能选取到最佳模型。那么最佳的核大小是10吗？或许是，但如果想得到更好
的效果，应该用10个不同的样本集做10次测试来比较结果。
'''