# -*- coding:UTF-8 -*-
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np
import random



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

def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))



def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)            #转换成numpy的mat
    labelMat = np.mat(classLabels).transpose()   #转换成numpy的mat,并进行转置
    m, n = np.shape(dataMatrix)        #返回dataMatrix的大小。m为行数,n为列数。
    alpha = 0.01              #移动步长,也就是学习速率,控制更新的幅度。
    maxCycles = 500                    #最大迭代次数
    weights = np.ones((n,1))
    weights_array = np.array([])
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)     #梯度上升矢量化公式
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
        weights_array = np.append(weights_array,weights)
    weights_array = weights_array.reshape(maxCycles,n)
    return weights.getA(),weights_array    #将矩阵转换为数组，并返回




def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    # 返回dataMatrix的大小。m为行数,n为列数。
    m,n = np.shape(dataMatrix)
    # 参数初始化
    weights = np.ones(n)
    # 存储每次更新的回归系数
    weights_array = np.array([])
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            # 降低alpha的大小，每次减小1/(j+i)。
            # 相比如前面所做的改变：Alpha在每一次迭代中，都进行了改变
            alpha = 4/(1.0+j+i)+0.01
            # 随机选取样本
            # 这里也是做出的改变：是随机的选择样本，而不是遍历所有的样本
            randIndex = int(random.uniform(0,len(dataIndex)))
            # 随机选取的一个样本，计算h
            # h就是代表z的含义，梯度上升矢量化公式
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            # 计算误差
            error = classLabels[randIndex] - h
            # 更新回归系数
            weights = weights + alpha * error * dataMatrix[randIndex]
            # 添加回归系数到数组当中去
            weights_array = np.append(weights_array, weights, axis=0)
            # 将已经使用过得样本进行删除
            del(dataIndex[randIndex])
    # 改变维度
    weights_array = weights_array.reshape(numIter * m, n)
    # 返回权重
    return weights, weights_array

"""
函数说明:绘制回归系数与迭代次数的关系

Parameters:
    weights_array1 - 回归系数数组1
    weights_array2 - 回归系数数组2

"""
def plotWeights(weights_array1,weights_array2):
    #设置汉字格式
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    #将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
    #当nrow=3,nclos=2时,代表fig画布被分为六个区域,axs[0][0]表示第一行第一列
    fig, axs = plt.subplots(nrows=3, ncols=2,sharex=False, sharey=False, figsize=(20,10))
    x1 = np.arange(0, len(weights_array1), 1)
    #绘制w0与迭代次数的关系
    axs[0][0].plot(x1,weights_array1[:,0])
    axs0_title_text = axs[0][0].set_title(u'梯度上升算法：回归系数与迭代次数关系',FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'W0',FontProperties=font)
    plt.setp(axs0_title_text, size=20, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    #绘制w1与迭代次数的关系
    axs[1][0].plot(x1,weights_array1[:,1])
    axs1_ylabel_text = axs[1][0].set_ylabel(u'W1',FontProperties=font)
    plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
    #绘制w2与迭代次数的关系
    axs[2][0].plot(x1,weights_array1[:,2])
    axs2_xlabel_text = axs[2][0].set_xlabel(u'迭代次数',FontProperties=font)
    axs2_ylabel_text = axs[2][0].set_ylabel(u'W2',FontProperties=font)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')


    x2 = np.arange(0, len(weights_array2), 1)
    #绘制w0与迭代次数的关系
    axs[0][1].plot(x2,weights_array2[:,0])
    axs0_title_text = axs[0][1].set_title(u'改进的随机梯度上升算法：回归系数与迭代次数关系',FontProperties=font)
    axs0_ylabel_text = axs[0][1].set_ylabel(u'W0',FontProperties=font)
    plt.setp(axs0_title_text, size=20, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    #绘制w1与迭代次数的关系
    axs[1][1].plot(x2,weights_array2[:,1])
    axs1_ylabel_text = axs[1][1].set_ylabel(u'W1',FontProperties=font)
    plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
    #绘制w2与迭代次数的关系
    axs[2][1].plot(x2,weights_array2[:,2])
    axs2_xlabel_text = axs[2][1].set_xlabel(u'迭代次数',FontProperties=font)
    axs2_ylabel_text = axs[2][1].set_ylabel(u'W2',FontProperties=font)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')

    plt.show()

if __name__ == '__main__':
    dataMat, labelMat = loadDataSet()
    weights1,weights_array1 = stocGradAscent1(np.array(dataMat), labelMat)

    weights2,weights_array2 = gradAscent(dataMat, labelMat)
    plotWeights(weights_array1, weights_array2)
