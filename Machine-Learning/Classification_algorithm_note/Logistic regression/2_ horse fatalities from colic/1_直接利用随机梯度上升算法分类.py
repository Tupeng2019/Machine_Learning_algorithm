# -*- coding:UTF-8 -*-
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

# 定义sigmiod函数
def sigmoid(inx):
    return 1.0/(1 + np.exp(-inx))


'''
下面就是我们的随机梯度上升算法梯度上升算法的实现
stocGradAscent
'''
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    # 返回dataMatrix的大小。m为行数,n为列数。
    m,n = np.shape(dataMatrix)
    # 参数初始化
    weights = np.ones(n)
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
            # 将已经使用过得样本进行删除
            del(dataIndex[randIndex])
    # 返回权重
    return weights

def plotBestFit(weights):

    # 数据集的实例化，也就是加载数据
    dataMat, labelMat = loadDataSet()
    # 将数据转换为有序列的numpy中的array数组
    dataArr = np.array(dataMat)
    # 获取数据的个数，也就是行数
    n = np.shape(dataArr)[0]
    # 下面两个不同的列表就是分别计算正样本和负样本的
    # 正样本
    xcord1 = []; ycord1 = []
    # 负样本
    xcord2 = []; ycord2 = []
    # 根据数据集的标签进行分类
    for i in range(n):
        # 如果标签是1
        if int(labelMat[i]) == 1:
            # 将为1 的放入我们的正样本中
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            # 其他，既是为0时放入负样本中
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    # 绘制绘图框
    fig = plt.figure()
    # 添加子图subplot
    ax = fig.add_subplot(111)
    # 实现正样本的可视化
    ax.scatter(xcord1, ycord1, s = 20, c = 'red', marker = 's')
    # 实现负样本的可视化
    ax.scatter(xcord2, ycord2, s = 20, c = 'blue')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0]- weights[1] *x)/ weights[2]
    ax.plot(x,y)
    # 设置绘图框的title
    plt.title('BestFit')
    plt.xlabel('x'); plt.ylabel('y')
    plt.show()

if __name__ == '__main__':
    dataMat, labelMat = loadDataSet()
    weights = stocGradAscent1(np.array(dataMat), labelMat)
    plotBestFit(weights)