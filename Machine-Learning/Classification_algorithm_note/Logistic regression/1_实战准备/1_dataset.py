# -*- coding:UTF-8 -*-
import matplotlib.pyplot as plt
import numpy as np



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

'''
下面就是我们事先数据可视化的步骤
plotDataSet函数就是实现数据集可视化的

'''
def plotDataSet():
    # 数据集的实例化，也就是加载数据
    dataMat, labelMat = loadDataSet()
    # 将数据转换为有序列的numpy中的array数组
    dataArr = np.array(dataMat)
    # 获取数据的个数，也就是行数
    n = np.shape(dataMat)[0]
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
    # 设置绘图框的title
    plt.title('DataSet')
    plt.xlabel('x'); plt.ylabel('y')
    plt.show()

if __name__ == '__main__':
    plotDataSet()
