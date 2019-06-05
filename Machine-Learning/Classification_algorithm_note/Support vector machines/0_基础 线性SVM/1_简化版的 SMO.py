# -*- coding:UTF-8 -*-
from time import sleep
import matplotlib.pyplot as plt
import numpy as np
import random
import types

"""
函数说明:读取数据

Parameters:
    fileName - 文件名
Returns:
    dataMat - 数据矩阵
    labelMat - 数据标签

 这里的对数据集的处理，和前面都是大同小异，没有什么区别
 可以把前面的直接拿过来用就行了   
"""


def loadDataSet(fileName):
    dataMat = [];
    labelMat = []
    fr = open(fileName)
    # 逐行读取数据，
    for line in fr.readlines():
        # 滤除空格等
        lineArr = line.strip().split('\t')
        # 添加数据
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        # 添加标签，下标为2的也就是最后一列就是类标签1，-1
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

'''
selectJrand（）
随机选择alpha
i 表示就是= alpha
m 表示就是= alpha参数个数
返回的j就是表示 = 最终的alpha值

'''
def selectJrand(i, m):
    # 选择一个不等于i的j
    j = i
    while (j == i):
        j = int(random.uniform(0, m))
    return j

'''
clipAlpha()
就是 修剪alpha值
aj = 输入的alpha
H =  alpha的上限
L = alpha的下限
最终的返回值也就是 aj =alpha值

'''
def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

'''
smoSimple()
就是简化版的SMO算法（序列最小化优化算法）

'''
# 输入的参数就是数据集的前三列dataMatIN， classLabels 类标签列
# c代表的就是一个衡量  constant
# tolerance = toler 表示的就是 容错率
# maxIter 最大的迭代次数
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    # 转换为numpy的mat存储
    # np.mat 也就是表示的数组，只是他和array所构成的数组还是有一些区别的
    # mat 和array函数生成的矩阵还是有一些小小的不同
    dataMatrix = np.mat(dataMatIn); labelMat = np.mat(classLabels).transpose()
    # 初始化b参数，这里的b也可以理解为 Wo 就是逻辑回归里面的那样
    # 统计dataMatrix的维度
    # m表示的就是 行，n表示的就是列
    b = 0; m,n = np.shape(dataMatrix)
    #初始化alpha参数，设为0
    # 表示的就是该为一个m行1列的向量
    alphas = np.mat(np.zeros((m,1)))
    #初始化迭代次数
    iter_num = 0
    #最多迭代matIter次
    while (iter_num < maxIter):
        # 该变量就是表示优化的次数
        alphaPairsChanged = 0
        for i in range(m):
            # 步骤1：计算误差Ei
            # 下面的公式就是书中的公式  .T 就是表示矩阵的转置
            # mat函数所构成的矩阵是可以直接运用 * 号，表示乘法
            # np.multiply 该乘法表示的就是矩阵对应元素之间的乘法
            # np.dot 这就是真正意义上的矩阵乘法了
            fXi = float(np.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b
            # 这就是计算误差的公式
            Ei = fXi - float(labelMat[i])
            # 优化alpha，更设定一定的容错率，这就是书中的的限制条件
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                # 随机选择另一个与alpha_i成对优化的alpha_j
                j = selectJrand(i,m)
                # 步骤1：计算误差Ej
                fXj = float(np.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                # 保存更新前的aplpha值，使用深拷贝
                alphaIold = alphas[i].copy(); alphaJold = alphas[j].copy();
                # 步骤2：计算上下界H和L
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H: print("L==H"); continue
                # 步骤3：计算eta
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0: print("eta>=0"); continue
                #步骤4：更新alpha_j
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                #步骤5：修剪alpha_j
                alphas[j] = clipAlpha(alphas[j],H,L)
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print("alpha_j变化太小");
                    continue
                #步骤6：更新alpha_i
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])
                #步骤7：更新b_1和b_2
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                #步骤8：根据b_1和b_2更新b
                if (0 < alphas[i]) and (C > alphas[i]): b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                else: b = (b1 + b2)/2.0
                #统计优化次数
                alphaPairsChanged += 1
                #打印统计信息
                print("第%d次迭代 样本:%d, alpha优化次数:%d" % (iter_num,i,alphaPairsChanged))
        #更新迭代次数
        if (alphaPairsChanged == 0): iter_num += 1
        else: iter_num = 0
        print("迭代次数: %d" % iter_num)
    return b,alphas


'''
get_w函数
就是用来计算直线的法向量的

'''
def get_w(dataMat, labelMat, alphas):
    alphas, dataMat, labelMat = np.array(alphas), np.array(dataMat), np.array(labelMat)
    w = np.dot((np.tile(labelMat.reshape(1, -1).T, (1, 2)) * dataMat).T, alphas)
    # 将数组转换为列表，这里就是返回w的列表形式
    return w.tolist()


'''
showClassifer()
就是将分类结果可视化的函数
dataMat = 数据集矩阵
w = 直线法向量
b = 直线截距

'''
def showClassifer(dataMat, w, b):
    #绘制样本点
    # 表示正样本的空列表
    data_plus = []
    # 表示负样本的空列表
    data_minus = []
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    # 将其转换为有序的numpy矩阵
    data_plus_np = np.array(data_plus)
    data_minus_np = np.array(data_minus)
    # 这就是 绘制正样本和负样本散点图
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1], s=30, alpha=0.7)
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1], s=30, alpha=0.7)
    #绘制直线
    # 选取矩阵的第一列也就是下标为0的那一列
    x1 = max(dataMat)[0]
    x2 = min(dataMat)[0]
    a1, a2 = w
    b = float(b)
    a1 = float(a1[0])
    a2 = float(a2[0])
    y1, y2 = (-b- a1*x1)/a2, (-b - a1*x2)/a2
    plt.plot([x1, x2], [y1, y2])
    #找出支持向量点
    for i, alpha in enumerate(alphas):
        if abs(alpha) > 0:
            x, y = dataMat[i]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='red')
    plt.show()



if __name__ == '__main__':
    dataMat, labelMat = loadDataSet('testSet.txt')
    b,alphas = smoSimple(dataMat, labelMat, 0.6, 0.001, 40)
    w = get_w(dataMat, labelMat, alphas)
    showClassifer(dataMat, w, b)
    print(b)
