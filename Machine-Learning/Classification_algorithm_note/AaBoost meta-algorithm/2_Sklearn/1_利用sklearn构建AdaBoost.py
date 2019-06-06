# -*-coding:utf-8 -*-
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


def loadDataSet(fileName):
    numFeat = len((open(fileName).readline().split('\t')))
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat
if __name__ == '__main__':
    dataArr, classLabels = loadDataSet('horseColicTraining2.txt')
    testArr, testLabelArr = loadDataSet('horseColicTest2.txt')
    # n_estimators 弱学习器的最大迭代次数，或者说最大的弱学习器的个数
    # max_depth 表示决策树的最大深度
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 2), algorithm = "SAMME", n_estimators = 10)
    bdt.fit(dataArr, classLabels)
    predictions = bdt.predict(dataArr)
    errArr = np.mat(np.ones((len(dataArr), 1)))
    print('训练集的错误率:%.3f%%' % float(errArr[predictions != classLabels].sum() / len(dataArr) * 100))
    predictions = bdt.predict(testArr)
    errArr = np.mat(np.ones((len(testArr), 1)))
    print('测试集的错误率:%.3f%%' % float(errArr[predictions != testLabelArr].sum() / len(testArr) * 100))


'''
使用DecisionTreeClassifier作为使用的弱分类器，使用AdaBoost算法训练分类器。
可以看到训练集的错误率为16.054%，测试集的错误率为：17.910%。更改n_estimators参数，
你会发现跟我们自己写的代码，更改迭代次数的效果是一样的。
n_enstimators参数过大，会导致过拟合。

'''