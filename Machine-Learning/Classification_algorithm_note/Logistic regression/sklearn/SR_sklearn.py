# -*- coding:UTF-8 -*-
from sklearn.linear_model import LogisticRegression

"""
函数说明:使用Sklearn构建Logistic回归分类器


"""
def colicSklearn():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    testSet = []; testLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))
    for line in frTest.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        testSet.append(lineArr)
        testLabels.append(float(currLine[-1]))
    # 这里的solver参数就是最优化参数选项
    # 这里选择的就是liblinear，当然火灾后面进行不同的选择算法
    # max_iter表示的就是最大的迭代次数
    #classifier = LogisticRegression(solver='liblinear',max_iter=10).fit(trainingSet, trainingLabels)

    '''
    下面就是选择sag方法，随机梯度下降方法
    同时sag对我们的最大迭代次数是由很高的要求
    所以我们将max_iter设置的比较大一点5000
    才会使sag收敛
    '''
    classifier = LogisticRegression(solver='sag', max_iter=5000).fit(trainingSet, trainingLabels)
    test_accurcy = classifier.score(testSet, testLabels) * 100
    print('正确率:%f%%' % test_accurcy)

if __name__ == '__main__':
    colicSklearn()