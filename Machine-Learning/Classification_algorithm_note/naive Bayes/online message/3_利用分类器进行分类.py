import numpy as np


# 这就是标记化文档，vocabulary
def loadDataSet():
    # 表示切分的词条
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 类别标签向量，1代表侮辱性词汇，0代表不是
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec


# 函数createVocabList（）将创建所有文档中所有唯一单词的列表
def createVocabList(dataSet):
    # 创建一个空的不重复的列表
    vocabSet = set([])
    for document in dataSet:
        # Create the union of two sets创建两个列表的联合，就是取并集
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


# 遍历词汇列表，输出数字向量（1代表存在在词汇列表中，0表示不在vocaSet
def setOfWords2Vec(vocabList, inputSet):
    # 创建于vocabList维度一样的列表，只是全部都是0
    returnVec = [0] * len(vocabList)
    # 遍历每一个词汇
    for word in inputSet:
        # 如果词条存在于词汇表当中，就变为1
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: {0}is not in my Vocabulary!".format(word))
    # 返回一个文档向量
    return returnVec


'''
函数说明:朴素贝叶斯分类器训练函数

Parameters:
    trainMatrix - 训练文档矩阵，即setOfWords2Vec返回的returnVec构成的矩阵
    trainCategory - 训练类别标签向量，即loadDataSet返回的classVec
Returns:
    p0Vect - 侮辱类的条件概率数组
    p1Vect - 非侮辱类的条件概率数组
    pAbusive - 文档属于侮辱类的概率

'''


# 朴素贝叶斯分类器-训练函数
def trainNB0(trainMatrix, trainCategory):
    # trainMatrix 训练文档矩阵，就是扇面函数返回的returnVec构成的矩阵
    # 计算训练的文档数目
    numTrainDocs = len(trainMatrix)
    # 计算每一篇文档的词条数目
    numWords = len(trainMatrix[0])
    # 所有文档属中，属于侮辱类的概率
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # 创建numpy.zero数组，表示词条出现的次数，并且初始化为0
    # numerator 分子，denominator分母，都初始化为0
    '''
    使用拉普拉斯平滑将下面的初始化进行改进
    p0Num = np.zeros(numWords);p1Num = np.zeros(numWords)
    p0Denom = 0.0;p1Denom = 0.0
    '''
    p0Num = np.ones(numWords); p1Num = np.ones(numWords)
    p0Denom = 2.0; p1Denom = 2.0
    for i in range(numTrainDocs):
        # 统计属于侮辱类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)···
        # 这里统计的都是数字不是具体的数据
        if trainCategory[i] == 1:
            # 向量加法
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:  # 统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)···
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 元素的划分
    # p1Vect就是属于侮辱类的条件概率数组
    '''
    利用取对数的方式处理概率的下溢的问题解决：
    p1Vect = p1Num / p1Denom
    # 属于非侮辱类的条件概率数组
    p0Vect = p0Num / p0Denom
    '''
    p1Vect = np.log(p1Num / p1Denom)
    # 属于非侮辱类的条件概率数组
    p0Vect = np.log(p0Num / p0Denom)
    # #返回属于侮辱类的条件概率数组，属于非侮辱类的条件概率数组，文档属于侮辱类的概率
    return p0Vect, p1Vect, pAbusive

'''
这就是朴素贝叶斯分类器分类函数
vec2Classify - 待分类的词条数组
p0Vec - 侮辱类的条件概率数组
p1Vec -非侮辱类的条件概率数组
pClass1 - 文档属于侮辱类的概率
'''
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # 对应元素相乘。logA * B = logA + logB，所以这里加上log(pClass1)
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

'''
这就是最后的测试函数，测试朴素贝叶斯分类器

'''
def testingNB():
    # 实例化，创建实验样本
    listOPosts,listClasses = loadDataSet()
    # 创建词汇表
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        # 将实验样本向量化
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    # 训练朴素贝叶斯分类器
    p0V,p1V,pAb = trainNB0(np.array(trainMat),np.array(listClasses))
    # 测试样本1
    testEntry = ['love', 'my', 'dalmation']
    # 将测试样本向量化
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    if classifyNB(thisDoc,p0V,p1V,pAb):
        # 执行分类并打印分类结果
        print(testEntry,'属于侮辱类')
    else:
        # 同上，都是执行分类
        print(testEntry,'属于非侮辱类')
        # 测试样本2
    testEntry = ['stupid', 'garbage']
    # 同样，将样本向量化
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    if classifyNB(thisDoc,p0V,p1V,pAb):
        print(testEntry,'属于侮辱类')
    else:
        print(testEntry,'属于非侮辱类')


if __name__ == '__main__':
    # 调用函数、
    testingNB()