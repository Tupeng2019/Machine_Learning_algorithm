
import numpy as np


# 这就是标记化文档，vocabulary
def loadDataSet():
    # 表示切分的词条
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 类别标签向量，1代表侮辱性词汇，0代表不是
    classVec = [0,1,0,1,0,1]
    return postingList,classVec


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
        else: print("the word: {0}is not in my Vocabulary!".format(word))
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
def trainNB0(trainMatrix,trainCategory):
    # trainMatrix 训练文档矩阵，就是扇面函数返回的returnVec构成的矩阵
    # 计算训练的文档数目
    numTrainDocs = len(trainMatrix)
    # 计算每一篇文档的词条数目
    numWords = len(trainMatrix[0])
    # 所有文档属中，属于侮辱类的概率
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    # 创建numpy.zero数组，表示词条出现的次数，并且初始化为0
    # numerator 分子，denominator分母，都初始化为0
    p0Num = np.zeros(numWords); p1Num = np.zeros(numWords)
    p0Denom = 0.0; p1Denom = 0.0
    for i in range(numTrainDocs):
        #统计属于侮辱类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)···
        #这里统计的都是数字不是具体的数据
        if trainCategory[i] == 1:
            # 向量加法
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:#统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)···
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 元素的划分
    # p1Vect就是属于侮辱类的条件概率数组
    p1Vect = p1Num/p1Denom
    # 属于非侮辱类的条件概率数组
    p0Vect = p0Num/p0Denom
    # #返回属于侮辱类的条件概率数组，属于非侮辱类的条件概率数组，文档属于侮辱类的概率
    return p0Vect,p1Vect,pAbusive

if __name__ == '__main__':
    postingList, classVec = loadDataSet()
    myVocabList = createVocabList(postingList)
    print('myVocabList:\n', myVocabList)
    trainMat = []
    for postinDoc in postingList:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(trainMat, classVec)
    print('p0V:\n', p0V)
    print('p1V:\n', p1V)
    print('classVec:\n', classVec)
    print('pAb:\n', pAb)
