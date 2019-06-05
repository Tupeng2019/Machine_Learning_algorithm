
import numpy as np
import re
import random


# bigString就是指一个大的字符串，就是相当于邮件当中的英语文章
# 该函数的作用就是将字符串转化为字符列表
def textParse(bigString):
    # 了利用正则表达式
    # 将特殊符号作为切分标志进行字符串切分，即非字母、非数字
    # 在书中用的是W*，估计是Python3和2的区别吧，在这里使用的W
    listOfTokens = re.split(r'\W', bigString)
    # 只是将字符长大大于2的，全部变成小写，除了单个字符的，如I
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


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

# 朴素贝叶斯分类器-训练函数
def trainNB0(trainMatrix,trainCategory):
    # trainMatrix 训练文档矩阵，就是扇面函数返回的returnVec构成的矩阵
    # 计算训练的文档数目
    numTrainDocs = len(trainMatrix)
    # 计算每一篇文档的词条数目
    numWords = len(trainMatrix[0])
    # 所有文档属中，属于垃圾类的概率
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    # 创建numpy.zero数组，表示词条出现的次数，并且初始化为0
    # numerator 分子，denominator分母，都初始化为0
    p0Num = np.zeros(numWords); p1Num = np.zeros(numWords)
    p0Denom = 0.0; p1Denom = 0.0
    for i in range(numTrainDocs):
        #统计属于垃圾类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)···
        #这里统计的都是数字不是具体的数据
        if trainCategory[i] == 1:
            # 向量加法
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:#统计属于非垃圾类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)···
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 元素的划分
    # p1Vect就是属于垃圾类的条件概率数组
    p1Vect = p1Num/p1Denom
    # 属于非垃圾类的条件概率数组
    p0Vect = p0Num/p0Denom
    # #返回属于垃圾类的条件概率数组，属于非垃圾类的条件概率数组，文档属于垃圾类的概率
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # 对应元素相乘。logA * B = logA + logB，所以这里加上log(pClass1)
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def spamTest():
    docList = []; classList = []; fullText = []
    # 遍历25个TXT文件
    for i in range(1, 26):                        #遍历25个txt文件
        # 读取每一个垃圾邮件，并字符串转换成字符串列表
        wordList = textParse(open('email//spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.append(wordList)
        # 标记垃圾邮件，1表示垃圾文件
        classList.append(1)
        # 读取每个非垃圾邮件，并字符串转换成字符串列表
        wordList = textParse(open('email//ham/%d.txt' % i, 'r').read())
        docList.append(wordList)
        fullText.append(wordList)
        # 标记废垃圾邮件0表示非垃圾邮件
        classList.append(0)
    # 创建不重复的词汇表，
    vocabList = createVocabList(docList)
    # 创建储存训练集的索引值的列表，和测试集的索引值的列表
    trainingSet = list(range(50)); testSet = []
    '''
    我们将我们的数据集分一部分作为我们的训练集，一部分作为我们的测试集
    在这里，选择40WieU训练集，10个为测试集
    这里的数据集都是没有什么规律，所以就是随机选就行
    '''
    for i in range(10):
        # 随机选取索引值0-39
        randIndex = int(random.uniform(0, len(trainingSet)))
        # 添加测试集的索引值
        testSet.append(trainingSet[randIndex])
        # 在训练集列表中删除添加测试集的索引值
        del(trainingSet[randIndex])
    # 创建训练集矩阵和训练集类别标签向量
    trainMat = []; trainClasses = []
    # 遍历训练集
    for docIndex in trainingSet:
        # 将生成的词汇表模型添加到训练中
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        # 将类别添加掉训练集类别标签向量中
        trainClasses.append(classList[docIndex])
    # 训练朴素贝叶斯模型
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))
    # 这就是计算错误分类的的数量
    errorCount = 0
    # 现在开始遍历测试集
    for docIndex in testSet:
        # 测试的词汇模型
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        # 如果分类错误，
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            # 分类错误加+1
            errorCount += 1
            print("分类错误的测试集：",docList[docIndex])
    print('错误率：%.2f%%' % (float(errorCount) / len(testSet) * 100))

if __name__ =='__main__':
    spamTest()