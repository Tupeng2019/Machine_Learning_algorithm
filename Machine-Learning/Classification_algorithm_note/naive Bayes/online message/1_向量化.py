


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

if __name__ == '__main__':
    # 进行实例化，将数据传给两个参数
    postingList, classVec = loadDataSet()
    print('postingList:\n',postingList)
    myVocabList = createVocabList(postingList)
    print('myVocabList:\n',myVocabList)
    trainMat = []
    for postinDoc in postingList:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    print('trainMat:\n', trainMat)
