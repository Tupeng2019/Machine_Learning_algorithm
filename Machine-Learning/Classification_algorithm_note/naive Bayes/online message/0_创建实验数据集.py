

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

if __name__ == '__main__':
    # 实例化
    postingList, classVec = loadDataSet()
    # 遍历PostingList中的单词
    for each in postingList:
        print(each)
    # 为了区分上下的内容
    print("*" * 50)
    print(classVec)
