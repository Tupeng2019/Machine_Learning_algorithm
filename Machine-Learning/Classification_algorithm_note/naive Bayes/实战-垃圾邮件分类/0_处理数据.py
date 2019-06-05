
import re

# bigString就是指一个大的字符串，就是相当于邮件当中的英语文章
# 该函数的作用就是将字符串转化为字符列表
def textParse(bigString):
    # 了利用正则表达式
    # 将特殊符号作为切分标志进行字符串切分，即非字母、非数字
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

if __name__ == '__main__':
    docList = []; classList = []
    # 遍历ham，spam中的25个TXT文件
    for i in range(1, 26):
        #读取每一个垃圾邮件，并将字符串装换为字符串列表
        # r 既是表示的只读形式，读取文件
        wordList = textParse(open('email//spam/{0}.txt'.format(i),'r').read())
        # 并将其存放在docList找那个
        docList.append(wordList)
        # 标记每一个垃圾邮件，1就表示垃圾邮件，有一个就增加1
        classList.append(1)
        #读取每一个非垃圾邮件，并将字符串装换为字符串列表
        wordList = textParse(open('email/ham/{0}.txt'.format(i), 'r').read())
        docList.append(wordList)
        # 标记非垃圾邮件，0就是表示非垃圾邮件
        classList.append(0)
    # 调用createVocabList（），创建不重复词汇表
    vocabList = createVocabList(docList)

    print(vocabList)