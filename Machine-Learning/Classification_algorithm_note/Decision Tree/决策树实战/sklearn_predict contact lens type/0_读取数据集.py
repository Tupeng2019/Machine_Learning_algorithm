from sklearn import tree



'''
在我们使用数据集是，对其编码，有两种方法：
1. LabelEncoder：将字符窜转为增量值
2. OneHoeEncoder： 使用One-of-K算法将字符串转为整数
为了对string类型的数据序列化，需要先生成pandas数据，这样方便
我们的序列化工作。这里我使用的方法是，原始数据->字典->pandas数据，
编写代码如下：
这里的pandas数据就是一维带标签的数组，数组里可以放任意的
数据（整数、浮点数、字符串、python Object)等等
'''

if __name__ == '__main__':
    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    print(lenses)
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    clf = tree.DecisionTreeClassifier()
    # 因为我们打印的数据集是string型，而函数fit是不能接受的string
    # 所以在调用函数时，要先对数据集进行编码
    lenses = clf.fit(lenses, lensesLabels)
