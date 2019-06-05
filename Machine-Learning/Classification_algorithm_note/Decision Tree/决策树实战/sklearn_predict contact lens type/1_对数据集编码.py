from sklearn import tree
import pandas as pd


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
    # 以制度的方式打开文件，即加载文件
    with open('lenses.txt', 'r') as fr:
        # 处理文件
        lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    # 创建一个新的列表，用来收集数据集中过得每一组数据
    lenses_target = []
    #提取每组数据的类别，保存在列表里
    for each in lenses:
        lenses_target.append(each[-1])
    # 表示数据集中的特征标签
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    # 保存lenses数据的临时列表
    lenses_list = []
    # 创建一个字典，用于保存lenses数据的，用于pandas数组
    lenses_dict = {}
    # 提取信息，对应于生成字典，就相当于每一个标签对应于一列数据
    for each_label in lensesLabels:
        for each in lenses:
            lenses_list.append(each[lensesLabels.index(each_label)])
        lenses_dict[each_label] = lenses_list
        lenses_list = []
    # 打印字典信息
    print(lenses_dict)
    print('*'* 50)
    # 将lenses_dict 中的数据转换为pandas数组，(即生成pandas.Dataframe)
    lenses_pd = pd.DataFrame(lenses_dict)
    print(lenses_pd)
