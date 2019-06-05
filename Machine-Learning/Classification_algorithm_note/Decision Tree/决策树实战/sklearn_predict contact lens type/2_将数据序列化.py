import pandas as pd
from  sklearn.preprocessing import LabelEncoder

import pydotplus
from sklearn.externals.six import StringIO

if __name__ == '__main__':
    # 加载文件
    with open('lenses.txt', 'r') as fr:
        # 处理文件
        lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    # 提取每组数据的类型，保存在新的列表中
    lenses_target = []
    for each in lenses:
        lenses_target.append(each[-1])
    # 数据集的特征标签
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    # # 保存lenses数据的临时列表
    lenses_list = []
    # 创建一个字典，用于保存lenses数据的，用于pandas数组
    lenses_dict = {}
    # 提取信息，对应于生成字典，就相当于每一个标签对应于一列数据
    for each_label in lensesLabels:
        for each in lenses:
            lenses_list.append(each[lensesLabels.index(each_label)])
        lenses_dict[each_label] = lenses_list
        lenses_list = []
    # 打印处理后的字典信息，在这里不需要打印
    # print(lenses_dict)
    # 生成pandas数据结构，DataFrame
    # lenses_pd 的数据就是有序的数据了
    lenses_pd = pd.DataFrame(lenses_dict)
    # 打印pandas.Dataframe
    print(lenses_pd)
    # 将LabelEncoder（）实例化，即创建对象，用于序列化
    le = LabelEncoder()
    # columns= 列，这就是将每一列都进行序列化
    # 就是将标签下的数据进行数字化，用0,1,2来表示

    '''
    fit_transform(trainData)对部分数据先拟合fit，找到该part的整体指标，
    如均值、方差、最大值最小值等等（根据具体转换的目的），然后对该trainData
    进行转换transform，从而实现数据的标准化、归一化等等。
    下面就是对fit函数的一写解释：
    
    fit():  Method calculates the parameters μ and σ and saves them as internal objects.
      解释：简单来说，就是求得训练集X的均值，方差，最大值，最小值,这些训练集X固有的属性。
    transform(): Method using these calculated parameters apply the transformation to 
                 a particular dataset.
           解释：在fit的基础上，进行标准化，降维，归一化等操作（看具体用的是哪个工具，
           如PCA，StandardScaler等）。

    '''
    for col in lenses_pd.columns:
        lenses_pd[col] = le.fit_transform(lenses_pd[col])
    # 最终打印该序列化后的数据
    print(lenses_pd)