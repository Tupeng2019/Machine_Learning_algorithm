import os
import pandas as pd
import requests



PATH = r'C:\Users\Tupeng\Desktop'
r= requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
with open(PATH + 'iris.data', 'w') as f:
    f.write(r.text)
os.chdir(PATH)

'''
这里的文件名不能是pandas.py，如果
两个名字相同了，这就会报错了
'''
df = pd.read_csv(PATH + 'iris.data', names=['sepal length', 'sepal width', 'petal length', 'petal width', 'class'])

# df = pd.read_csv()
df.head(5)

# 选择特定的列明，输出：
b = df['sepal length']
print(b)
print('*' * 50)

# 选择前四行，和前两列
# 后面的意思既是前面的数字是被包含的，后面的数字是不被包含的
c = df.ix[:3 , : 2]
print(c)

print('*' * 50 )
# 使用迭代器并且只选择描述width的列
d = df.ix[: 3, [x for x in df.columns if 'width' in x]]
print(d)

# 列出所有的可用的为一类，然后选择其中之一
e = df['class'].unique()
print(e)