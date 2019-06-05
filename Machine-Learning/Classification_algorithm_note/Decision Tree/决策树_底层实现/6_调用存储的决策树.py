import pickle

'''
就是将刚刚的存储的决策树
读取出来
grab = 读取
'''
def grabTree(filename):
    fr = open(filename,'rb')
    return pickle.load(fr)

if __name__== '__main__':
    myTree = grabTree('lenses.txt')
    print(myTree)