import pickle

'''
storeTree  这是用来我们存储我们自己写的决策树
inputTree 就是我们所生成的决策树
filename 就是我们存储树文件的名字

'''
def storeTree(inputTree, filename):
    # 'w' =open for writing, truncating the file first 打开写入，先截断文件，
    # 'b'=binary mode 既是以二进制模式存存储
    with open(filename,'wb') as fw:
        # dump 就是转存的意思
        pickle.dump(inputTree, fw)

if __name__ =='__main__':
    myTree = {'job': {0: 'no', 1: {'credit': {0: 'no', 1: {'house': {0: 'no', 1: 'yes', 2: 'yes'}}, 2: 'yes'}}}}
    # storage 就是仓库的意思
    storeTree(myTree, 'lenses.txt')