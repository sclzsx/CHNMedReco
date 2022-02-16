from get_data import get_data
X_train, X_test, y_train, y_test = get_data()
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print(X_train.dtype, X_test.dtype, y_train.dtype, y_test.dtype)

from itertools import product

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier


# from sklearn.datasets import load_iris
# from sklearn import tree
# X, y = load_iris(return_X_y=True)
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(X, y)
# print(X.shape, y.shape)
# print('sss')
# print(X_train.shape, y_train.shape)

# https://blog.csdn.net/qq_29003925/article/details/75222560

from sklearn.metrics import classification_report
import pandas as pd
from sklearn import tree

# #训练集数据读取
# train = pd.read_csv('train.csv')
# target='TRADER' # TRADER的值就是二元分类的输出（列名）
# ID = 'USER_ID'
# train['TRADER'].value_counts() #类别计算

# x_columns0 = [x for x in train.columns if x not in [target, ID]]
# X = train[x_columns0]
# y = train['TRADER']

# #测试集数据读取
# test = pd.read_csv('test.csv')
# test['TRADER'].value_counts() #类别计算
# x_columns1 = [x for x in test.columns if x not in [target, ID]]
# x_test = test[x_columns1]
# y_test = test['TRADER']



print ('数据读取完毕')

#引入tree模块，对应的参数设置将于后面提及
# clf = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
#             max_features=7, max_leaf_nodes=None,
#             min_impurity_split=0.005, min_samples_leaf=3,
#             min_samples_split=2, min_weight_fraction_leaf=0.0,
#             presort=False, random_state=1, splitter='random')

clf = tree.DecisionTreeClassifier()

clf = clf.fit(X_train,y_train) #此时完成训练

pred_test=clf.predict(X_test) #预测

#混淆矩阵模块
print(confusion_matrix(y_test,pred_test))#验证集上的混淆矩阵

train_pred=clf.predict(X_train)
print(confusion_matrix(y_train,train_pred))#训练集上的混淆矩阵

#准确率及召回率等Report模块
print(classification_report(y_test,pred_test))

# tree.DecisionTreeClassifier(class_weight=None, #balanced & None 可选
#                             criterion='gini',#"gini"或者"entropy"，前者代表基尼系数，后者代表信息增益。
#                             max_depth=None,#max_depth控制树的深度防止overfitting
#             max_features=None, #可使用多种类型值，默认是"None",划分时考虑所有的特征数；
#                                #"log2" 划分时最多考虑log2Nlog2N个特征；
#                                #"sqrt"或者"auto" 划分时最多考虑√N个特征。
#                                #整数，代表考虑的特征绝对数。
#                                #浮点数，代表考虑特征百分比，即考虑（百分比xN）取整后的特征数。
#                                #其中N为样本总特征数。
#             max_leaf_nodes=None,#最大叶节点树
#             min_impurity_split=1e-07, #限制决策树的增长，
#                             #如果某节点的不纯度(基尼系数，信息增益)小于这个阈值，则该节点不再生成子节点，即为叶子节点。 
#             min_samples_leaf=1,min_samples_split=2,#min_samples_split或min_samples_leaf来控制叶节点上的样本数量；
#             #两者之间的主要区别在于min_samples_leaf保证了叶片中最小的样本数量，而min_samples_split可以创建任意的小叶子。但min_samples_split在文献中更常见。
#             min_weight_fraction_leaf=0.0,#限制叶子节点所有样本权重和的最小值。如果小于这个值，则会和兄弟节点一起被剪枝。
#                             # 默认是0，就是不考虑权重问题。
#                             #一般来说，如果我们有较多样本有缺失值，或者分类树样本的分布类别偏差很大，
#                             #就会引入样本权重，这时我们就要注意这个值了。
#             presort=False,#布尔值，默认是False不排序。预排序，提高效率。
#                           #设置为true可以让划分点选择更加快，决策树建立的更加快。
#             random_state=None, #随机生成器种子设置，默认设置为None，如此，则每次模型结果都会有所不同。
#             splitter='best')#split"best"或者"random"。
#            #前者在特征的所有划分点中找出最优的划分点。
#            #后者是随机的在部分划分点中找局部最优的划分点。
#            #默认的"best"适合样本量不大的时候，而如果样本数据量非常大，此时决策树构建推荐"random"。