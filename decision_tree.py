import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.metrics import plot_roc_curve
import matplotlib.pyplot as plt
from IPython.display import Image
import graphviz
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, roc_auc_score, auc, plot_roc_curve
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn import datasets
from sklearn.model_selection import train_test_split
import json

from main import get_data, get_metrics

X_train, X_test, y_train, y_test = get_data()

# tree.DecisionTreeClassifier(class_weight=None,  # balanced & None 可选
#                             criterion='gini',  # "gini"或者"entropy"，前者代表基尼系数，后者代表信息增益。
#                             max_depth=None,  # max_depth控制树的深度防止overfitting
#                             max_features=None,  # 可使用多种类型值，默认是"None",划分时考虑所有的特征数；
#                             # "log2" 划分时最多考虑log2Nlog2N个特征；
#                             # "sqrt"或者"auto" 划分时最多考虑√N个特征。
#                             # 整数，代表考虑的特征绝对数。
#                             # 浮点数，代表考虑特征百分比，即考虑（百分比xN）取整后的特征数。
#                             # 其中N为样本总特征数。
#                             max_leaf_nodes=None,  # 最大叶节点树
#                             min_impurity_split=1e-07,  # 限制决策树的增长，
#                             # 如果某节点的不纯度(基尼系数，信息增益)小于这个阈值，则该节点不再生成子节点，即为叶子节点。
#                             min_samples_leaf=1, min_samples_split=2,  # min_samples_split或min_samples_leaf来控制叶节点上的样本数量；
#                             # 两者之间的主要区别在于min_samples_leaf保证了叶片中最小的样本数量，而min_samples_split可以创建任意的小叶子。但min_samples_split在文献中更常见。
#                             min_weight_fraction_leaf=0.0,  # 限制叶子节点所有样本权重和的最小值。如果小于这个值，则会和兄弟节点一起被剪枝。
#                             # 默认是0，就是不考虑权重问题。
#                             # 一般来说，如果我们有较多样本有缺失值，或者分类树样本的分布类别偏差很大，
#                             # 就会引入样本权重，这时我们就要注意这个值了。
#                             presort=False,  # 布尔值，默认是False不排序。预排序，提高效率。
#                             # 设置为true可以让划分点选择更加快，决策树建立的更加快。
#                             random_state=None,  # 随机生成器种子设置，默认设置为None，如此，则每次模型结果都会有所不同。
#                             splitter='best')  # split"best"或者"random"。
#                             # 前者在特征的所有划分点中找出最优的划分点。
#                             # 后者是随机的在部分划分点中找局部最优的划分点。
#                             # 默认的"best"适合样本量不大的时候，而如果样本数据量非常大，此时决策树构建推荐"random"。

# Class0: 胆热犯胃证  Class1: 非胆热犯胃证
# Feature0~26: 性别0女	GerdQ 评分	反酸	烧心	呃逆	嗳气	口苦	胃胀满	心烦易怒	胁肋胀满	胸骨后灼痛	口干	咽干	食少纳呆	寐差	神疲乏力	大便溏薄	大便秘结	舌淡红	舌红	舌齿痕	舌苔薄黄	舌苔黄腻	脉弦	脉滑	脉数	脉缓

##################### gini

clf = tree.DecisionTreeClassifier(criterion="gini")

clf = clf.fit(X_train, y_train)

feature_names = ['Feature' + str(i) for i in range(27)]
class_names = ['Class' + str(i) for i in range(2)]

dot_data = tree.export_graphviz(clf,
                                out_file=None,
                                feature_names=feature_names,
                                class_names=class_names,
                                filled=True,
                                rounded=True)
graph = graphviz.Source(dot_data)
graph.render("decision_tree_[gini_default]_tree")

feature_importance = clf.feature_importances_
print(feature_importance)
with open('decision_tree_[gini_default]_feature_importance.txt', 'w') as f:
    for i in feature_importance:
        f.write(str(i))
        f.write('\n')

p_test = clf.predict(X_test)
metrics = get_metrics('decision_tree_[gini_default]_metrics', y_test, p_test)
with open('decision_tree_[gini_default]_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

fig, ax = plt.subplots(figsize=(6, 5))
roc = plot_roc_curve(estimator=clf, X=X_test, y=y_test, ax=ax, linewidth=1)
ax.legend(fontsize=12)
plt.savefig('decision_tree_[gini_default]_roc.png')
plt.cla()


########################## entropy

clf1 = tree.DecisionTreeClassifier(criterion="entropy")

clf1 = clf1.fit(X_train, y_train)

feature_names = ['Feature' + str(i) for i in range(27)]
class_names = ['Class' + str(i) for i in range(2)]

dot_data = tree.export_graphviz(clf1,
                                out_file=None,
                                feature_names=feature_names,
                                class_names=class_names,
                                filled=True,
                                rounded=True)
graph = graphviz.Source(dot_data)
graph.render("decision_tree_[entropy_default]_tree")

feature_importance = clf1.feature_importances_
print(feature_importance)
with open('decision_tree_[entropy_default]_feature_importance.txt', 'w') as f:
    for i in feature_importance:
        f.write(str(i))
        f.write('\n')

p_test = clf1.predict(X_test)
metrics = get_metrics('decision_tree_[entropy_default]_metrics', y_test, p_test)
with open('decision_tree_[entropy_default]_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

fig, ax = plt.subplots(figsize=(6, 5))
roc1 = plot_roc_curve(estimator=clf1, X=X_test, y=y_test, ax=ax, linewidth=1)
ax.legend(fontsize=12)
plt.savefig('decision_tree_[entropy_default]_roc.png')
plt.cla()

fig, ax = plt.subplots(figsize=(6, 5))
roc = plot_roc_curve(estimator=clf, X=X_test, y=y_test, ax=ax, linewidth=1)
roc1 = plot_roc_curve(estimator=clf1, X=X_test, y=y_test, ax=ax, linewidth=1)
ax.legend(fontsize=12)
plt.savefig('decision_tree_[entropy_and_gini_default]_roc.png')
plt.cla()




