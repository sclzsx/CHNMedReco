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

# Class0: 胆热犯胃证  Class1: 非胆热犯胃证
# Feature0~26: 性别0女	GerdQ 评分	反酸	烧心	呃逆	嗳气	口苦	胃胀满	心烦易怒	胁肋胀满	胸骨后灼痛	口干	咽干	食少纳呆	寐差	神疲乏力	大便溏薄	大便秘结	舌淡红	舌红	舌齿痕	舌苔薄黄	舌苔黄腻	脉弦	脉滑	脉数	脉缓

#####################
clf1 = svm.SVC(kernel='rbf')
clf1.fit(X_train, y_train)
p_test = clf1.predict(X_test)

metrics = get_metrics('svm_[rbf]_metrics', y_test, p_test)
with open('svm_[rbf]_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)



#####################
clf2 = svm.SVC(kernel='linear')
clf2.fit(X_train, y_train)
p_test = clf2.predict(X_test)

metrics = get_metrics('svm_[linear]_metrics', y_test, p_test)
with open('svm_[linear]_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)



#####################
clf = svm.SVC(kernel='poly')
clf.fit(X_train, y_train)
p_test = clf.predict(X_test)

metrics = get_metrics('svm_[poly]_metrics', y_test, p_test)
with open('svm_[poly]_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

fig, ax = plt.subplots(figsize=(6, 5))
roc = plot_roc_curve(estimator=clf, X=X_test, y=y_test, ax=ax, linewidth=1)
# ax.legend(fontsize=12)
# plt.savefig('svm_[poly]_roc.png')
# plt.cla()

# fig, ax = plt.subplots(figsize=(6, 5))
roc1 = plot_roc_curve(estimator=clf1, X=X_test, y=y_test, ax=ax, linewidth=1)
# ax.legend(fontsize=12)
# plt.savefig('svm_[linear]_roc.png')
# plt.cla()

# fig, ax = plt.subplots(figsize=(6, 5))
roc2 = plot_roc_curve(estimator=clf2, X=X_test, y=y_test, ax=ax, linewidth=1)
ax.legend(fontsize=12)
plt.savefig('svm_[rbf_poly_linear]_roc.png')
plt.cla()


##################### rbf 1
# l  C：C-SVC的惩罚参数C?默认值是1.0
# C越大，相当于惩罚松弛变量，希望松弛变量接近0，即对误分类的惩罚增大，趋向于对训练集全分对的情况，这样对训练集测试时准确率很高，但泛化能力弱。C值小，对误分类的惩罚减小，允许容错，将他们当成噪声点，泛化能力较强。
# l  kernel ：核函数，默认是rbf，可以是‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
#   　　0 – 线性：u'v
#  　　 1 – 多项式：(gamma*u'*v + coef0)^degree
#   　　2 – RBF函数：exp(-gamma|u-v|^2)
#   　　3 –sigmoid：tanh(gamma*u'*v + coef0)
# l  degree ：多项式poly函数的维度，默认是3，选择其他核函数时会被忽略。
# l  gamma ： ‘rbf’,‘poly’ 和‘sigmoid’的核函数参数。默认是’auto’，则会选择1/n_features
# l  coef0 ：核函数的常数项。对于‘poly’和 ‘sigmoid’有用。
# l  probability ：是否采用概率估计？.默认为False
# l  shrinking ：是否采用shrinking heuristic方法，默认为true
# l  tol ：停止训练的误差值大小，默认为1e-3
# l  cache_size ：核函数cache缓存大小，默认为200
# l  class_weight ：类别的权重，字典形式传递。设置第几类的参数C为weight*C(C-SVC中的C)
# l  verbose ：允许冗余输出？
# l  max_iter ：最大迭代次数。-1为无限制。
# l  decision_function_shape ：‘ovo’, ‘ovr’ or None, default=None3
# l  random_state ：数据洗牌时的种子值，int值
# 主要调节的参数有：C、kernel、degree、gamma、coef0。

clf1 = svm.SVC(kernel='rbf', C=0.5)
clf1.fit(X_train, y_train)
p_test = clf1.predict(X_test)

metrics = get_metrics('svm_[rbf_c05]_metrics', y_test, p_test)
with open('svm_[rbf_c05]_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)



# from sklearn.calibration import calibration_curve
# y_prob=clf1.predict_proba(X_test,probability=True)
# print(y_prob.shape)
# # # 分成10箱，ytest代表是真实标签，y_prob标示返回的概率
# # trueproba, predproba = calibration_curve(y_test, y_prob, n_bins=10)
# # # 紧接着我们就可以画图了
# # fig = plt.figure()
# # ax1 = plt.subplot()
# # ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")  # 得做一条对角线来对比呀
# # ax1.plot(predproba, trueproba, "s-", label="%s (%1.3f)" % ("Bayes", 10))  # 10代表是分10个箱子
# # ax1.set_ylabel("True label")
# # ax1.set_xlabel("predcited probability")
# # ax1.set_ylim([-0.05, 1.05])
# # ax1.legend()
# # plt.show()