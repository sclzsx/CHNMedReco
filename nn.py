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

# hidden_layer_sizes	tuple，length = n_layers - 2，默认值（100，）第i个元素表示第i个隐藏层中的神经元数量。
# 激活	{‘identity’，‘logistic’，‘tanh’，‘relu’}，默认’relu’ 隐藏层的激活函数：‘identity’，无操作激活，对实现线性瓶颈很有用，返回f（x）= x；‘logistic’，logistic sigmoid函数，返回f（x）= 1 /（1 + exp（-x））；‘tanh’，双曲tan函数，返回f（x）= tanh（x）；‘relu’，整流后的线性单位函数，返回f（x）= max（0，x）
# slover	{‘lbfgs’，‘sgd’，‘adam’}，默认’adam’。权重优化的求解器：'lbfgs’是准牛顿方法族的优化器；'sgd’指的是随机梯度下降。'adam’是指由Kingma，Diederik和Jimmy Ba提出的基于随机梯度的优化器。注意：默认解算器“adam”在相对较大的数据集（包含数千个训练样本或更多）方面在训练时间和验证分数方面都能很好地工作。但是，对于小型数据集，“lbfgs”可以更快地收敛并且表现更好。
# alpha	float，可选，默认为0.0001。L2惩罚（正则化项）参数。
# batch_size	int，optional，默认’auto’。用于随机优化器的minibatch的大小。如果slover是’lbfgs’，则分类器将不使用minibatch。设置为“auto”时，batch_size = min（200，n_samples）
# learning_rate	{‘常数’，‘invscaling’，‘自适应’}，默认’常数"。 用于权重更新。仅在solver ='sgd’时使用。'constant’是’learning_rate_init’给出的恒定学习率；'invscaling’使用’power_t’的逆缩放指数在每个时间步’t’逐渐降低学习速率learning_rate_， effective_learning_rate = learning_rate_init / pow（t，power_t）；只要训练损失不断减少，“adaptive”将学习速率保持为“learning_rate_init”。每当两个连续的时期未能将训练损失减少至少tol，或者如果’early_stopping’开启则未能将验证分数增加至少tol，则将当前学习速率除以5。
# learning_rate_init	double，可选，默认为0.001。使用初始学习率。它控制更新权重的步长。仅在solver ='sgd’或’adam’时使用。
# power_t	double，可选，默认为0.5。反缩放学习率的指数。当learning_rate设置为“invscaling”时，它用于更新有效学习率。仅在solver ='sgd’时使用。
# max_iter	int，optional，默认值200。最大迭代次数。solver迭代直到收敛（由’tol’确定）或这个迭代次数。对于随机解算器（‘sgd’，‘adam’），请注意，这决定了时期的数量（每个数据点的使用次数），而不是梯度步数。
# shuffle	bool，可选，默认为True。仅在solver ='sgd’或’adam’时使用。是否在每次迭代中对样本进行洗牌。
# random_state	int，RandomState实例或None，可选，默认无随机数生成器的状态或种子。如果是int，则random_state是随机数生成器使用的种子;如果是RandomState实例，则random_state是随机数生成器;如果为None，则随机数生成器是np.random使用的RandomState实例。
# tol	float，optional，默认1e-4 优化的容忍度，容差优化。当n_iter_no_change连续迭代的损失或分数没有提高至少tol时，除非将learning_rate设置为’adaptive’，否则认为会达到收敛并且训练停止。
# verbose	bool，可选，默认为False 是否将进度消息打印到stdout。
# warm_start	bool，可选，默认为False，设置为True时，重用上一次调用的解决方案以适合初始化，否则，只需擦除以前的解决方案。请参阅词汇表。
# momentum	float，默认0.9，梯度下降更新的动量。应该在0和1之间。仅在solver ='sgd’时使用。
# nesterovs_momentum	布尔值，默认为True。是否使用Nesterov的势头。仅在solver ='sgd’和momentum> 0时使用。
# early_stopping	bool，默认为False。当验证评分没有改善时，是否使用提前停止来终止培训。如果设置为true，它将自动留出10％的训练数据作为验证，并在验证得分没有改善至少为n_iter_no_change连续时期的tol时终止训练。仅在solver ='sgd’或’adam’时有效
# validation_fraction	float，optional，默认值为0.1。将训练数据的比例留作早期停止的验证集。必须介于0和1之间。仅在early_stopping为True时使用
# beta_1	float，optional，默认值为0.9，估计一阶矩向量的指数衰减率应为[0,1)。仅在solver ='adam’时使用
# beta_2	float，可选，默认为0.999,估计一阶矩向量的指数衰减率应为[0,1)。仅在solver ='adam’时使用
# epsilon	float，optional，默认值1e-8, adam稳定性的价值。 仅在solver ='adam’时使用
# n_iter_no_change	int，optional，默认值10,不符合改进的最大历元数。 仅在solver ='sgd’或’adam’时有效

# Class0: 胆热犯胃证  Class1: 非胆热犯胃证
# Feature0~26: 性别0女	GerdQ 评分	反酸	烧心	呃逆	嗳气	口苦	胃胀满	心烦易怒	胁肋胀满	胸骨后灼痛	口干	咽干	食少纳呆	寐差	神疲乏力	大便溏薄	大便秘结	舌淡红	舌红	舌齿痕	舌苔薄黄	舌苔黄腻	脉弦	脉滑	脉数	脉缓

##################### h100

clf = MLPClassifier(hidden_layer_sizes=(100,))

clf = clf.fit(X_train, y_train)

p_test = clf.predict(X_test)
metrics = get_metrics('nn_[h100]_metrics', y_test, p_test)
with open('nn_[h100]_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

fig, ax = plt.subplots(figsize=(6, 5))
roc = plot_roc_curve(estimator=clf, X=X_test, y=y_test, ax=ax, linewidth=1)
ax.legend(fontsize=12)
plt.savefig('nn_[h100]_roc.png')
plt.cla()

##################### h70

clf = MLPClassifier(hidden_layer_sizes=(70,))

clf = clf.fit(X_train, y_train)

p_test = clf.predict(X_test)
metrics = get_metrics('nn_[h70]_metrics', y_test, p_test)
with open('nn_[h70]_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

fig, ax = plt.subplots(figsize=(6, 5))
roc = plot_roc_curve(estimator=clf, X=X_test, y=y_test, ax=ax, linewidth=1)
ax.legend(fontsize=12)
plt.savefig('nn_[h70]_roc.png')
plt.cla()

##################### h40

clf = MLPClassifier(hidden_layer_sizes=(100,))

clf = clf.fit(X_train, y_train)

p_test = clf.predict(X_test)
metrics = get_metrics('nn_[h40]_metrics', y_test, p_test)
with open('nn_[h40]_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

fig, ax = plt.subplots(figsize=(6, 5))
roc = plot_roc_curve(estimator=clf, X=X_test, y=y_test, ax=ax, linewidth=1)
ax.legend(fontsize=12)
plt.savefig('nn_[h40]_roc.png')
plt.cla()

##################### h27

clf = MLPClassifier(hidden_layer_sizes=(27,))

clf = clf.fit(X_train, y_train)

p_test = clf.predict(X_test)
metrics = get_metrics('nn_[h27]_metrics', y_test, p_test)
with open('nn_[h27]_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

fig, ax = plt.subplots(figsize=(6, 5))
roc = plot_roc_curve(estimator=clf, X=X_test, y=y_test, ax=ax, linewidth=1)
ax.legend(fontsize=12)
plt.savefig('nn_[h27]_roc.png')
plt.cla()

##################### h27new

clf = MLPClassifier(hidden_layer_sizes=(27,50,20,))

clf = clf.fit(X_train, y_train)

p_test = clf.predict(X_test)
metrics = get_metrics('nn_[h27new]_metrics', y_test, p_test)
with open('nn_[h27new]_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

fig, ax = plt.subplots(figsize=(6, 5))
roc = plot_roc_curve(estimator=clf, X=X_test, y=y_test, ax=ax, linewidth=1)
ax.legend(fontsize=12)
plt.savefig('nn_[h27new]_roc.png')
plt.cla()

##################### h27_tanh

clf = MLPClassifier(hidden_layer_sizes=(27,), activation='tanh')

clf = clf.fit(X_train, y_train)

p_test = clf.predict(X_test)
metrics = get_metrics('nn_[h27_tanh]_metrics', y_test, p_test)
with open('nn_[h27_tanh]_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

fig, ax = plt.subplots(figsize=(6, 5))
roc = plot_roc_curve(estimator=clf, X=X_test, y=y_test, ax=ax, linewidth=1)
ax.legend(fontsize=12)
plt.savefig('nn_[h27_tanh]_roc.png')
plt.cla()

##################### h27_logistic

clf = MLPClassifier(hidden_layer_sizes=(27,), activation='logistic')

clf = clf.fit(X_train, y_train)

p_test = clf.predict(X_test)
metrics = get_metrics('nn_[h27_logistic]_metrics', y_test, p_test)
with open('nn_[h27_logistic]_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

fig, ax = plt.subplots(figsize=(6, 5))
roc = plot_roc_curve(estimator=clf, X=X_test, y=y_test, ax=ax, linewidth=1)
ax.legend(fontsize=12)
plt.savefig('nn_[h27_logistic]_roc.png')
plt.cla()


##################### h27new2

clf = MLPClassifier(hidden_layer_sizes=(128,64,32,16,8,16,32, 64, 128))

clf = clf.fit(X_train, y_train)

p_test = clf.predict(X_test)
metrics = get_metrics('nn_[h27new2]_metrics', y_test, p_test)
with open('nn_[h27new2]_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

fig, ax = plt.subplots(figsize=(6, 5))
roc = plot_roc_curve(estimator=clf, X=X_test, y=y_test, ax=ax, linewidth=1)
ax.legend(fontsize=12)
plt.savefig('nn_[h27new2]_roc.png')
plt.cla()

