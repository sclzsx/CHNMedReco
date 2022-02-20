import tensorflow as tf

print(tf.test.is_gpu_available())

################################################

# from tensorflow.keras.python.models import Sequential

# from tensorflow.keras.layers import Dense

import numpy

# fix random seed for reproducibility

numpy.random.seed(7)

# load pima indians dataset

# dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
#
# # split into input (X) and output (Y) variables
#
# X = dataset[:,0:8]
#
# Y = dataset[:,8]

from main import get_data

X, X_test, Y, y_test = get_data()

# create model

model = tf.keras.models.Sequential()

# model.add(tf.keras.layers.Dense(32, input_dim=27, activation='relu'))
# model.add(tf.keras.layers.Dense(16,  activation='relu'))
# model.add(tf.keras.layers.Dense(8,  activation='relu'))
# model.add(tf.keras.layers.Dense(16, activation='relu'))
# model.add(tf.keras.layers.Dense(32, activation='relu'))
# model.add(tf.keras.layers.Dense(27,  activation='relu'))
# model.add(tf.keras.layers.Dense(2, activation='sigmoid'))

model.add(tf.keras.layers.Dense(40, input_dim=27, activation='relu'))
# model.add(tf.keras.layers.Dense(2,  activation='relu'))
# model.add(tf.keras.layers.Dense(8,  activation='relu'))
# model.add(tf.keras.layers.Dense(16, activation='relu'))
# model.add(tf.keras.layers.Dense(32, activation='relu'))
# model.add(tf.keras.layers.Dense(27,  activation='relu'))
model.add(tf.keras.layers.Dense(2, activation='sigmoid'))

# Compile model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model

# model.fit(X, Y, epochs=150, batch_size=10)
#
# # evaluate the model
#
# scores = model.evaluate(X, Y)
#
# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#
# 要生成可视化文件，您需要遵循以下命令结构：
#

from ann_visualizer.visualize import ann_viz

ann_viz(model, view=True, filename='network.gv', title='MLP (27, 40, 2)')
#
# model - 您的Keras顺序模型
#
# view - 如果设置为true，则会在命令执行后打开图形预览
#
# filename - 保存图表的位置。（它以'.gv'文件格式保存）
#
# title - 可视ANN的标题
#
# 你刚刚看到你如何轻松地在Keras中创建你的第一个神经网络模型。
#
# 让我们将它与ann_viz（）一起放入此代码中。
#
# from ann_visualizer.visualize import ann_viz;
#
# ann_viz(model, title="My first neural network")
#
# 使用以下命令运行index.py：
#
# python3 index.py
#
# 以下是最终可视化内容的示例：
