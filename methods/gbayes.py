from get_data import get_data
X_train, X_test, y_train, y_test = get_data()

from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

# # 实例化分类器
clf = GaussianNB()

# 训练分类器
clf.fit(
    X_train,
    y_train
)
pred_test = clf.predict(X_test)

# 打印结果
print("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
      .format(
          X_test.shape[0],
          (y_test != pred_test).sum(),
          100*(1-(y_test != pred_test).sum()/X_test.shape[0])
))

# # 实例化分类器
clf = MultinomialNB()

# 训练分类器
clf.fit(
    X_train,
    y_train
)
pred_test = clf.predict(X_test)

# 打印结果
print("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
      .format(
          X_test.shape[0],
          (y_test != pred_test).sum(),
          100*(1-(y_test != pred_test).sum()/X_test.shape[0])
))

# # 实例化分类器
clf = BernoulliNB()

# 训练分类器
clf.fit(
    X_train,
    y_train
)
pred_test = clf.predict(X_test)

# 打印结果
print("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
      .format(
          X_test.shape[0],
          (y_test != pred_test).sum(),
          100*(1-(y_test != pred_test).sum()/X_test.shape[0])
))