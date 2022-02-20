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
import json

def get_metrics(name, y_test, pred_test):
    acc = accuracy_score(y_test, pred_test)
    p = precision_score(y_test, pred_test, average='binary')
    r = recall_score(y_test, pred_test, average='binary')
    f1 = f1_score(y_test, pred_test, average='binary')
    metrics = {'accuracy': acc, 'precision': p, 'recall': r, 'f1_score': f1}
    print('[', name, ']\t', metrics)
    return metrics

def get_data():
    df = pd.read_excel('data.xlsx')
    data = df.values

    data = data[:, :-1]
    labels = data[:, 0]
    data = data[:, 1:]
    labels = labels - 1

    data = np.array(data).astype('float32')
    labels = np.array(labels).astype('int32')

    min_ = np.min(data)
    max_ = np.max(data)

    data = (data - min_) / (max_ - min_)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.9, random_state=21, stratify=labels)

    return X_train, X_test, y_train, y_test

def all_run():
    X_train, X_test, y_train, y_test = get_data()

    results = []

    dt_clf = tree.DecisionTreeClassifier()
    dt_clf = dt_clf.fit(X_train, y_train)
    dt_pred_test = dt_clf.predict(X_test)
    metrics = get_metrics('DecisionTree', y_test, dt_pred_test)
    results.append(metrics)

    gnb_clf = GaussianNB()
    gnb_clf.fit(X_train, y_train)
    gnb_pred_test = gnb_clf.predict(X_test)
    metrics = get_metrics('GaussianNB', y_test, gnb_pred_test)
    results.append(metrics)

    mnb_clf = MultinomialNB()
    mnb_clf.fit(X_train, y_train)
    mnb_pred_test = mnb_clf.predict(X_test)
    metrics = get_metrics('MultinomialNB', y_test, mnb_pred_test)
    results.append(metrics)

    bnb_clf = BernoulliNB()
    bnb_clf.fit(X_train, y_train)
    bnb_pred_test = bnb_clf.predict(X_test)
    metrics = get_metrics('BernoulliNB', y_test, bnb_pred_test)
    results.append(metrics)

    rf_clf = RandomForestClassifier()
    rf_clf.fit(X_train, y_train)
    rf_pred_test = rf_clf.predict(X_test)
    metrics = get_metrics('RandomForest', y_test, rf_pred_test)
    results.append(metrics)

    mlp_clf = MLPClassifier()
    mlp_clf.fit(X_train, y_train)
    mlp_pred_test = mlp_clf.predict(X_test)
    metrics = get_metrics('MLP', y_test, mlp_pred_test)
    results.append(metrics)

    lsvm_clf = svm.SVC(kernel='linear')
    lsvm_clf.fit(X_train, y_train)
    lsvm_pred_test = lsvm_clf.predict(X_test)
    metrics = get_metrics('SVM', y_test, lsvm_pred_test)
    results.append(metrics)

    gb_clf = GradientBoostingClassifier()
    gb_clf.fit(X_train, y_train)
    gb_pred_test = gb_clf.predict(X_test)
    metrics = get_metrics('GradientBoosting', y_test, gb_pred_test)
    results.append(metrics)

    fig, ax = plt.subplots(figsize=(6, 5))
    dt_roc = plot_roc_curve(estimator=dt_clf, X=X_test,
                            y=y_test, ax=ax, linewidth=1)
    gnb_roc = plot_roc_curve(estimator=gnb_clf, X=X_test,
                             y=y_test, ax=ax, linewidth=1)
    mnb_roc = plot_roc_curve(estimator=mnb_clf, X=X_test,
                             y=y_test, ax=ax, linewidth=1)
    bnb_roc = plot_roc_curve(estimator=bnb_clf, X=X_test,
                             y=y_test, ax=ax, linewidth=1)
    rf_roc = plot_roc_curve(estimator=rf_clf, X=X_test,
                            y=y_test, ax=ax, linewidth=1)
    mlp_roc = plot_roc_curve(estimator=mlp_clf, X=X_test,
                             y=y_test, ax=ax, linewidth=1)
    lsvm_roc = plot_roc_curve(estimator=lsvm_clf, X=X_test,
                              y=y_test, ax=ax, linewidth=1)
    gb_roc = plot_roc_curve(estimator=gb_clf, X=X_test,
                            y=y_test, ax=ax, linewidth=1)
    ax.legend(fontsize=12)
    plt.savefig('all_roc_0.9.png')

    with open('all_metrics_0.9.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    all_run()
