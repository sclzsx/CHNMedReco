from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def exal():
    print(confusion_matrix(y_test, pred_test))#验证集上的混淆矩阵
    print(classification_report(y_test,pred_test))
