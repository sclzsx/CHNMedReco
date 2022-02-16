 
from get_data import get_data
X_train, X_test, y_train, y_test = get_data()

import pandas as pd
import sklearn
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
 
 
# 使用决策树
dtc = DecisionTreeClassifier()
 
dtc.fit(X_train, y_train)
 
dt_predict = dtc.predict(X_test)
 
print(dtc.score(X_test, y_test))
 
print(classification_report(y_test, dt_predict, target_names=["died", "survived"]))
 
# 使用随机森林
 
rfc = RandomForestClassifier()
 
rfc.fit(X_train, y_train)
 
rfc_pred_testict = rfc.predict(X_test)
 
print(rfc.score(X_test, y_test))
 
print(classification_report(y_test, rfc_pred_testict, target_names=["died", "survived"]))