import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


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

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=21, stratify=labels)

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_data()
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    print(y_test)
    print(y_train)
    print(X_train[0, :])
    print(X_test[0, :])

    print(type(X_train[0]))
    print(X_train.dtype)
