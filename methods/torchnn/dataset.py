# import cv2
import random

import numpy as np
import pandas as pd
from collections import Counter
from matplotlib import pyplot as plt
from argparse import ArgumentParser

import torch
from torch.utils.data import Dataset

from scipy.signal import resample
from sklearn.model_selection import train_test_split

from numpy.matlib import repmat


def split_data():
    df = pd.read_excel('../data.xlsx')
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

    pd.DataFrame(X_train).to_csv('X_train.csv', index=False, header=False)
    pd.DataFrame(X_test).to_csv('X_test.csv', index=False, header=False)
    pd.DataFrame(y_train).to_csv('y_train.csv', index=False, header=False)
    pd.DataFrame(y_test).to_csv('y_test.csv', index=False, header=False)
  


class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        # print(labels.shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):

        source = self.data[i]
        source = torch.tensor(source, dtype=torch.float32).unsqueeze(0)  # required shape [N 1 27]

        target = self.labels[i]
        # print('sss', target)
        target = torch.tensor(target, dtype=torch.int64)  # 分类时，为了适应交叉熵函数，需要转成long

        target = target.squeeze()
        # print(source.shape, target.shape)
        return source, target


if __name__ == '__main__':

    split_data()

    X_train = np.array(pd.read_csv('X_train.csv'))
    X_test = np.array(pd.read_csv('X_test.csv'))
    y_train = np.array(pd.read_csv('y_train.csv'))
    y_test = np.array(pd.read_csv('y_test.csv'))

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    # print(y_train.shape)

    X_train = np.squeeze(X_train)
    X_trainset = MyDataset(data=X_train, labels=y_train)
    # print('sssssss', X_train.shape, y_train.shape)
    for i, j in X_trainset:
        print(i.shape, j.shape, j)
        # print(i)
        break
