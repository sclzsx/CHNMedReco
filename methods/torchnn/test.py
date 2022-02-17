import os
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

from dataset import MyDataset
from choices import choose_model, choose_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")






if __name__ == '__main__':

    args = get_args()  # 获取参数

    if not os.path.exists(args.save_dir):  # 新建保存文件夹
        os.makedirs(args.save_dir)

    for item in vars(args).items():  # 打印参数信息
        print(item)
    print()

    if args.merged_mode:  # 五分类模式
        classifier_test(args)

    else:  # 自编码器加四分类模式
        autoencoder_test(args)  # 预测自编码器
        # classifier_test(args)  # 预测四分类
        # cascaded_test(args)  # 预测级联
