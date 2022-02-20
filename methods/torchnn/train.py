import os
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from argparse import ArgumentParser

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from dataset import MyDataset
from choices import choose_model, choose_loss

from sklearn.metrics import f1_score

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


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--save_dir", type=str, default='./results')

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epoch", type=int, default=200)

    parser.add_argument("--classifier", type=str, default='autoencoder3')

    return parser.parse_args()


def get_metrics(name, y_test, pred_test):
    acc = accuracy_score(y_test, pred_test)
    p = precision_score(y_test, pred_test, average='binary')
    r = recall_score(y_test, pred_test, average='binary')
    f1 = f1_score(y_test, pred_test, average='binary')
    metrics = {'accuracy': acc, 'precision': p, 'recall': r, 'f1_score': f1}
    print('[', name, ']\t', metrics)
    return metrics


def classifier_train(args):
    X_train = np.array(pd.read_csv('X_train.csv'))
    y_train = np.array(pd.read_csv('y_train.csv'))

    X_test = np.array(pd.read_csv('X_test.csv'))
    y_test = np.array(pd.read_csv('y_test.csv'))

    save_tag = args.save_dir + '/' + args.classifier + '_' + str(args.batch_size) + '_' + str(
        args.max_epoch) + '_' + str(args.lr)
    print('training classifier:', save_tag)

    X_trainset = MyDataset(data=X_train, labels=y_train)
    X_testset = MyDataset(data=X_test, labels=y_test)

    X_trainloader = DataLoader(X_trainset, batch_size=args.batch_size, shuffle=True)
    X_testloader = DataLoader(X_testset, batch_size=len(y_test), shuffle=False)

    model = choose_model(args.classifier, num_classes=2, dim=27).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss().to(device)

    trainlog = []
    max_test_f1 = -1
    max_iter_num = 1000
    for epoch in range(1, args.max_epoch + 1):

        if max_iter_num < 0:
            break

        epoch_loss_train = 0
        for batch_data, batch_labels in X_trainloader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            batch_pred = model(batch_data)
            batch_loss = criterion(batch_pred, batch_labels)
            iter_loss = batch_loss.item()
            batch_loss.backward()
            optimizer.step()
            epoch_loss_train = epoch_loss_train + iter_loss
            max_iter_num = max_iter_num - 1
        epoch_loss_train = epoch_loss_train / len(X_trainloader)

        epoch_loss_test = 0
        epoch_pred_test = []
        with torch.no_grad():
            for batch_data, batch_labels in X_testloader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)

                batch_pred = model(batch_data)
                epoch_pred_test.append(batch_pred.cpu().numpy().argmax(axis=1))

                batch_loss = criterion(batch_pred, batch_labels)
                iter_loss = batch_loss.item()
                epoch_loss_test = epoch_loss_test + iter_loss
            epoch_loss_test = epoch_loss_test / len(X_testloader)
            epoch_pred_test = np.array(epoch_pred_test).T
            epoch_f1_test = f1_score(y_test, epoch_pred_test, average='binary')

            trainlog.append({'epoch': epoch, 'train_loss': epoch_loss_train, 'test_loss': epoch_loss_test,
                             'test_f1': epoch_f1_test})
            print('epoch:{} train_loss:{} test_loss:{} epoch_f1_test:{}'.format(epoch, round(epoch_loss_train, 5),
                                                                                round(epoch_loss_test, 5),
                                                                                round(epoch_f1_test, 5)))

            if max_test_f1 < epoch_f1_test:
                max_test_f1 = epoch_f1_test
                # torch.save(model.state_dict(),
                #            save_tag + '_loss' + str(epoch_loss_train) + '_f1' + str(epoch_f1_test) + '_best.pth')

                best_metrics = get_metrics(save_tag, y_test, epoch_pred_test)
                with open(save_tag + '.json', 'w') as f:
                    json.dump(best_metrics, f, indent=2)

    epoch, loss1, loss2, f1 = [], [], [], []
    for info in trainlog:
        epoch.append(info['epoch'])
        loss1.append(info['train_loss'])
        loss2.append(info['test_loss'])
        f1.append(info['test_f1'])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(epoch, loss1, '-r', label='train loss')
    ax.plot(epoch, loss2, '-g', label='test loss')
    ax.plot(epoch, f1, '-b', label='test F1')
    ax.legend(loc=0)
    ax.grid()
    ax.set_xlabel("Epoch")
    plt.savefig(save_tag + '.png')
    plt.cla()


def calculate_metrics(y_test, preds, average, all=False):
    if all:
        acc = accuracy_score(y_test, preds)
        p = precision_score(y_test, preds, average=average)
        r = recall_score(y_test, preds, average=average)
        f1 = f1_score(y_test, preds, average=average)
        conf = confusion_matrix(y_test, preds)
        metrics = {'accuracy': acc, 'precision': p, 'recall': r, 'f1_score': f1, 'confusion_matric': conf}
        for item in metrics.items():
            print(item)
        print()
        return metrics
    else:
        f1 = f1_score(y_test, preds, average=average)
        return f1


if __name__ == '__main__':

    args = get_args()  # 获取参数

    if not os.path.exists(args.save_dir):  # 新建保存路径
        os.makedirs(args.save_dir)

    for item in vars(args).items():  # 参数转为字典,并打印
        print(item)
    print()  # 换行

    classifier_train(args)  # 训练分类
