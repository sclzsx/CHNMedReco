import torch
from models.adnet import adnet
from models.brdnet import brdnet
from models.classifier import classifier
from models.autoencoder import autoencoder, autoencoder1, autoencoder2, autoencoder3, autoencoder4
from models.dncnn import dncnn
from models.LaneNet0508 import LaneNet0508
from models.resnet import resnet34, resnet34_5
from models.mcae import mcae
from models.tinysxnet import tinysxnet, tinysxnet1, tinysxnet2, tinysxnet3, tinysxnet4
from models.unet import unet
from models.unetnoskip import unetnoskip
from models.unettiny import unettiny
from models.unettinynoskip import unettinynoskip
from models.vgg import VGG
from models.unettinywithmlp import unettinywithmlp


def choose_model(model_name, num_classes, dim):
    # if model_name == 'adnet':
    #     return adnet(num_classes=num_classes, dim=dim)
    # elif model_name == 'brdnet':
    #     return brdnet(num_classes=num_classes, dim=dim)
    # elif model_name == 'classifier':
    #     return classifier(num_classes=num_classes, dim=dim)
    # elif model_name == 'autoencoder':
    #     return autoencoder(num_classes=num_classes, dim=dim)
    # elif model_name == 'autoencoder1':
    #     return autoencoder1(num_classes=num_classes, dim=dim)
    # elif model_name == 'autoencoder2':
    #     return autoencoder2(num_classes=num_classes, dim=dim)
    # elif model_name == 'autoencoder3':
    #     return autoencoder3(num_classes=num_classes, dim=dim)
    # elif model_name == 'autoencoder4':
    #     return autoencoder4(num_classes=num_classes, dim=dim)
    # elif model_name == 'dncnn':
    #     return dncnn(num_classes=num_classes, dim=dim)
    # elif model_name == 'lanenet':
    #     assert num_classes == 1
    #     return LaneNet0508(num_classes=1, divisor=1)
    # elif model_name == 'resnet34':
    #     if num_classes == 4:
    #         return resnet34()
    #     elif num_classes == 5:
    #         return resnet34_5()
    #     else:
    #         return None
    # elif model_name == 'vgg16':
    #     return VGG('VGG16', num_classes=num_classes)
    # elif model_name == 'mcae':
    #     return mcae(num_classes=num_classes, dim=dim)
    # elif model_name == 'tinysxnet':
    #     return tinysxnet(num_classes=num_classes, dim=dim)
    # elif model_name == 'tinysxnet1':
    #     return tinysxnet1(num_classes=num_classes, dim=dim)
    # elif model_name == 'tinysxnet2':
    #     return tinysxnet2(num_classes=num_classes, dim=dim)
    # elif model_name == 'tinysxnet3':
    #     return tinysxnet3(num_classes=num_classes, dim=dim)
    # elif model_name == 'tinysxnet4':
    #     return tinysxnet4(num_classes=num_classes, dim=dim)
    # elif model_name == 'unet':
    #     return unet(num_classes=num_classes, dim=dim)
    # elif model_name == 'unetnoskip':
    #     return unetnoskip(num_classes=num_classes, dim=dim)
    # elif model_name == 'unettiny':
    #     return unettiny(num_classes=num_classes, dim=dim)
    # elif model_name == 'unettinynoskip':
    #     return unettinynoskip(num_classes=num_classes, dim=dim)
    # elif model_name == 'unettinywithmlp':
    #     return unettinywithmlp(num_classes=num_classes, dim=dim)
    # else:
    return resnet34()


def choose_loss(loss_name, reduction='mean'):
    if loss_name == 'mse':
        return torch.nn.MSELoss(reduction=reduction)
    elif loss_name == 'l1':
        return torch.nn.L1Loss(reduction=reduction)
    elif loss_name == 'sl1':
        return torch.nn.SmoothL1Loss(reduction=reduction)
    else:
        return None
