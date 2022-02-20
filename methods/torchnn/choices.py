import torch

from models.autoencoder import autoencoder, autoencoder1, autoencoder2, autoencoder3, autoencoder4

from models.resnet import resnet34, resnet34_5


def choose_model(model_name, num_classes, dim):

    if model_name == 'autoencoder':
        return autoencoder(num_classes=num_classes, dim=dim)
    elif model_name == 'autoencoder1':
        return autoencoder1(num_classes=num_classes, dim=dim)
    elif model_name == 'autoencoder2':
        return autoencoder2(num_classes=num_classes, dim=dim)
    elif model_name == 'autoencoder3':
        return autoencoder3(num_classes=num_classes, dim=dim)
    elif model_name == 'autoencoder4':
        return autoencoder4(num_classes=num_classes, dim=dim)
    else:
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
