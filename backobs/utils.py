import torch


def has_batchnorm(model):
    batchnorm_cls = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
    )
    for module in model.children():
        if isinstance(module, batchnorm_cls):
            return True
    return False


def has_dropout(model):
    dropout_cls = (torch.nn.Dropout,)
    for module in model.children():
        if isinstance(module, dropout_cls):
            return True
    return False
