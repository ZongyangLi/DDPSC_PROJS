from torchvision import models
from torch import nn

def resnet_50_embedding(out_feature_dim=32, reset_last_n_layers=0):
    model = models.resnet50(pretrained=True)
    model.avgpool =  nn.AdaptiveAvgPool2d((1, 1))
    fc_in_dim = model.fc.in_features
    model.fc = nn.Linear(fc_in_dim, out_feature_dim)
    if reset_last_n_layers != 0:
        model_layers = get_resnet_layers(model)
        for l in model_layers[-reset_last_n_layers:]:
            if hasattr(l, 'reset_parameters'):
                l.reset_parameters()
    return model

def resnet_101_embedding(out_feature_dim=32, reset_last_n_layers=0):
    model = models.resnet101(pretrained=True)
    model.avgpool =  nn.AdaptiveAvgPool2d((1, 1))
    fc_in_dim = model.fc.in_features
    model.fc = nn.Linear(fc_in_dim, out_feature_dim)
    if reset_last_n_layers != 0:
        model_layers = get_resnet_layers(model)
        for l in model_layers[-reset_last_n_layers:]:
            if hasattr(l, 'reset_parameters'):
                l.reset_parameters()
    return model

def get_resnet_layers(model):
    return [module for module in model.modules() if type(module) != nn.Sequential and type(module) != models.resnet.Bottleneck]