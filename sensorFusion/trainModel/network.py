from torchvision import models
from torch import nn
import copy

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

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class joint_resnet(nn.Module):
    def __init__(self, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(joint_resnet, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        block = Bottleneck
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1_l = self.resnet.conv1
        self.bn1_l = self.resnet.bn1
        self.relu_l = self.resnet.relu
        self.maxpool_l = self.resnet.maxpool
        
        self.conv1_r = copy.deepcopy(self.resnet.conv1)
        self.bn1_r = copy.deepcopy(self.resnet.bn1)
        self.relu_r = copy.deepcopy(self.resnet.relu)
        self.maxpool_r = copy.deepcopy(self.resnet.maxpool)
        
        self.layer1_l = self.resnet.layer1
        self.layer1_r = copy.deepcopy(self.resnet.layer1)
        self.layer2_l = self.resnet.layer2
        self.layer2_r = copy.deepcopy(self.resnet.layer2)
        self.layer3_l = self.resnet.layer3
        self.layer3_r = copy.deepcopy(self.resnet.layer3)
        
        self.layer4 = self._make_layer(block, 1024, 4, stride=2,
                                       dilate=replace_stride_with_dilation[2])
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024 * block.expansion, num_classes)
        
        
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        
        x1, x2 = x
        
        x1 = self.conv1_l(x1)
        x1 = self.bn1_l(x1)
        x1 = self.relu_l(x1)
        x1 = self.maxpool_l(x1)

        x1 = self.layer1_l(x1)
        x1 = self.layer2_l(x1)
        x1 = self.layer3_l(x1)
        
        x2 = self.conv1_r(x2)
        x2 = self.bn1_r(x2)
        x2 = self.relu_r(x2)
        x2 = self.maxpool_r(x2)

        x2 = self.layer1_r(x2)
        x2 = self.layer2_r(x2)
        x2 = self.layer3_r(x2)
        
        x = torch.cat((x1, x2), dim=1)
        
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)