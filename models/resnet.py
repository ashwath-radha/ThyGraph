# defines the models for feature extraction

import torch
import torch.nn as nn
from torch import Tensor
import h5py
import numpy as np
import torch.utils.model_zoo as model_zoo
import torchvision.models as models

from functools import partial
from typing import Any, Callable, List, Optional, Type, Union, Tuple

################
# Below is from pytorch with some modifications. Needed to add bias to convolutional layers
################


def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=True,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
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

    def forward(self, x: Tensor) -> Tensor:
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


class ResNetCustom(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()

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
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=True
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
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
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


# wrapper function for creating the ResNet50 model that is compatible with tensoflow weights
def resnet50custom() -> torch.nn.Module:
    # Create resnet50 model similar to pytorch's but with slight modifications to bias terms
    model = ResNetCustom(Bottleneck, [3, 4, 6, 3])

    # modify resnet50 for feature extraction
    model.fc = nn.Identity()
    return model


################
# the below implements functionality for loading a pytorch resnet model pretarined on RadImageNet

# convert_layer_param and get_layer can be combined but separated for simplicity
################


def convert_layer_param(k: str) -> str:
    # convert paramaters from pytorch to tensorflow for a layer

    # convolution layer parameters. keys are pytorch parameter names, values are the corresponding tensorflow names
    conv_conversion = {"weight": "kernel:0", "bias": "bias:0"}

    # batchnorm layer parameters. keys are pytorch parameter names, values are the corresponding tensorflow names
    bn_conversion = {
        "weight": "gamma:0",
        "bias": "beta:0",
        "running_mean": "moving_mean:0",
        "running_var": "moving_variance:0",
    }

    # look for 'conv' in the layer name
    if "conv" in k:
        conversion_dict = conv_conversion
    # look for 'bn' in the layer name
    elif "bn" in k:
        conversion_dict = bn_conversion
    # look for 'downsample' in the layer name
    elif "downsample" in k:
        # for downsample layers, convolutions are indicated with a '0'
        if k.split(".")[3] == "0":
            conversion_dict = conv_conversion
        # for downsample layers, batchnorms are indicated with a '1'
        elif k.split(".")[3] == "1":
            conversion_dict = bn_conversion

    # the last part of the layer name is the parameter name
    parameter = k.split(".")[-1]

    # convert the parameter name from pytorch to tensorflow
    return conversion_dict[parameter]


def get_layer(k: str) -> Tuple[str, str]:
    # convert the layer type and number from pytorch to tensorflow

    # if 'conv' is in the layer name, then the type is convolution
    # the layer number is immediately after the conv and before the next period
    if "conv" in k:
        layer_num = k.split("conv")[-1].split(".")[0]
        layer_type = "conv"
    # if 'bn' is in the layer name, then the type is batchnorm
    # the layer number is immediately after the conv and before the next period
    elif "bn" in k:
        layer_num = k.split("bn")[-1].split(".")[0]
        layer_type = "bn"
    # if 'downsample' is in the layer name, then the convolutions are indicated with a '0', batchnorms are indicated with a '1'
    # the layer number is always 0
    elif "downsample" in k:
        layer_num = "0"
        if k.split(".")[3] == "0":
            layer_type = "conv"
        elif k.split(".")[3] == "1":
            layer_type = "bn"

    return layer_type, layer_num


def weight_translate(k: str, w: torch.Tensor) -> torch.Tensor:
    # reshape parameters from tensorflow to pytorch
    # tensorflow weights are store: channel_in, channel_out, height, width
    # pytorch needs: width, height, channel_in, channel out

    if k.endswith(".weight"):
        if w.dim() == 2:
            w = w.t()
        elif w.dim() == 1:
            pass
        else:
            assert w.dim() == 4
            w = w.permute(3, 2, 0, 1)
    return w


class ResNetRadImageNet:
    # resnet pretrained on RadImageNet

    def __init__(self):
        print("Creating ResNet module")

        # pretrained RadImageNet ResNet50 weights
        weights_file = "./iodata/feature_extractor/RadImageNet-ResNet50_notop.h5"

        # load pretrained weights
        tf_weights = h5py.File(weights_file, mode="r")["model_weights"]

        # model needs bias parameters
        model = resnet50custom()
        pytorch_state = model.state_dict()

        # iterate through all parameters
        for i, (k, v) in enumerate(pytorch_state.items()):
            # skip these (no learnable params)
            if "num_batches" in k:
                continue

            # create a string that maps the pytorch key to tensorflow key
            tflow_key = ""
            if "layer" not in k:
                # first layer does not have the word 'layer' in it
                tflow_key = tflow_key + "conv1_"

                # the parameter name is before the '1.'
                tflow_key = tflow_key + k.split("1.")[0]

            else:
                # resnets are built with the idea of having parameter layers (ie convolution) within abstracted resnet layers (each resnet layer contains mutiple blocks with multiple convolutions)
                # the resnet layer is at the beginning fo the string, and maps to 'conv' in tensorflow
                resnet_layer_num = int(k.split(".")[0].split("layer")[-1])
                tflow_key = tflow_key + "conv" + str(resnet_layer_num + 1)

                # the block number is the second part of the string
                block_num = int(k.split(".")[1])
                tflow_key = tflow_key + "_block" + str(block_num + 1)

                # the convolutional/batch layer in the block is in the final part of the string
                layer_type, layer_num = get_layer(k)
                tflow_key = tflow_key + "_" + layer_num + "_" + layer_type

            # tflow_key now defines the exact layer we want in tensorflow
            # each layer can have multiple parameters (weight, bias)
            # get the parameters from the tensorflow dictionary
            param = tf_weights[tflow_key][tflow_key][convert_layer_param(k)][:]

            # convert from numpy to tensorflow and transform
            param = torch.from_numpy(param)
            param = weight_translate(k, param)

            assert v.shape == param.shape

            # add to the pytorch model state
            pytorch_state[k] = param

        # load the pretrained weights into the model
        model.load_state_dict(pytorch_state)

        self.pretrained_model = model

    def __call__(self, nodules: torch.Tensor) -> np.ndarray:
        features = self.pretrained_model(nodules)
        features = features.cpu().numpy()
        return features

    def to(self, device):
        self.pretrained_model.to(device)


################
# pretrained resnet using imagenet weights
# customized for obtaining the third layer
################


class Bottleneck_Baseline(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_Baseline, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet_Baseline(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet_Baseline, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x


# class ResNetPretrained:
#     # resnet pretrained on ImageNet

#     def __init__(self):
#         print("Creating ResNet module")

#         """
#         model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
#         model.fc = nn.Identity()
#         self.pretrained_model = model #torch.nn.Sequential(*list(model.children())[:-1])
#         """

#         model = ResNet_Baseline(Bottleneck_Baseline, [3, 4, 6, 3])
#         pretrained_dict = model_zoo.load_url(
#             "https://download.pytorch.org/models/resnet50-19c8e357.pth"
#         )
#         model.load_state_dict(pretrained_dict, strict=False)
#         self.pretrained_model = model

#     def __call__(self, nodules: torch.Tensor) -> np.ndarray:
#         features = self.pretrained_model(nodules)
#         features = features.cpu().numpy()
#         return features

#     def to(self, device):
#         self.pretrained_model.to(device)

#     def eval(self):
#         self.pretrained_model.eval()

class ResNetPretrained:
    def __init__(self):
        print("Creating PretrainedModels module")

        # Load pretrained ResNet50
        resnet = models.resnet101(pretrained=True)
        # resnet = models.resnet50(pretrained=True)
        resnet.fc = nn.Identity()

        # Load pretrained DenseNet121
        densenet = models.densenet201(pretrained=True)
        # densenet = models.densenet121(pretrained=True)
        densenet.classifier = nn.Identity()

        # Load pretrained ResNeXt50_32x4d
        resnext = models.resnext101_32x8d(pretrained=True)
        # resnext = models.resnext50_32x4d(pretrained=True)
        resnext.fc = nn.Identity()

        self.models = [resnet, densenet, resnext]

    def __call__(self, nodules: torch.Tensor) -> np.ndarray:
        # device_count = torch.cuda.device_count()
        # if device_count > 1:
        #     parallel_models = [nn.DataParallel(model) for model in self.models]
        #     model_outputs = [model(nodules) for model in parallel_models]
        # else:
        #     model_outputs = [model(nodules) for model in self.models]
        # concatenated_features = torch.cat(model_outputs, dim=1)
        # concatenated_features = concatenated_features.cpu().numpy()
        # print('concatenated: ', concatenated_features.shape)
        # return concatenated_features

        features = []
        for model in self.models:
            model_features = model(nodules)
            model_features = model_features.cpu().numpy()
            features.append(model_features)
            # print('feature - ', model_features.shape)
        concatenated_features = np.concatenate(features, axis=1)
        # print('concatenated: ', concatenated_features.shape)
        return concatenated_features

    def to(self, device):
        for model in self.models:
            model.to(device)

    def eval(self):
        for model in self.models:
            model.eval()