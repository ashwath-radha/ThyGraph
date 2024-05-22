# defines the models for pretrained segmentation

import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import url_map, url_map_advprop, get_model_params
import torch.utils.model_zoo as model_zoo
from typing import Optional
import numpy as np
from skimage.measure import label as sklabel
from skimage.transform import resize
from PIL import Image
from torchvision import transforms as T
from skimage.measure import regionprops

################ 
# the below is from: https://github.com/WAMAWAMA/TNSCUI2020-Seg-Rank1st
################

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

class ASPPSeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(
            SeparableConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates, separable=False):
        super(ASPP, self).__init__()
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
        )

        rate1, rate2, rate3 = tuple(atrous_rates)
        ASPPConvModule = ASPPConv if not separable else ASPPSeparableConv

        modules.append(ASPPConvModule(in_channels, out_channels, rate1))
        modules.append(ASPPConvModule(in_channels, out_channels, rate2))
        modules.append(ASPPConvModule(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

class SeparableConv2d(nn.Sequential):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
    ):
        dephtwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        pointwise_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias,
        )
        super().__init__(dephtwise_conv, pointwise_conv)

class DeepLabV3PlusDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        out_channels=256,
        atrous_rates=(12, 24, 36),
        output_stride=16,
    ):
        super().__init__()
        if output_stride not in {8, 16}:
            raise ValueError("Output stride should be 8 or 16, got {}.".format(output_stride))

        self.out_channels = out_channels
        self.output_stride = output_stride

        self.aspp = nn.Sequential(
            ASPP(encoder_channels[-1], out_channels, atrous_rates, separable=True),
            SeparableConv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        scale_factor = 2 if output_stride == 8 else 4
        self.up = nn.UpsamplingBilinear2d(scale_factor=scale_factor)

        highres_in_channels = encoder_channels[-4]
        highres_out_channels = 48   # proposed by authors of paper
        self.block1 = nn.Sequential(
            nn.Conv2d(highres_in_channels, highres_out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(highres_out_channels),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            SeparableConv2d(
                highres_out_channels + out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, *features):
        aspp_features = self.aspp(features[-1])
        aspp_features = self.up(aspp_features)
        high_res_features = self.block1(features[-4])
        concat_features = torch.cat([aspp_features, high_res_features], dim=1)
        fused_features = self.block2(concat_features)
        return fused_features

def patch_first_conv(model, in_channels):
    """Change first convolution layer input channels.
    In case:
        in_channels == 1 or in_channels == 2 -> reuse original weights
        in_channels > 3 -> make random kaiming normal initialization
    """

    # get first conv
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            break

    # change input channels for first conv
    module.in_channels = in_channels
    weight = module.weight.detach()
    reset = False

    if in_channels == 1:
        weight = weight.sum(1, keepdim=True)
    elif in_channels == 2:
        weight = weight[:, :2] * (3.0 / 2.0)
    else:
        reset = True
        weight = torch.Tensor(
            module.out_channels,
            module.in_channels // module.groups,
            *module.kernel_size
        )

    module.weight = nn.parameter.Parameter(weight)
    if reset:
        module.reset_parameters()

def replace_strides_with_dilation(module, dilation_rate):
    """Patch Conv2d modules replacing strides with dilation"""
    for mod in module.modules():
        if isinstance(mod, nn.Conv2d):
            mod.stride = (1, 1)
            mod.dilation = (dilation_rate, dilation_rate)
            kh, kw = mod.kernel_size
            mod.padding = ((kh // 2) * dilation_rate, (kh // 2) * dilation_rate)

            # Kostyl for EfficientNet
            if hasattr(mod, "static_padding"):
                mod.static_padding = nn.Identity()

class EncoderMixin:
    """Add encoder functionality such as:
        - output channels specification of feature tensors (produced by encoder)
        - patching first convolution for arbitrary input channels
    """

    @property
    def out_channels(self):
        """Return channels dimensions for each tensor of forward output of encoder"""
        return self._out_channels[: self._depth + 1]

    def set_in_channels(self, in_channels):
        """Change first convolution chennels"""
        if in_channels == 3:
            return

        self._in_channels = in_channels
        if self._out_channels[0] == 3:
            self._out_channels = tuple([in_channels] + list(self._out_channels)[1:])

        patch_first_conv(model=self, in_channels=in_channels)

    def get_stages(self):
        """Method should be overridden in encoder"""
        raise NotImplementedError

    def make_dilated(self, stage_list, dilation_list):
        stages = self.get_stages()
        for stage_indx, dilation_rate in zip(stage_list, dilation_list):
            replace_strides_with_dilation(
                module=stages[stage_indx],
                dilation_rate=dilation_rate,
            )

class EfficientNetEncoder(EfficientNet, EncoderMixin):
    def __init__(self, stage_idxs, out_channels, model_name, depth=5):

        blocks_args, global_params = get_model_params(model_name, override_params=None)
        super().__init__(blocks_args, global_params)

        self._stage_idxs = stage_idxs
        self._out_channels = out_channels
        self._depth = depth
        self._in_channels = 3

        del self._fc

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self._conv_stem, self._bn0, self._swish),
            self._blocks[:self._stage_idxs[0]],
            self._blocks[self._stage_idxs[0]:self._stage_idxs[1]],
            self._blocks[self._stage_idxs[1]:self._stage_idxs[2]],
            self._blocks[self._stage_idxs[2]:],
        ]

    def forward(self, x):
        stages = self.get_stages()

        block_number = 0.
        drop_connect_rate = self._global_params.drop_connect_rate

        features = []
        for i in range(self._depth + 1):

            # Identity and Sequential stages
            if i < 2:
                x = stages[i](x)

            # Block stages need drop_connect rate
            else:
                for module in stages[i]:
                    drop_connect = drop_connect_rate * block_number / len(self._blocks)
                    block_number += 1.
                    x = module(x, drop_connect)

            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("_fc.bias")
        state_dict.pop("_fc.weight")
        super().load_state_dict(state_dict, **kwargs)

def _get_pretrained_settings(encoder):
    pretrained_settings = {
        "imagenet": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "url": url_map[encoder],
            "input_space": "RGB",
            "input_range": [0, 1],
        },
        "advprop": {
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
            "url": url_map_advprop[encoder],
            "input_space": "RGB",
            "input_range": [0, 1],
        }
    }
    return pretrained_settings

def get_encoder(encoder_name, in_channels=3, depth=5, weights=None):
    encoders = {"encoder": EfficientNetEncoder,
        "pretrained_settings": _get_pretrained_settings("efficientnet-b6"),
        "params": {
            "out_channels": (3, 56, 40, 72, 200, 576),
            "stage_idxs": (9, 15, 31, 45),
            "model_name": "efficientnet-b6",}
        }
    
    Encoder = encoders["encoder"]
    params = encoders["params"]
    params.update(depth=depth)
    encoder = Encoder(**params)

    if weights is not None:
        settings = encoders["pretrained_settings"][weights]
        encoder.load_state_dict(model_zoo.load_url(settings["url"]))

    encoder.set_in_channels(in_channels)

    return encoder

class Activation(nn.Module):

    def __init__(self, name, **params):

        super().__init__()

        if name is None or name == 'identity':
            self.activation = nn.Identity(**params)
        elif name == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif name == 'softmax2d':
            self.activation = nn.Softmax(dim=1, **params)
        elif name == 'softmax':
            self.activation = nn.Softmax(**params)
        elif name == 'logsoftmax':
            self.activation = nn.LogSoftmax(**params)
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError('Activation should be callable/sigmoid/softmax/logsoftmax/None; got {}'.format(name))

    def forward(self, x):
        return self.activation(x)

class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = Activation(activation)
        super().__init__(conv2d, upsampling, activation)

class DeepLabV3Plus(nn.Module):
    def __init__(self, encoder_name: str = "efficientnet-b6",
            encoder_depth: int = 5,
            encoder_weights: Optional[str] = "imagenet",
            encoder_output_stride: int = 16,
            decoder_channels: int = 256,
            decoder_atrous_rates: tuple = (12, 24, 36),
            in_channels: int = 3,
            classes: int = 1,
            activation: Optional[str] = None,
            upsampling: int = 4,
            aux_params: Optional[dict] = None,):

        super(DeepLabV3Plus, self).__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        if encoder_output_stride == 8:
            self.encoder.make_dilated(
                stage_list=[4, 5],
                dilation_list=[2, 4]
            )

        elif encoder_output_stride == 16:
            self.encoder.make_dilated(
                stage_list=[5],
                dilation_list=[2]
            )
        else:
            raise ValueError(
                "Encoder output stride should be 8 or 16, got {}".format(encoder_output_stride)
            )

        self.decoder = DeepLabV3PlusDecoder(
            encoder_channels=self.encoder.out_channels,
            out_channels=decoder_channels,
            atrous_rates=decoder_atrous_rates,
            output_stride=encoder_output_stride,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        mask = self.segmentation_head(decoder_output)

        return mask

def largestConnectComponent(bw_img):
    if np.sum(bw_img)==0:
        return bw_img
    labeled_img, num = sklabel(bw_img, connectivity=None, background=0, return_num=True)
    if num == 1:
        return bw_img

    max_label = 0
    max_num = 0
    for i in range(0,num):
        if np.sum(labeled_img == (i+1)) > max_num:
            max_num = np.sum(labeled_img == (i+1))
            max_label = i+1
    mcr = (labeled_img == max_label)
    return mcr.astype(np.int)


def preprocess(mask_c1_array_biggest, c1_size=256):
    if np.sum(mask_c1_array_biggest) == 0:
            minr, minc, maxr, maxc = [0, 0, c1_size, c1_size]
    else:
        region = regionprops(mask_c1_array_biggest)[0]
        minr, minc, maxr, maxc = region.bbox

    dim1_center, dim2_center = [(maxr + minr) // 2, (maxc + minc) // 2]
    max_length = max(maxr - minr, maxc - minc)

    max_lengthl = int((c1_size/256)*80)
    preprocess1 = int((c1_size/256)*19)
    pp22 = int((c1_size/256)*31)


    if max_length > max_lengthl:
        ex_pixel = preprocess1 + max_length // 2
    else:
        ex_pixel = pp22 + max_length // 2

    dim1_cut_min = dim1_center - ex_pixel
    dim1_cut_max = dim1_center + ex_pixel
    dim2_cut_min = dim2_center - ex_pixel
    dim2_cut_max = dim2_center + ex_pixel

    if dim1_cut_min < 0:
        dim1_cut_min = 0
    if dim2_cut_min < 0:
        dim2_cut_min = 0
    if dim1_cut_max > c1_size:
        dim1_cut_max = c1_size
    if dim2_cut_max > c1_size:
        dim2_cut_max = c1_size
    return [dim1_cut_min,dim1_cut_max,dim2_cut_min,dim2_cut_max]

################
# this is the wrapper class that instantiates the model and loads the pretrained weights
################

class SegTNSCUI(nn.Module):
    def __init__(self):
        super(SegTNSCUI, self).__init__()
        
        print('Creating TNSCUI segmentor')

        # hardcoded path to weights
        weight_c1 = './iodata/feature_extractor/fold1_stage1_trained_on_size_256.pkl'
        weight_c2 = './iodata/feature_extractor/fold1_stage2_trained_on_size_512.pkl'
        
        # create first model and load weights
        self.model_cascade1 = DeepLabV3Plus(encoder_name="efficientnet-b6", encoder_weights=None, in_channels=1, classes=1)
        self.model_cascade1.load_state_dict(torch.load(weight_c1))
        self.model_cascade1.eval()
        
        # create second model and load weights
        self.model_cascade2 = DeepLabV3Plus(encoder_name="efficientnet-b6", encoder_weights=None, in_channels=1, classes=1)
        self.model_cascade2.load_state_dict(torch.load(weight_c2))
        self.model_cascade2.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print('Segmenting using model from TNSCUI competition')

        with torch.no_grad():

            # first cascade
            mask_c1 = self.model_cascade1(x)
            mask_c1 = torch.sigmoid(mask_c1)
            mask_c1_array = (torch.squeeze(mask_c1)).data.cpu().numpy()
            mask_c1_array = (mask_c1_array>0.5)
            mask_c1_array = mask_c1_array.astype(np.float32)
            mask_c1_array_biggest = largestConnectComponent(mask_c1_array.astype(np.int))

            # second cascade
            dim1_cut_min, dim1_cut_max, dim2_cut_min, dim2_cut_max = preprocess(mask_c1_array_biggest,256)
            img_array = torch.squeeze(x)
            img_array_roi = img_array[dim1_cut_min:dim1_cut_max,dim2_cut_min:dim2_cut_max]
            img_array_roi_shape = img_array_roi.shape
            img_array_roi = resize(img_array_roi, (512, 512), order=3)
            img_array_roi_tensor = torch.tensor(data = img_array_roi,dtype=x.dtype)
            img_array_roi_tensor = torch.unsqueeze(img_array_roi_tensor,0)
            img_array_roi_tensor = torch.unsqueeze(img_array_roi_tensor,0)
            mask_c2 = self.model_cascade2(img_array_roi_tensor)
            mask_c2 = torch.sigmoid(mask_c2)
            mask_c2_array = (torch.squeeze(mask_c2)).data.cpu().numpy()
            mask_c2_array = mask_c2_array.astype(np.float32)
            mask_c2_array = resize(mask_c2_array, img_array_roi_shape, order=0)
            mask_c1_array_biggest[dim1_cut_min:dim1_cut_max, dim2_cut_min:dim2_cut_max] = mask_c2_array
            mask_c1_array_biggest = mask_c1_array_biggest.astype(np.float32)

        mask_c1_array_biggest = mask_c1_array_biggest.astype(np.float32)
        mask_c1_array_biggest = resize(mask_c1_array_biggest, x.shape[2:], order=1)
        mask_c1_array_biggest = (mask_c1_array_biggest > 0.5)

        return torch.from_numpy(mask_c1_array_biggest).unsqueeze(0).unsqueeze(0)

################ 
# the below is from: https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation
################

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class TRFE(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        print('Creating TRFE-Net segmentor')

        super(TRFE, self).__init__()

        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)

        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64, out_ch, 1)

        self.up6_align = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.up7_align = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up8_align = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up9_align = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.up6t = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6t = DoubleConv(1024, 512)
        self.up7t = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7t = DoubleConv(512, 256)
        self.up8t = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8t = DoubleConv(256, 128)
        self.up9t = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9t = DoubleConv(128, 64)
        self.conv10t = nn.Conv2d(64, out_ch, 1)

        self.reduction6 = nn.Conv2d(1024, 512, 1, padding=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print('Segmenting using TRFE-Net')

        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)

        up_6t = self.up6t(c5)
        merge6t = torch.cat([up_6t, c4], dim=1)
        c6t = self.conv6t(merge6t)

        up_7t = self.up7t(c6t)
        merge7t = torch.cat([up_7t, c3], dim=1)
        c7t = self.conv7t(merge7t)

        up_8t = self.up8t(c7t)
        merge8t = torch.cat([up_8t, c2], dim=1)
        c8t = self.conv8t(merge8t)

        up_9t = self.up9t(c8t)
        merge9t = torch.cat([up_9t, c1], dim=1)
        c9t = self.conv9t(merge9t)

        thyroid = self.conv10t(c9t)

        thyroid_norm = nn.Sigmoid()(thyroid)

        c8_mask = F.interpolate(thyroid_norm, scale_factor=0.5, mode='nearest')
        c7_mask = F.interpolate(c8_mask, scale_factor=0.5, mode='nearest')
        c6_mask = F.interpolate(c7_mask, scale_factor=0.5, mode='nearest')  

        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)

        c6f = c6.mul(c6_mask)
        c6f = c6+c6f

        up_7 = self.up7(c6f)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)

        c7f = c7.mul(c7_mask)
        c7f = c7+c7f

        up_8 = self.up8(c7f)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)

        c8f = c8.mul(c8_mask)
        c8f = c8+c8f

        up_9 = self.up9(c8f)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)

        nodule = self.conv10(c9)

        # out = nn.Sigmoid()(c10)
        return nodule, thyroid

################
# this is the wrapper class that instantiates the model and loads the pretrained weights
################
class GongSeg(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(GongSeg, self).__init__()
        print('Creating TRFE-Net segmentor')

        # hardcoded path to weights
        weights = './iodata/feature_extractor/trfe_best.pth'

        # create model and load weights
        self.model = TRFE(in_ch=in_ch, out_ch=out_ch)
        self.model.load_state_dict(torch.load(weights))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print('Segmenting using TRFE-Net') 
        nodule = self.model(x)[0]
        nodule = torch.sigmoid(nodule)    
        nodule = (nodule>0.5)
        return nodule

################ 
# Below is a wrapper function that creates one of the two pretrained semgentation models
################
def get_segmentor(mode: str='tnsui') -> torch.nn.Module:
    
    if mode == 'tnsui':
        segmentor = SegTNSCUI()
    elif mode == 'gongseg':
        segmentor = GongSeg()
    
    return segmentor
