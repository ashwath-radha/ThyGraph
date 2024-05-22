import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple

# import sys
# sys.path.append('/raid/aradhachandran/thyroid/code/classification/BE223A_thyroidProj')
from utils.utils import initialize_weights
# from models.mil import *
import os
import timm

class wang_attention_net(nn.Module):
    def __init__(self) -> None:
        super(wang_attention_net, self).__init__()
        self.attn1 = nn.Sequential(
            nn.Conv2d(in_channels=2080, out_channels=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        A = self.attn1(features)
        return A

class WangModel(nn.Module):
    def __init__(self) -> None:
        print('Creating WangModel module')
        super(WangModel, self).__init__()
        
        self.feature_extractor = timm.create_model('inception_resnet_v2', pretrained=True)
        self.module1to4 = nn.Sequential(
            *list(self.feature_extractor.children())[:-4]
        ) # cut everything before conv2d_7b
        
        self.attention = wang_attention_net()
        
        self.module5 = nn.Sequential(
            *list(self.feature_extractor.children())[-4:-3],  # Adjust the number of layers to exclude as needed
            nn.MaxPool2d(kernel_size=8, stride=1, padding=0, dilation=1, ceil_mode=False),
            nn.Dropout(p=0.2, inplace=False),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(1536, 2)
        )

        # self.amil = AMIL()
        
        initialize_weights(self) # this function doesn't work on this model

    def forward(self, bag_imgs, mode, return_attention: bool = False, return_features: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # print('model input: ', bag_imgs.shape)
        batch_size, squiggly_n = bag_imgs.size()[:2]
        bag_imgs = bag_imgs.view(-1, *bag_imgs.size()[2:])#.unsqueeze(1)#.repeat(1,3,1,1)

        # # print('forward input shape: ', bag_imgs.shape)
        # input_data = {'frame_features': bag_imgs, 'clinical': None}
        # logits, y_prob, A = self.amil(input_data, batch_size, squiggly_n)

        # print('shape: ', bag_imgs.shape)
        x = self.module1to4(bag_imgs)
        # print('moduel1to4: ', x.shape)
        A = self.attention(x)
        # print('A: ', A.shape)
        A = A.view(batch_size, 1, squiggly_n) # A.reshape(batch_size, 1, squiggly_n)
        # print('reshape A: ', A.shape)
        A_softmax = F.softmax(A, dim=2)
        # print('A_softmax: ', A_softmax.shape)
        x = x.view(batch_size, squiggly_n, -1) # x.reshape(batch_size, squiggly_n, -1)
        # print('reshape x: ', x.shape)
        agg = torch.matmul(A_softmax, x) # torch.bmm(A_softmax, x)
        # print('att bmm feat: ', agg.shape)#, torch.bmm(A_softmax, x).shape)
        agg = agg.view(batch_size, 2080, 8, 8) # agg.reshape(batch_size, 2080, 8, 8)
        logits = self.module5(agg)
        # print('logits: ', logits.shape, logits)
        if return_features:
            return x
        if return_attention:
            return A
        
        y_prob = F.softmax(logits, dim = 1)
        # print('y_prob: ', logits, y_prob)

        return logits, y_prob, A

class attention_net(nn.Module):
    def __init__(self) -> None:
        super(attention_net, self).__init__()
        self.attn1 =  nn.Sequential(
            nn.Linear(512,256), 
            nn.Tanh(),
            nn.Dropout(0.25))
        self.attn2 =  nn.Sequential(
            nn.Linear(512,256), 
            nn.Sigmoid(),
            nn.Dropout(0.25))
        self.fc =  nn.Sequential(
            nn.Linear(256,1), 
            nn.ReLU())

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        attn1 = self.attn1(features)
        attn2 = self.attn2(features)
        A = self.fc(torch.mul(attn1,attn2))
        return A

class AMIL(nn.Module):
    def __init__(self, use_clinical: bool = False, include_tirads: bool = False ) -> None:
        print('Creating Attention-MIL module')
        self.use_clinical = use_clinical
        super(AMIL, self).__init__()
        self.cnn =  nn.Sequential(
                nn.Conv2d(1,2,padding=1,kernel_size=3,stride=1), 
                nn.Dropout2d(p = 0.25),
                nn.MaxPool2d((2,2)),
                nn.Conv2d(2,4,padding=1,kernel_size=3,stride=1), 
                nn.Dropout2d(p = 0.25),
                nn.MaxPool2d((2,2)),
                nn.Conv2d(4,8,padding=1,kernel_size=3,stride=1), 
                nn.Dropout2d(p = 0.25),
                nn.MaxPool2d((2,2)),
                nn.Conv2d(8,16,padding=1,kernel_size=3,stride=1), 
                nn.Dropout2d(p = 0.25),
                nn.MaxPool2d((2,2)),
                nn.Conv2d(16,32,padding=1,kernel_size=3,stride=1), 
                nn.Dropout2d(p = 0.25),
                nn.MaxPool2d((2,2)),
                nn.SELU(inplace=True))
        
        self.fc1 = nn.Sequential(
            nn.Linear(2592,1024), 
            nn.ReLU(), 
            nn.Linear(1024,512), 
            nn.ReLU(), 
            nn.Dropout(0.25))
        self.attention = attention_net()

        if use_clinical:
            if include_tirads:
                self.fc2 = nn.Sequential(
                    nn.Linear(512 + 3, 2))
            else:
                self.fc2 = nn.Sequential(
                    nn.Linear(512 + 2, 2))
        else:
          self.fc2 = nn.Sequential(
              nn.Linear(512,2))
        initialize_weights(self)

    def forward(self, input_data, batch_size, squiggly_n, return_attention: bool = False, return_features: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = input_data['frame_features'][:, 0].unsqueeze(1)
        clinical = input_data['clinical']
        # print(features.shape)
        features = self.cnn(features)
        features = features.view(features.shape[0],-1)
        # print(features.shape)
        x = self.fc1(features)
        A = self.attention(x)
        
        # print(A.shape)
        A = A.view(batch_size, 1, squiggly_n) # A.reshape(batch_size, 1, squiggly_n)
        # print('reshape A: ', A.shape)
        A_softmax = F.softmax(A, dim=2)
        # print('A_softmax: ', A_softmax.shape)
        x = x.view(batch_size, squiggly_n, -1) # x.reshape(batch_size, squiggly_n, -1)
        # print('reshape x: ', x.shape)
        x = torch.matmul(A_softmax, x).squeeze()
        if len(x.size())==1:
            x = x.unsqueeze(0) # torch.bmm(A_softmax, x)

        if return_features:
          if return_attention:
            return x, A
          else:
            return x
        logits = self.fc2(x)
        y_prob = F.softmax(logits, dim = 1)

        if return_attention:
            return A
        return logits,y_prob,A