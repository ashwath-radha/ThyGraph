import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple
from utils.utils import initialize_weights
import os

class MIL(nn.Module):
    def __init__(self) -> None:
        print('Creating MIL module')
        
        super(MIL, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(2048,512), 
            nn.ReLU(), 
            nn.Dropout(0.5))
        self.fc2 = nn.Sequential(
            nn.Linear(512,2))

        initialize_weights(self)

    def forward(self, features: torch.Tensor, return_attention: bool = False, return_logits: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        x = self.fc1(features)
        logits = self.fc2(x)
        probs = F.softmax(logits,dim = 1) 
        top_instance_idx = torch.topk(probs[:, 1], 1, dim=0)[1].view(1,)
        top_instance = torch.index_select(logits, dim=0, index=top_instance_idx)
        y_prob = F.softmax(top_instance, dim = 1) 
        if return_attention:
            return F.softmax(probs[:,1], dim=0)
        return top_instance,y_prob, F.softmax(probs[:,1], dim=0)


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
        # if not self.training:
        #     print(features)
        attn1 = self.attn1(features)
        # if not self.training:
        #     print(attn1)
        attn2 = self.attn2(features)
        # if not self.training:
        #     print(attn2)
        A = self.fc(torch.mul(attn1,attn2))
        # if not self.training:
        #     print('mul attn: ', torch.mul(attn1,attn2))
        #     print('mul attn shape: ', torch.mul(attn1,attn2).shape)
        #     print('final Aunique: ', A.flatten().unique())
        #     print('final A: ', A)
        return A

class AMIL(nn.Module):
    def __init__(self, use_clinical: bool = False, include_tirads: bool = False ) -> None:
        print('Creating Attention-MIL module')
        self.use_clinical = use_clinical
        super(AMIL, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(6016,1024), 
            # nn.Linear(2048,1024), 
            nn.ELU(), # nn.ReLU(), 
            nn.Linear(1024,512), 
            nn.ELU(), # nn.ReLU(), 
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

    def forward(self, input_data, return_attention: bool = False, return_features: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = input_data['frame_features']
        clinical = input_data['clinical']

        x = self.fc1(features)
        A = self.attention(x)
        A_softmax = torch.reshape(F.softmax(A, dim=0),(1,-1))
        #print(A_softmax)
        x = torch.mm(A_softmax,x)

        if self.use_clinical:
          x = torch.cat([x, clinical.float()], dim = 1)

        if return_features:
          if return_attention:
            return x, A
          else:
            return x
        logits = self.fc2(x)
        #y_pred = torch.topk(logits, 1, dim = 1)[1]
        y_prob = F.softmax(logits, dim = 1)
        #import pdb;pdb.set_trace()
        if return_attention:
            return A
        return logits,y_prob, A


class cnn_extractor(nn.Module):
    def __init__(self, tile_size) -> None:
        super(cnn_extractor, self).__init__()

        # TODO: make this flexible (as well as the MIL module) depending on the tile_size, number of cnn layers,
        # and output dimension

        # 64x64, 2 layers

        if tile_size == 64:
            self.cnn =  nn.Sequential(
                nn.Conv2d(1,8,padding=1,kernel_size=3,stride=1), 
                nn.MaxPool2d((2,2)),
                nn.Conv2d(8,16,padding=1,kernel_size=3,stride=1), 
                nn.MaxPool2d((2,2)),
                nn.Conv2d(16,32,padding=1,kernel_size=3,stride=1), 
                nn.MaxPool2d((2,2)),
                nn.SELU(inplace=True)
                )
        elif tile_size ==128:
            self.cnn =  nn.Sequential(
                nn.Conv2d(1,4,padding=1,kernel_size=3,stride=1),
                # nn.ReLU(), 
                nn.MaxPool2d((2,2)),
                nn.Conv2d(4,8,padding=1,kernel_size=3,stride=1), 
                # nn.ReLU(), 
                nn.MaxPool2d((2,2)),
                nn.Conv2d(8,16,padding=1,kernel_size=3,stride=1), 
                # nn.ReLU(), 
                nn.MaxPool2d((2,2)),
                nn.Conv2d(16,32,padding=1,kernel_size=3,stride=1), 
                # nn.ELU(), 
                nn.MaxPool2d((2,2)),
                # nn.ELU())
                nn.SELU(inplace=True))
            # resnet = models.resnet34(pretrained=True)
            # resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # # resnet = nn.Sequential(*list(resnet.children())[:-1])
            # resnet.fc = nn.Linear(512,2048, bias=True)
            # for m in resnet.modules():
            #     if isinstance(m, nn.Conv2d):
            #         m.bias = nn.Parameter(torch.zeros(m.out_channels))  # Initialize bias to zeros
            # self.cnn = resnet
        elif tile_size == 32:
            self.cnn =  nn.Sequential(
                nn.Conv2d(1,32,padding=1,kernel_size=3,stride=1), 
                nn.MaxPool2d((2,2)),
                nn.Conv2d(32,64,padding=1,kernel_size=3,stride=1), 
                nn.MaxPool2d((2,2)),
                nn.Conv2d(64,128,padding=1,kernel_size=3,stride=1), 
                nn.MaxPool2d((2,2)),
                nn.SELU(inplace=True)
                )
        elif tile_size == 256:
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


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(x.shape)

        feat = self.cnn(x.float()) 
        # print(feat.shape)
        feat = feat.view(feat.shape[0],-1)
        # print(feat.shape)
        return feat

class PMIL(nn.Module):
    def __init__(self) -> None:
        print('Creating MIL module')
        
        super(PMIL, self).__init__()
        self.cnn = cnn_extractor()
        self.fc1 = nn.Sequential(
            nn.Linear(2048,512), 
            nn.ReLU(), 
            nn.Dropout(0.5))
        self.fc2 = nn.Sequential(
            nn.Linear(512,2))

        initialize_weights(self)

    def forward(self, input_data, return_attention: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inp = input_data['patch_features']
        features = self.cnn(inp)
        x = self.fc1(features)
        logits = self.fc2(x)
        probs = F.softmax(logits,dim = 1) 
        top_instance_idx = torch.topk(probs[:, 1], 1, dim=0)[1].view(1,)
        top_instance = torch.index_select(logits, dim=0, index=top_instance_idx)
        y_prob = F.softmax(top_instance, dim = 1) 
        if return_attention:
            return F.softmax(probs[:,1], dim=0)
        return top_instance,y_prob, F.softmax(probs[:,1], dim=0)

class PAMIL(nn.Module):
    def __init__(self, tile_size, use_clinical: bool = False, include_tirads: bool = False) -> None:
        print('Creating Attention-MIL module')

        super(PAMIL, self).__init__()
        if tile_size == 64:
            self.cnn = cnn_extractor(tile_size = 64)
        elif tile_size == 128:
            self.cnn = cnn_extractor(tile_size = 128)
        elif tile_size == 32:
            self.cnn = cnn_extractor(tile_size = 32)
        elif tile_size == 256:
            self.cnn = cnn_extractor(tile_size = 256)
        self.use_clinical = use_clinical

        self.fc1 = nn.Sequential(
            nn.Linear(2048,512), 
            nn.ELU(),
            # nn.ReLU(),
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

    def forward(self, input_data, return_attention: bool = False, return_features: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inp = input_data['patch_features']
        clinical = input_data['clinical']

        features = self.cnn(inp)
        x = self.fc1(features)
        A = self.attention(x)
        # print(f'model attention scores ({A.shape}) for {features.shape}: ', A.flatten().unique()[:5])
        A_softmax = torch.reshape(F.softmax(A, dim=0),(1,-1))
        # print(f'softmax attention: ', A_softmax.flatten().unique()[:5])
        x = torch.mm(A_softmax,x)

        if self.use_clinical:
          x = torch.cat([x, clinical.float()], dim = 1)
        
        if return_features:
            return x
        
        logits = self.fc2(x)
        y_prob = F.softmax(logits, dim = 1)
        if return_attention:
            return A
        return logits,y_prob, A

