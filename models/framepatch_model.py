import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from utils.utils import initialize_weights
from models.mil import AMIL, MIL, cnn_extractor, attention_net
class Frame_PAMIL(nn.Module):
    def __init__(self, tile_size) -> None:
        print('Creating Attention-MIL module')

        super(Frame_PAMIL, self).__init__()
        if tile_size == 64:
            self.cnn = cnn_extractor(tile_size = 64)
        elif tile_size == 128:
            self.cnn = cnn_extractor(tile_size = 128)
        elif tile_size == 32:
            self.cnn = cnn_extractor(tile_size = 32)
        elif tile_size == 256:
            self.cnn = cnn_extractor(tile_size = 256)
            
        self.fc1 = nn.Sequential(
            nn.Linear(2048,512), 
            nn.ReLU(), 
            nn.Dropout(0.25))        
        self.attention = attention_net()
        self.fc2 = nn.Sequential(
            nn.Linear(512,2))
        initialize_weights(self)

    def forward(self, input_data, frame_attention: torch.Tensor, return_attention: bool = False, return_features: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inp = input_data['patch_features']
        patch_lens = input_data['patch_lens']
        #option 1 Find the top 25% of the frames
        frame_quantile_ind = frame_attention >= torch.quantile(frame_attention, 0.75)#frame_attention.median()
        ind_repeat = [frame_quantile_ind[i].repeat(patch_lens[i]) for i in range(len(frame_quantile_ind))]
        ind_repeat = torch.cat(ind_repeat)
        inp = inp[ind_repeat]
        
        features = self.cnn(inp)
        x = self.fc1(features)
    

        #option 2
        #features = self.cnn(inp)
        #x = self.fc1(features)
        #x = x * frame_attention
        
        A = self.attention(x)
        A_softmax = torch.reshape(F.softmax(A, dim=0),(1,-1))
        x = torch.mm(A_softmax,x)
        if return_features:
            return x
        logits = self.fc2(x)
        y_prob = F.softmax(logits, dim = 1)
        if return_attention:
            return A
        return logits,y_prob, A


class FramePatchParallel(nn.Module):
    def __init__(self, tile_size, use_clinical: bool = False, include_tirads: bool = False) -> None:
        print('Creating Attention-MIL module (Frame and Patch)')
        super(FramePatchParallel, self).__init__()
        self.frame_model = AMIL()
        self.patch_model = Frame_PAMIL(tile_size = tile_size )

        self.use_clinical = use_clinical

        if use_clinical:
            if include_tirads:
                self.classifier = nn.Sequential(
                    nn.Linear(1024+3,512),
                    nn.SELU(),
                    nn.Linear(512,2))
            else:
                self.classifier = nn.Sequential(
                    nn.Linear(1024+2,512),
                    nn.SELU(),
                    nn.Linear(512,2))
        else:
          self.classifier = nn.Sequential(
              nn.Linear(1024,512),
              nn.SELU(),
              nn.Linear(512,2))

        initialize_weights(self)

    def forward(self, input_data):
        clinical = input_data['clinical']

        frame_x , frame_A= self.frame_model(input_data, return_features = True, return_attention = True)
        frame_A = frame_A.squeeze()

        patch_x = self.patch_model(input_data, frame_A,return_features = True)

        if self.use_clinical:
          x = torch.cat([frame_x,patch_x, clinical.float()], dim = 1)
        else:
          x = torch.cat([frame_x,patch_x], dim = 1)
        
        logits = self.classifier(x)
        y_prob = F.softmax(logits, dim = 1)
        return logits,y_prob, None
        