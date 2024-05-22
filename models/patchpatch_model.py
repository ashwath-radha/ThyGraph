
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from utils.utils import initialize_weights
from models.mil import PAMIL


class PatchPatchParallel(nn.Module):
    def __init__(self, tile_size, use_clinical: bool = False, include_tirads: bool = False) -> None:
        print('Creating Attention-MIL module (Patch and Patch)')
        super(PatchPatchParallel, self).__init__()
        self.patch_model1 = PAMIL(tile_size = tile_size[0] , use_clinical = False)
        self.patch_model2 = PAMIL(tile_size = tile_size[1] , use_clinical = False )

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

    def forward(self, input_data, return_attention = False):
        clinical = input_data['clinical']
        input_data_1 = {'patch_features': input_data['patch_features'][0], 'clinical':input_data['clinical']}
        input_data_2 = {'patch_features': input_data['patch_features'][1], 'clinical':input_data['clinical']}

        if return_attention:
          A1 = self.patch_model1(input_data_1, return_attention = True)
          A2 = self.patch_model2(input_data_2, return_attention = True)
          return [A1, A2]

        x1 = self.patch_model1(input_data_1, return_features = True)
        x2 = self.patch_model2(input_data_2, return_features = True)

        if self.use_clinical:
          x = torch.cat([x1,x2, clinical.float()], dim = 1)
        else:
          x = torch.cat([x1,x2], dim = 1)
        
        logits = self.classifier(x)
        y_prob = F.softmax(logits, dim = 1)
        return logits,y_prob, None
        