

import torch
import torchvision
import torch.nn as nn
import numpy as np
import math

from scripts.utils.model_init import *
from scripts.models.basic import *

__all__ = ['jcs', 'jcs_lite','sgn','res_sgn']


class JCS_LITE(nn.Module):
    """docstring for J_C_S"""

    def __init__(self,):
        super(JCS_LITE, self).__init__()
        patch_size = 64
        hidden_dim = 256
        number_of_class = 2
        self.encode_lstm = LSTMEncoder(patch_size, hidden_dim)
        self.classification = Classification(hidden_dim, number_of_class)

    def forward(self, x):
        local_feature = self.encode_lstm(x)
        classification = self.classification(local_feature)
        return classification

def jcs_lite(**kwargs):
    model = JCS_LITE()
    model.apply(weights_init_kaiming)
    return model

class JCS(nn.Module):
    """docstring for J_C_S"""
    def __init__(self,):
        super(JCS, self).__init__()
        patch_size = 64
        hidden_dim = 256
        number_of_class = 2
        self.encode_lstm = LSTMEncoder(patch_size, hidden_dim)
        self.classification = Classification(hidden_dim, number_of_class)
        self.Segmentation = Segmentation(patch_size)

    def forward(self,x):
        local_feature = self.encode_lstm(x)
        seg = self.Segmentation(local_feature)
        classification = self.classification(local_feature)

        return classification,seg


def jcs(**kwargs):
    model = JCS()
    model.apply(weights_init_kaiming)
    return model


class SGN(nn.Module):
    def __init__(self):
        super(SGN, self).__init__()
        
        patch_size = 64
        hidden_dim = 256
        global_dim = 256
        number_of_class = 2
        self.encode_lstm = LSTMEncoder(patch_size,hidden_dim)
        self.global_network = GlobalNetwork(global_dim)
        self.classification = SGNClassification(hidden_dim, global_dim,number_of_class)
        self.Segmentation = Segmentation(patch_size)

    def forward(self,x,x_global):
        local_feature = self.encode_lstm(x)
        global_feature = self.global_network(x_global)
        classification = self.classification(local_feature,global_feature)
        seg = self.Segmentation(local_feature)
        return classification,seg
        
def sgn(**kwargs):
    model = SGN()
    model.apply(weights_init_kaiming)
    return model


class SGNResNet(nn.Module):
    def __init__(self):
        super(SGNResNet, self).__init__()
        
        patch_size = 64
        hidden_dim = 256
        global_dim = 256
        number_of_class = 2
        self.encode_lstm = LSTMEncoder(patch_size,hidden_dim)
        self.global_network = ResAsGlobal(global_dim)
        self.classification = SGNClassification(hidden_dim, global_dim,number_of_class)
        self.Segmentation = Segmentation(patch_size)

    def forward(self,x,x_global):
        local_feature = self.encode_lstm(x)
        global_feature = self.global_network(x_global)
        classification = self.classification(local_feature,global_feature)
        seg = self.Segmentation(local_feature)
        return classification,seg
        
def res_sgn(**kwargs):
    model = SGNResNet()
    model.apply(weights_init_kaiming)
    return model
