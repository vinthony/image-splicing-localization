import torch
import torchvision
import torch.nn as nn
import numpy as np
import math
import torchvision.models as models

class LSTMEncoder(nn.Module):
    """docstring for Encoder"""
    def __init__(self, input_dim, hidden_dim):
        super(LSTMEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.conv1 = nn.Conv2d(3, 16, 5, padding=2,bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(16, 1, 5, padding=2,bias=False) 
        self.relu2 = nn.ReLU(inplace=True)
        self.lstm = nn.LSTM(self.input_dim,self.hidden_dim,num_layers=3,batch_first=True)
        self.lstm.flatten_parameters() 

    def forward(self, x):
        bs = x.size(0)
        
        x = self.relu1(self.conv1(x)) # bsx16x64x64

        y = self.relu2(self.conv2(x)) # bsx1x64x64

        # split x to 8x8 blocks
        y_list = y.split(8,dim=3) # 8x[(bsx1x8x64)]
        xy_list = [ x.split(8,dim=2) for x in y_list] # 8x8x( bsx 1x 8 x 8)

        xy = [item for items in xy_list for item in items]

        xy = torch.cat(xy,1) # bsx64x(8x8)

        xy = xy.view(bs,64,64) # bs x 64 x 64

        self.lstm.flatten_parameters() 
        # 8x8 list 
        outputs, (ht, ct) = self.lstm(xy)

        return outputs

class Segmentation(nn.Module):
    def __init__(self,patch_size):
        super(Segmentation,self).__init__()

        self.sqrt_patch_size = int(math.sqrt(patch_size))
        self.patch_size = patch_size

        self.conv3 = nn.Conv2d(1, 32, 5, padding=2,bias=False) 
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(32, 2, 5, padding=2,bias=False) 
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(2, 2, 5, padding=2,bias=False) 
        self.relu5 = nn.ReLU(inplace=True)

        self.max_pool = nn.MaxPool2d(2,stride=2)

        self.softmax = nn.LogSoftmax(dim=1)


    def forward(self,outputs_lstm):
        bs = outputs_lstm.size(0)
        # outputs = [bs,64,256]
        outputs_lstm = outputs_lstm.contiguous().view(bs,self.sqrt_patch_size,self.sqrt_patch_size,self.sqrt_patch_size*2,self.sqrt_patch_size*2).permute(0,1,3,2,4).contiguous()

        # bs,8*16,8*16
        outputs_lstm = outputs_lstm.view(bs,1,self.patch_size*2,self.patch_size*2)

        # bs x 32 x 96x96
        x = self.relu3(self.conv3(outputs_lstm))

        x = self.max_pool(x)

        x = self.relu4(self.conv4(x))

        output_mask = self.softmax(self.conv5(x))

        return output_mask
        
class Classification(nn.Module):
    """docstring for Classification"""
    def __init__(self, hidden_dim, number_of_class=2):
        super(Classification, self).__init__()
        self.hidden_dim = hidden_dim
        self.number_of_class = number_of_class

        self.linear = nn.Linear(self.hidden_dim, self.number_of_class)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, outputs_lstm):

        outputs = outputs_lstm[:,-1,:] #bsx256

        y = self.softmax(self.linear(outputs))

        return y

class SGNClassification(nn.Module):
    """docstring for Classification"""
    def __init__(self, hidden_dim, global_dim, number_of_class=2):
        super(SGNClassification, self).__init__()
        self.hidden_dim = hidden_dim
        self.number_of_class = number_of_class
        self.global_dim = global_dim

        self.linear = nn.Linear(self.hidden_dim+self.global_dim, self.number_of_class)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self,local_feature,global_feature):

        local_feature = local_feature[:,-1,:] #bsx256

        outputs = torch.cat([local_feature,global_feature],1)

        y = self.softmax(self.linear(outputs))

        return y


class GlobalNetwork(nn.Module):
    """docstring for GlobalNetwork"""
    def __init__(self,global_feature_channel):
        super(GlobalNetwork, self).__init__()

        self.global_feature_channel = global_feature_channel

        self.global_network = nn.Sequential(
                nn.Conv2d(3,16,3,padding=1,stride=2),# 112
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(16),
                nn.Conv2d(16,16,3,padding=1,stride=2), # 56
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(16),
                nn.Conv2d(16,32,3,padding=1,stride=2), # 28
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(32),
                nn.Conv2d(32,32,3,padding=1,stride=2), # 14
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(32),
                nn.Conv2d(32,64,3,padding=1,stride=2), # 7
                nn.ReLU(inplace=True),
                )        
        
            
        self.global_classification = nn.Linear(64*7*7,self.global_feature_channel)

    def forward(self,x):
        x = self.global_network(x)
        x = x.view(x.size(0),-1)
        x = self.global_classification(x)
        return x

    
class ResAsGlobal(nn.Module):
    """docstring for GlobalNetwork"""
    def __init__(self,global_feature_channel):
        super(ResAsGlobal, self).__init__()
        
        self.global_feature_channel = global_feature_channel
        
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        resnet_feature = nn.Sequential(*modules)
        
        self.global_network = resnet_feature      
        
         # Freeze those weights
        for p in self.global_network.parameters():
            p.requires_grad = False
            
        self.global_classification = nn.Linear(resnet.fc.in_features,self.global_feature_channel)
        
        

    def forward(self,x):
        x = self.global_network(x)
        x = x.view(x.size(0),-1)
        x = self.global_classification(x)
        return x
