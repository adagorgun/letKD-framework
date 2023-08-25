from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F
import torch
import distillation.architectures.tools as tools
import numpy as np

class Classifier(nn.Module):
    def __init__(self, s_shape, t_shape, scale, learn_scale=True):
        super(Classifier, self).__init__()

        self.classifier_type = 'conv_cosine'
        self.num_channels = s_shape
        self.num_classes = t_shape
        self.global_pooling = False
        bias = False

        if self.classifier_type == 'linear':
            
            self.layers = nn.Linear(
                self.num_channels, self.num_classes, bias=bias)
            if bias:
                self.layers.bias.data.zero_()
            fout = self.layers.out_features
            self.layers.weight.data.normal_(0.0,  np.sqrt(2.0/fout))
        elif self.classifier_type == 'conv_cosine':
            assert self.global_pooling is False
            self.layers = tools.Conv2dCos(
                self.num_channels, self.num_classes, bias=bias,
                scale=scale, learn_scale=learn_scale)
        else:
            raise ValueError(f'Not supported classifier type {self.classifier_type}')

    def flatten(self):
        return self.classifier_type == 'linear'

    def forward(self, features):
        if self.global_pooling:
            features = tools.global_pooling(features, pool_type='avg')

        if features.dim() > 2 and self.flatten():
            features = features.view(features.size(0), -1)
        scores = self.layers(features)

        return scores

class valueLayer(nn.Module):

    def __init__(self, inp_size, emb_size=4096, value_ratio=1.0):
        super(valueLayer, self).__init__()


        # self.conv_value = nn.Conv2d(emb_size, 64, kernel_size=1, bias=False)
        self.conv_value = Classifier(emb_size, inp_size, scale=1, learn_scale=True)
        
        self.bn = nn.BatchNorm2d(emb_size)
        self.relu = nn.ReLU(inplace=True) 
        self.value_ratio = value_ratio      
        
        # self.conv_sum = nn.Conv2d(64, 64, kernel_size=1, bias=False)

    def forward(self, scores, stage_out):
        
        out = self.relu(self.bn(scores))

        value = self.conv_value(out)
        
        features = self.value_ratio*value + stage_out
        
        return features, value
    
class valueLayer_1x1sum(nn.Module):

    def __init__(self, inp_size, emb_size=4096):
        super(valueLayer_1x1sum, self).__init__()


        # self.conv_value = nn.Conv2d(emb_size, 64, kernel_size=1, bias=False)
        self.conv_value = Classifier(emb_size, inp_size, scale=1, learn_scale=True)
        
        self.bn = nn.BatchNorm2d(emb_size)
        self.relu = nn.ReLU(inplace=True)        
        
        self.conv_sum = nn.Conv2d(inp_size, inp_size, kernel_size=1, bias=False)

    def forward(self, scores, stage_out):
        
        out = self.relu(self.bn(scores))

        value = self.conv_value(out)
        
        features = value + self.conv_sum(stage_out)
        
        return features, value
        
    
        

def create_model(opt):

    emb_size = opt['emb_size']
    inp_size = opt['inp_size']
    value_ratio = opt.get('value_ratio', 1.0)

    return valueLayer(inp_size = inp_size, emb_size = emb_size, value_ratio = value_ratio)
    # return valueLayer_1x1sum(inp_size = inp_size, emb_size = emb_size)
