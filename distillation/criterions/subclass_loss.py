from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F
import torch
import distillation.architectures.tools as tools
import numpy as np

def sub_dense_prediction(scores, targets, record):
    

    num_target_levels = len(scores)
    loss_weights = [1.0 for _ in range(num_target_levels)]
    assert isinstance(loss_weights, (list, tuple))
    assert len(loss_weights) == num_target_levels

    loss_total = 0.0
    for i in range(num_target_levels):

        scores[i] = scores[i].permute(0, 2, 3, 1).contiguous()
        scores[i] = scores[i].view(-1, scores[i].size(3))
        scores_log = F.log_softmax(scores[i], dim=1)
        loss = F.kl_div(scores_log, targets[i], reduction='batchmean')

        loss_total = loss_total + loss_weights[i] * loss
        if num_target_levels > 1:
            record[f'loss_sub_l{i}'] = loss.item()

    record[f'loss_sub'] = loss_total.item()
    return loss_total, targets, record

class LDA_Layer(nn.Module):
    def __init__(self, scalings, xbar):
        super(LDA_Layer, self).__init__()

        weight = np.transpose(scalings)

        weight = torch.from_numpy(weight).to(torch.float32)
    
        weight = weight.unsqueeze(-1).unsqueeze(-1)
        
        self.weight = nn.Parameter(weight, requires_grad=False)
    
        bias = -np.matmul(np.transpose(scalings),xbar)
    
        bias = torch.from_numpy(bias).to(torch.float32)
        
        self.bias = nn.Parameter(bias, requires_grad=False)

    def forward(self, features):
        return F.conv2d(features, self.weight, bias=self.bias, stride=1, padding=0)

class subclass_loss(nn.Module):

    def __init__(self, num_classes, emb_size=4096, centers_filename=None, lda_filename=None, scores_filename=None):
        super(subclass_loss, self).__init__()


        
        with open(centers_filename, 'rb') as f:

            cluster_centers = np.load(f)
            
        with open(lda_filename + '_scalings.npy', 'rb') as f:

            lda_model_scalings = np.load(f)
            
        with open(lda_filename + '_xbar.npy', 'rb') as f:

            lda_model_xbar = np.load(f)
            
        self.each_subclass = int(emb_size/num_classes)
        self.num_classes = num_classes
        self.LDA = LDA_Layer(lda_model_scalings,lda_model_xbar)
        self.LDA_comp = cluster_centers.shape[1]
        
        self.cluster_centers = torch.from_numpy(cluster_centers).to(torch.float32)
        self.cluster_centers = self.cluster_centers.cuda()
        
        with open(scores_filename, 'rb') as f:

            teacher_scores = np.load(f)
        
        self.teacher_scores = torch.from_numpy(teacher_scores).to(torch.float32)
        self.teacher_scores = self.teacher_scores.cuda()
        

    def forward(self, feature_teacher, scores, labels, record):
        
        feature_teacher = self.LDA(feature_teacher)
        feature_teacher = feature_teacher[:,0:self.LDA_comp,:,:]
        
        feature_teacher = feature_teacher.permute(0, 2, 3, 1).contiguous()
        patches = feature_teacher.shape[1]**2

        # Flatten input
        flat_teacher = feature_teacher.view(-1, feature_teacher.shape[-1])
        
        distances = (torch.sum(flat_teacher**2, dim=1, keepdim=True)
                + torch.sum(self.cluster_centers**2, dim=1)
                - 2 * torch.matmul(flat_teacher, self.cluster_centers.t()))
        
        row_max = distances.max(axis=-1,keepdims=True).values
        
        label_mask = F.one_hot(labels, self.num_classes)
        label_mask = label_mask.repeat_interleave(self.each_subclass,dim=1)
        label_mask = label_mask.repeat_interleave(patches,dim=0)
        
        masked_distances = label_mask*(row_max-distances)
        encoding_targets = torch.argmax(masked_distances, dim=1)
        encoding_targets = F.one_hot(encoding_targets, self.each_subclass*self.num_classes).to(torch.float32)
        
        encoding_targets = torch.matmul(encoding_targets, self.teacher_scores)
        
        loss_sub, _, record = sub_dense_prediction(
            [scores,], [encoding_targets], record)
    
        
        return loss_sub, record
    
    
def create_criterion(opt):
    
    emb_size = opt['emb_size']
    centers_filename = opt['centers_filename']
    lda_filename = opt['lda_filename']
    scores_filename = opt['scores_filename']
    num_classes = opt['num_classes']

    return subclass_loss(num_classes = num_classes, emb_size = emb_size, 
                         centers_filename = centers_filename, lda_filename = lda_filename,
                         scores_filename = scores_filename)
