from __future__ import print_function

import os
import tempfile
import torch
import torch.nn.functional as F

import distillation.algorithms.algorithm as algorithm
import distillation.algorithms.clustering.cluster_utils as cluster_utils
from distillation.algorithms.tSNE.tsne_utils import *


from distillation.utils import top1accuracy, accuracy


def quest_dense_prediction(scores, targets, record):
    

    num_target_levels = len(scores)
    loss_weights = [1.0 for _ in range(num_target_levels)]
    assert isinstance(loss_weights, (list, tuple))
    assert len(loss_weights) == num_target_levels

    # targets shape: batch_size x 1 x height x width
    # features shape: batch_size x num_channels x height x width
   
    # scores shape: batch_size x num_words x height x width

    loss_total = 0.0
    for i in range(num_target_levels):
        # scores[i] shape [batch_size x clusters x height x width]
        # targets[i] shape [batch_size x height x width x clusters]
        assert scores[i].dim() == 4
        assert targets[i].size(0) == scores[i].size(0) # batch size
        assert targets[i].size(1) == scores[i].size(2) # height
        assert targets[i].size(2) == scores[i].size(3) # width
        assert targets[i].size(3) == scores[i].size(1) # channels/clusters

        targets[i] = targets[i].view(-1, targets[i].size(3))
        scores[i] = scores[i].permute(0, 2, 3, 1).contiguous()
        scores[i] = scores[i].view(-1, scores[i].size(3))
        scores_log = F.log_softmax(scores[i], dim=1)
        loss = F.kl_div(scores_log, targets[i], reduction='batchmean')

        loss_total = loss_total + loss_weights[i] * loss
        if num_target_levels > 1:
            record[f'loss_quest_l{i}'] = loss.item()

        with torch.no_grad():
            key = 'Accur_vword' if (num_target_levels == 1) else f'Accur_vword_l{i}'
            targets[i] = targets[i].max(dim=1)[1]
            record[key] = top1accuracy(scores[i], targets[i])

    record[f'loss_quest'] = loss_total.item()
    return loss_total, targets, record



def object_classification_with_subclass_quest(
    feature_extractor,
    feature_extractor_optimizer,
    classifier,
    classifier_optimizer,
    BoW_predictor,
    BoW_predictor_optimizer,
    feature_extractor_target,
    vector_quantizer_target,
    valueLayer,
    valueLayer_optimizer,
    which_stage_t,
    which_stage_s,
    images,
    labels,
    is_train,
    quest_loss_coef=1.0,
    sub_loss_coef=1.0,
    cls_loss_coef=1.0):

    assert images.dim() == 4
    assert labels.dim() == 1
    assert images.size(0) == labels.size(0)
    
    t_do_without_layer = 0  # WRN requires .layer
    s_do_without_layer = 0  # WRN requires .layer
    t_rest = 0  # some models do not end with stage3
    s_rest = 0  # some models do not end with stage3

    record = {}
    if is_train: # Zero gradients.
        feature_extractor_optimizer.zero_grad()
        classifier_optimizer.zero_grad()
        for i in range(len(BoW_predictor)):
            BoW_predictor_optimizer[i].zero_grad()
            valueLayer_optimizer[i].zero_grad()

    # Extrack knowledge from teacher network.
    with torch.no_grad():
        feature_extractor_target.eval()
        
        if [*feature_extractor_target[which_stage_t]._modules][0] != 'layer':
            t_do_without_layer = 1
            
        if len(feature_extractor_target._modules.keys()) - which_stage_t > 1:
            t_rest = 1
        
        features_target_hint = feature_extractor_target[0:which_stage_t](images)
        
        if t_do_without_layer:
            
            features_target_hint = feature_extractor_target[which_stage_t][0](features_target_hint)
            features_target = feature_extractor_target[which_stage_t][1:](features_target_hint)

        else:
            features_target_hint = feature_extractor_target[which_stage_t].layer[0](features_target_hint)            
            features_target = feature_extractor_target[which_stage_t].layer[1:](features_target_hint)
            
        if t_rest:
            features_target = feature_extractor_target[which_stage_t+1:](features_target)

        if not isinstance(features_target, (list, tuple)):
            features_target = [features_target, ]

    if not isinstance(vector_quantizer_target, (list, tuple)):
        vector_quantizer_target = [vector_quantizer_target, ]
    encoding_targets = []
    num_vq_targets = len(features_target)
    for k in range(num_vq_targets):
        vector_quantizer_target[k].eval()
        _, _, perp_trg, _, enc_this, mean_assign_score = (
            vector_quantizer_target[k](features_target[k], True))
        encoding_targets.append(enc_this)
        record[f'perp_trg_{k}'] = perp_trg.item()
        record[f'mean_assign_dist_{k}'] = mean_assign_score.item()

    with torch.set_grad_enabled(is_train):
        
        if len(feature_extractor._modules.keys()) == 1:  # MobileNet v2 Imagenet
        
             features_sc = feature_extractor[0][0:which_stage_s+1](images)
             
             scores = BoW_predictor[0](features_sc)
            
             features_hint, _, loss_sub, record = valueLayer[0](features_target_hint, scores, features_sc, labels, record)
             
             features_final_stage = feature_extractor[0][which_stage_s+1:](features_hint)
        
        else:
        
            if [*feature_extractor[which_stage_s]._modules][0] != 'layer':
                s_do_without_layer = 1
                
            if len(feature_extractor._modules.keys()) - which_stage_s > 1:
                s_rest = 1
            # Extract features from the images.
            features_sc = feature_extractor[0:which_stage_s](images)
            
            if s_do_without_layer:        
                features_sc = feature_extractor[which_stage_s][0](features_sc)
            else:
                features_sc = feature_extractor[which_stage_s].layer[0](features_sc)
            
            scores = BoW_predictor[0](features_sc)
            
            features_hint, _, loss_sub, record = valueLayer[0](features_target_hint, scores, features_sc, labels, record)
            
            if s_do_without_layer: 
                features_final_stage = feature_extractor[which_stage_s][1:](features_hint)
            else:
                features_final_stage = feature_extractor[which_stage_s].layer[1:](features_hint)
                
            if s_rest:
                features_final_stage = feature_extractor[which_stage_s+1:](features_final_stage)
        
        scores_final_stage = BoW_predictor[1](features_final_stage)
        features, _ = valueLayer[1](scores_final_stage, features_final_stage)
                
        
        loss_quest, _, record = quest_dense_prediction(
            [scores_final_stage], encoding_targets, record)

        # Perform the object classification task.
        scores_cls = classifier(features)
        loss_cls = F.cross_entropy(scores_cls, labels)
        record['loss_cls'] = loss_cls.item()
        loss_total = loss_cls * cls_loss_coef        

        loss_total = loss_total + loss_quest * quest_loss_coef + loss_sub * sub_loss_coef
        record['loss_total'] = loss_total.item()

    with torch.no_grad(): # Compute accuracies.
        AccTop1 = top1accuracy(scores_cls, labels)
        # AccTop1, AccTop5 = compute_top1_and_top5_accuracy(scores_cls, labels)
        record['AccuracyTop1'] = AccTop1
        # record['AccuracyTop5'] = AccTop5
        record['Error'] = 100.0 - AccTop1


    if is_train:
        # Backward loss and apply gradient steps.
        loss_total.backward()
        feature_extractor_optimizer.step()
        classifier_optimizer.step()
        for i in range(len(BoW_predictor)):
            BoW_predictor_optimizer[i].step()
            valueLayer_optimizer[i].step()

    return record


class ClassificationSubQUESTValue(algorithm.Algorithm):
    def __init__(self, opt, _run=None, _log=None):
        super().__init__(opt, _run, _log)
        self.quest_loss_coef = opt.get('quest_loss_coef', 1.0)
        self.sub_loss_coef = opt.get('sub_loss_coef', 1.0)
        self.cls_loss_coef = opt.get('cls_loss_coef', 1.0)
        self.which_stage_t = opt.get('which_stage_t', 5)
        self.which_stage_s = opt.get('which_stage_s', 5)
        self.keep_best_model_metric_name = 'AccuracyTop1'

    def allocate_tensors(self):
        self.tensors = {}
        self.tensors['images'] = torch.FloatTensor()
        self.tensors['labels'] = torch.LongTensor()

    def set_tensors(self, batch):
        assert len(batch) == 2
        images, labels = batch
        self.tensors['images'].resize_(images.size()).copy_(images)
        self.tensors['labels'].resize_(labels.size()).copy_(labels)

        return 'classification'

    def train_step(self, batch):
        return self.process_batch_classification_task(batch, is_train=True)

    def evaluation_step(self, batch):
        return self.process_batch_classification_task(batch, is_train=False)

    def process_batch_classification_task(self, batch, is_train):
        self.set_tensors(batch)


            
        BoW_predictor = [
            self.networks.get(f'BoW_predictor_{key}')
            for key in range(2)]
        
        BoW_predictor_optimizer = [
            self.optimizers.get(f'BoW_predictor_{key}')
            for key in range(2)]
        
        valueLayer = [
            self.networks.get(f'valueLayer_{key}')
            for key in range(2)]
        
        valueLayer_optimizer = [
            self.optimizers.get(f'valueLayer_{key}')
            for key in range(2)]
            
        
        vector_quantizer_target = self.networks.get(
            'vector_quantizer_target')

        record = object_classification_with_subclass_quest(
            feature_extractor=self.networks['feature_extractor'],
            feature_extractor_optimizer=self.optimizers.get('feature_extractor'),
            classifier=self.networks['classifier'],
            classifier_optimizer=self.optimizers.get('classifier'),
            BoW_predictor=BoW_predictor,
            BoW_predictor_optimizer=BoW_predictor_optimizer,
            feature_extractor_target=self.networks.get('feature_extractor_target'),
            vector_quantizer_target=vector_quantizer_target,
            valueLayer=valueLayer,
            valueLayer_optimizer=valueLayer_optimizer,
            which_stage_t=self.which_stage_t,
            which_stage_s=self.which_stage_s,
            images=self.tensors['images'],
            labels=self.tensors['labels'],
            is_train=is_train,
            quest_loss_coef=self.quest_loss_coef,
            sub_loss_coef=self.sub_loss_coef,
            cls_loss_coef=self.cls_loss_coef)

        return record
    
    def apply_tSNE(
        self,
        dataloader,
        file_dir):
        
        feature_extractor = self.networks['feature_extractor']
        BoW_predictor = [
            self.networks.get(f'BoW_predictor_{key}')
            for key in range(2)]
        valueLayer = [
            self.networks.get(f'valueLayer_{key}')
            for key in range(2)]
        
        f_b_s, f_a_s, v_s, softmax, targets, f_b_q, f_a_q, v_q, weight_value = extract_tSNE_features_with_subclass_value_v3(dataloader,
                                                                                                                       feature_extractor,
                                                                                                                       BoW_predictor,
                                                                                                                       valueLayer)
        
        weight_value = weight_value/np.max(weight_value, axis=0, keepdims=True)
        v_s_max = np.max(v_s)
        weight_value = weight_value*v_s_max
        
        all_labels = np.repeat(np.arange(10), 8, axis=0)
        
                
        visualizeEmbeddingPointsSuperClass(weight_value, all_labels, file_dir, 'deneme')
        
        
        # visualizeEmbeddingPoints(weight_value, all_labels, file_dir, 'value_s')
        
        # selected_classes = [0,30,35,90,45]
        # ind_list = take_specific_class(targets,
        #                                1280,
        #                                selected_classes)
         
        
        # targets = targets[ind_list.astype(int)]
        # f_b_q = f_b_q[ind_list.astype(int)]
        # f_a_q = f_a_q[ind_list.astype(int)]
        # v_q = v_q[ind_list.astype(int)]
        
        # f_b_s = f_b_s[ind_list.astype(int)]
        # f_a_s = f_a_s[ind_list.astype(int)]
        # softmax = softmax[ind_list.astype(int)]
        # v_s = v_s[ind_list.astype(int)]

         
        # visualizeEmbeddingPoints(v_s, targets, selected_classes, file_dir, 'value_s')
        # visualizeEmbeddingPoints(softmax, targets, selected_classes, file_dir, 'softmax')
        # visualizeEmbeddingPoints(f_b_s, targets, selected_classes, file_dir, 'before_value_tree')
        # visualizeEmbeddingPoints(f_a_s, targets, selected_classes, file_dir, 'after_value_tree')
        # visualizeEmbeddingPoints(v_q, targets, selected_classes, file_dir, 'value_quest')
        # visualizeEmbeddingPoints(f_a_q, targets, selected_classes, file_dir, 'after_value_quest')
        # visualizeEmbeddingPoints(f_b_q, targets, selected_classes, file_dir, 'before_value_quest')
    
    # def apply_tSNE(
    #     self,
    #     dataloader,
    #     file_dir):
        
    #     feature_extractor = self.networks['feature_extractor']
    #     BoW_predictor = [
    #         self.networks.get(f'BoW_predictor_{key}')
    #         for key in range(2)]
    #     valueLayer = [
    #         self.networks.get(f'valueLayer_{key}')
    #         for key in range(2)]
        
    #     features_before, features_after, value, softmax, targets_f, targets_s, f_quest_b, f_quest, v_quest = extract_tSNE_features_with_subclass_value_v2(dataloader,
    #                                                                                                                    feature_extractor,
    #                                                                                                                    BoW_predictor,
    #                                                                                                                    valueLayer)
        
        
    #     ind_list_f = take_balanced(targets_f,
    #                                1000)
         
        
    #     targets_f = targets_f[ind_list_f.astype(int)]
    #     f_quest_b = f_quest_b[ind_list_f.astype(int)]
    #     f_quest = f_quest[ind_list_f.astype(int)]
    #     v_quest = v_quest[ind_list_f.astype(int)]
    #     softmax = softmax[ind_list_f.astype(int)]
    #     value = value[ind_list_f.astype(int)]
        
        
    #     # ind_list_s = take_balanced(targets_s,
    #     #                            1000)
    #     # features_before = features_before[ind_list_f.astype(int)]
    #     # features_after = features_after[ind_list_f.astype(int)]
    #     # targets_s = targets_s[ind_list_s.astype(int)]
         
    #     visualizeEmbeddingPoints(value, targets_f, file_dir, 'value')
    #     visualizeEmbeddingPoints(softmax, targets_f, file_dir, 'softmax')
    #     visualizeEmbeddingPoints(features_before, targets_s, file_dir, 'before_value_tree')
    #     visualizeEmbeddingPoints(features_after, targets_s, file_dir, 'after_value_tree')
    #     visualizeEmbeddingPoints(v_quest, targets_f, file_dir, 'value_quest')
    #     visualizeEmbeddingPoints(f_quest, targets_f, file_dir, 'after_value_quest')
    #     visualizeEmbeddingPoints(f_quest_b, targets_f, file_dir, 'before_value_quest')
    
    # def apply_tSNE(
    #         self,
    #         dataloader,
    #         file_dir):
        
    #     feature_extractor = self.networks['feature_extractor']
    #     BoW_predictor = [
    #         self.networks.get(f'BoW_predictor_{key}')
    #         for key in range(2)]
    #     valueLayer = [
    #         self.networks.get(f'valueLayer_{key}')
    #         for key in range(2)]
        
    #     f_b_sub, scores, v_sub, f_a_sub, f_b_quest, v_quest, f_a_quest, target = extract_tSNE_features_with_subclass_value(dataloader,
    #                                                                                                                    feature_extractor,
    #                                                                                                                    BoW_predictor,
    #                                                                                                                    valueLayer)
        
    #     target = np.repeat(target, 64, axis=0)
         
    #     visualizeEmbeddingPoints(v_sub, target, file_dir, 'value_sub')
    #     visualizeEmbeddingPoints(scores, target, file_dir, 'softmax')
    #     visualizeEmbeddingPoints(f_b_sub, target, file_dir, 'before_value_sub')
    #     visualizeEmbeddingPoints(f_a_sub, target, file_dir, 'after_value_sub')
    #     visualizeEmbeddingPoints(v_quest, target, file_dir, 'value_quest')
    #     visualizeEmbeddingPoints(f_a_quest, target, file_dir, 'after_value_quest')
    #     visualizeEmbeddingPoints(f_b_quest, target, file_dir, 'before_value_quest')


