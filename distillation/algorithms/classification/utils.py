from __future__ import print_function

import torch
import torch.nn.functional as F
import numpy as np
from distillation.utils import compute_top1_and_top5_accuracy, top1accuracy, accuracy


def extract_features(feature_extractor, images, feature_name=None):
    if feature_name:
        if isinstance(feature_name, str):
            feature_name = [feature_name,]
        assert isinstance(feature_name, (list, tuple))
        return feature_extractor(images, out_feat_keys=feature_name)
    else:
        return feature_extractor(images)


def object_classification(
    feature_extractor,
    feature_extractor_optimizer,
    classifier,
    classifier_optimizer,
    images,
    labels,
    is_train,
    criterions,
    feature_name=None):

    assert labels.dim() == 1
    assert images.size(0) == labels.size(0)

    if is_train: # Zero gradients.
        feature_extractor_optimizer.zero_grad()
        classifier_optimizer.zero_grad()

    record = {}
    with torch.set_grad_enabled(is_train):
        # Extract features from the images.
        features = extract_features(
            feature_extractor, images, feature_name=feature_name)
        # Perform the object classification task.
        scores = classifier(features)
        
        loss = F.cross_entropy(scores, labels)
        # loss = criterions['loss'](scores, labels) #cross-entropy
        record['loss'] = loss.item()

    with torch.no_grad(): # Compute accuracies.
        #1
        # accur_top1, accur_top5 = compute_top1_and_top5_accuracy(scores, labels)
        # record['AccuracyTop1'] = accur_top1
        # record['AccuracyTop5'] = accur_top5
        
        #2
        # AccTop1 = top1accuracy(scores, labels)     
        # record['AccuracyTop1'] = AccTop1
        # record['Error'] = 100.0 - AccTop1
        
        #3
        Accur_top1, Accur_top5 = accuracy(scores, labels, topk=(1, 5))
        record['AccuracyTop1'] = Accur_top1.cpu().numpy()[0].astype(np.float64)
        record['AccuracyTop5'] = Accur_top5.cpu().numpy()[0].astype(np.float64)

    if is_train: # Backward loss and apply gradient steps.
        loss.backward()
        feature_extractor_optimizer.step()
        classifier_optimizer.step()

    return record
