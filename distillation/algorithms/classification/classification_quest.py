from __future__ import print_function

import os
import tempfile
import torch
import torch.nn.functional as F

import distillation.algorithms.algorithm as algorithm
import distillation.algorithms.clustering.cluster_utils as cluster_utils
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

import numpy as np
from distillation.algorithms.classification.utils import extract_features
from distillation.utils import top1accuracy, accuracy


def quest_dense_prediction(predictor, features, targets, record):
    if not isinstance(features, (list, tuple)):
        features = [features,]
    if not isinstance(targets, (list, tuple)):
        targets = [targets,]
    assert len(features) == len(targets)

    num_target_levels = len(features)
    loss_weights = [1.0 for _ in range(num_target_levels)]
    assert isinstance(loss_weights, (list, tuple))
    assert len(loss_weights) == num_target_levels

    # targets shape: batch_size x 1 x height x width
    # features shape: batch_size x num_channels x height x width
    if len(features) == 1:
        scores = [predictor(features[0]),]
    else:
        scores = predictor(features)
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
    return scores, loss_total, targets, record


def object_classification_with_quest(
    feature_extractor,
    feature_extractor_optimizer,
    classifier,
    classifier_optimizer,
    BoW_predictor,
    BoW_predictor_optimizer,
    feature_extractor_target,
    vector_quantizer_target,
    images,
    labels,
    is_train,
    feature_name=None,
    quest_loss_coef=1.0,
    cls_loss_coef=1.0):

    assert images.dim() == 4
    assert labels.dim() == 1
    assert images.size(0) == labels.size(0)

    record = {}
    if is_train: # Zero gradients.
        feature_extractor_optimizer.zero_grad()
        classifier_optimizer.zero_grad()
        BoW_predictor_optimizer.zero_grad()

    # Extrack knowledge from teacher network.
    with torch.no_grad():
        feature_extractor_target.eval()
        features_target = extract_features(feature_extractor_target, images, feature_name)
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
        # Extract features from the images.
        features = extract_features(feature_extractor, images, feature_name)
        if not isinstance(features, (list, tuple)):
            features = [features,]
        assert len(features) == len(encoding_targets)

        # Perform the object classification task.
        scores_cls = classifier(features[-1])
        loss_cls = F.cross_entropy(scores_cls, labels)
        record['loss_cls'] = loss_cls.item()
        loss_total = loss_cls * cls_loss_coef

        _, loss_quest, _, record = quest_dense_prediction(
            BoW_predictor, features, encoding_targets, record)

        loss_total = loss_total + loss_quest * quest_loss_coef
        record['loss_total'] = loss_total.item()

    with torch.no_grad(): # Compute accuracies.
        AccTop1 = top1accuracy(scores_cls, labels)
        record['AccuracyTop1'] = AccTop1
        record['Error'] = 100.0 - AccTop1

    if is_train:
        # Backward loss and apply gradient steps.
        loss_total.backward()
        feature_extractor_optimizer.step()
        classifier_optimizer.step()
        BoW_predictor_optimizer.step()

    return record


class ClassificationQUEST(algorithm.Algorithm):
    def __init__(self, opt, _run=None, _log=None):
        super().__init__(opt, _run, _log)
        self.quest_loss_coef = opt.get('quest_loss_coef', 1.0)
        self.cls_loss_coef = opt.get('cls_loss_coef', 1.0)
        self.keep_best_model_metric_name = 'AccuracyTop1'
        feature_name = opt.get('feature_name', None)

        if feature_name:
            assert isinstance(feature_name, (list, tuple))

        self.feature_name = feature_name

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

        multiple_levels = (isinstance(self.feature_name, (list, tuple))
                          and (len(self.feature_name) > 1))
        if multiple_levels:
            vector_quantizer_target = [
                self.networks.get(f'vector_quantizer_target_{key}')
                for key in self.feature_name]
        else:
            vector_quantizer_target = self.networks.get(
                'vector_quantizer_target')

        record = object_classification_with_quest(
            feature_extractor=self.networks['feature_extractor'],
            feature_extractor_optimizer=self.optimizers.get('feature_extractor'),
            classifier=self.networks['classifier'],
            classifier_optimizer=self.optimizers.get('classifier'),
            BoW_predictor=self.networks.get('BoW_predictor'),
            BoW_predictor_optimizer=self.optimizers.get('BoW_predictor'),
            feature_extractor_target=self.networks.get('feature_extractor_target'),
            vector_quantizer_target=vector_quantizer_target,
            images=self.tensors['images'],
            labels=self.tensors['labels'],
            is_train=is_train,
            feature_name=self.feature_name,
            quest_loss_coef=self.quest_loss_coef,
            cls_loss_coef=self.cls_loss_coef)

        return record

    def apply_kmeans_to_dataset(
        self,
        dataloader,
        num_embeddings,
        feature_name=None,
        memmap=False,
        memmap_dir=None):
        feature_extractor = self.networks['feature_extractor']
        feature_extractor.eval()

        self.dloader = dataloader
        dataloader_iterator = dataloader.get_iterator()

        self.logger.info(f'==> Extract features from dataset.')

        if memmap_dir is not None:
            os.makedirs(memmap_dir, exist_ok=True)
            tempfile.tempdir = memmap_dir
        with tempfile.TemporaryFile() as fp:
            all_features_dataset, _ = cluster_utils.extract_features_from_dataset(
                feature_extractor=feature_extractor,
                dataloader_iterator=dataloader_iterator,
                feature_name=feature_name,
                logger=self.logger,
                memmap_filename=(fp if memmap else None))

            # clustering algorithm to use
            self.logger.info(f'==> Apply kmeans to dataset')
            deepcluster = cluster_utils.Kmeans(
                num_embeddings, preprocess=False)
            clustering_loss = deepcluster.cluster(
                all_features_dataset, verbose=True)
            centroids = deepcluster.centroids
            cluster_size = deepcluster.cluster_size
            points2ids = deepcluster.point2ids
            

            vector_quantizer = cluster_utils.initialize_vector_quantizer(
                centroids=centroids, cluster_size=cluster_size,
                commitment_cost=0.25, decay=0.99, epsilon=1e-5)
            vector_quantizer = vector_quantizer.to(self.device)

            prefix = f'vector_quantizer_kmeansK{num_embeddings}'
            filename = self._get_net_checkpoint_filename(
                prefix, self.curr_epoch)
            state = {
                'epoch': self.curr_epoch,
                'network': vector_quantizer.state_dict(),
                'metric': None,}
            self.logger.info(
                f'==> Saving vector quantizer with kmeans centroids to {filename}')
            torch.save(state, filename)

    def apply_kmeans_to_dataset_incremental_sklearn(
        self,
        dataloader,
        num_embeddings,
        feature_name=None,
        memmap=False,
        memmap_dir=None):
        feature_extractor = self.networks['feature_extractor']
        feature_extractor.eval()

        self.dloader = dataloader
        dataloader_iterator = dataloader.get_iterator()

        self.logger.info(f'==> Extract features from dataset.')

        if memmap_dir is not None:
            os.makedirs(memmap_dir, exist_ok=True)
            tempfile.tempdir = memmap_dir
        with tempfile.TemporaryFile() as fp:
            all_features_dataset, _ = cluster_utils.extract_features_from_dataset(
                feature_extractor=feature_extractor,
                dataloader_iterator=dataloader_iterator,
                feature_name=feature_name,
                logger=self.logger,
                memmap_filename=(fp if memmap else None))

            # clustering algorithm to use
            self.logger.info(f'==> Apply kmeans to dataset')
            
            kmeans = MiniBatchKMeans(n_clusters=num_embeddings,
                                     random_state=0,
                                     batch_size=64,
                                     verbose=0).fit(all_features_dataset)

            centroids = kmeans.cluster_centers_
            
            points2ids = []
        
            iter_number = 1000
            K = int(all_features_dataset.shape[0]/iter_number)
    
            for i in tqdm(range(iter_number)):
        
                related_features = all_features_dataset[i*K:i*K+K]
                
                indexes = kmeans.predict(related_features)
                
                points2ids.append(indexes)
            
            points2ids = list(np.hstack(points2ids))
            images_lists = [[] for i in range(num_embeddings)]
            
            self.logger.info(f'==> Apply images_lists to clusters')
            
            for i in range(len(all_features_dataset)):
                cluster_id = points2ids[i]
                images_lists[cluster_id].append(i)

            cluster_size = [len(images_lists[i]) for i in range(num_embeddings)]


            vector_quantizer = cluster_utils.initialize_vector_quantizer(
                centroids=centroids, cluster_size=cluster_size,
                commitment_cost=0.25, decay=0.99, epsilon=1e-5)
            vector_quantizer = vector_quantizer.to(self.device)

            prefix = f'vector_quantizer_kmeansK{num_embeddings}'
            filename = self._get_net_checkpoint_filename(
                prefix, self.curr_epoch)
            state = {
                'epoch': self.curr_epoch,
                'network': vector_quantizer.state_dict(),
                'metric': None,}
            self.logger.info(
                f'==> Saving vector quantizer with kmeans centroids to {filename}')
            torch.save(state, filename)