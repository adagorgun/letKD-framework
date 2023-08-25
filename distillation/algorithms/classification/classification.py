from __future__ import print_function

import torch

import distillation.algorithms.algorithm as algorithm
import distillation.algorithms.classification.utils as utils
import os
import tempfile
import numpy as np
from tqdm import tqdm
import distillation.algorithms.subclass_decision.subclass_utils_incremental_sklearn as subclass_utils_inc
import distillation.algorithms.subclass_decision.subclass_utils as subclass_utils

from sklearn.utils import shuffle

class Classification(algorithm.Algorithm):
    def __init__(self, opt, _run=None, _log=None):
        super().__init__(opt, _run, _log)
        feature_name = opt.get('feature_name', None)

        if feature_name:
            assert isinstance(feature_name, (list, tuple))

        self.feature_name = feature_name
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

        if is_train and (self.optimizers.get('feature_extractor') is None):
            self.networks['feature_extractor'].eval()

        record = utils.object_classification(
            feature_extractor=self.networks['feature_extractor'],
            feature_extractor_optimizer=self.optimizers.get('feature_extractor'),
            classifier=self.networks['classifier'],
            classifier_optimizer=self.optimizers.get('classifier'),
            images=self.tensors['images'],
            labels=self.tensors['labels'],
            is_train=is_train,
            criterions = self.criterions,
            feature_name=self.feature_name)

        return record
    
    def apply_sublass(
        self,
        dataloader,
        num_components,
        num_subclass,
        which_stages=None,
        directory=None,
        visualize=False):
        feature_extractor = self.networks['feature_extractor']
        feature_extractor.eval()

        self.dloader = dataloader
        dataloader_iterator = dataloader.get_iterator()

        self.logger.info(f'==> Extract features from dataset.')


        # all_features_dataset, all_targets = tree_like_utils.extract_features_from_dataset(
        #     feature_extractor=feature_extractor,
        #     dataloader_iterator=dataloader_iterator,
        #     which_stages=which_stages,
        #     logger=self.logger)
        
        all_features_dataset, all_images, all_targets, all_targets_r = subclass_utils.extract_inner_features_from_dataset(
            feature_extractor=feature_extractor,
            dataloader_iterator=dataloader_iterator,
            which_stages=which_stages,
            downscale=False,  # cross models
            which_blocks=0,
            logger=self.logger)

        if not visualize:
            del all_images
            del all_targets

        self.logger.info(f'==> Apply LDA to dataset')
        
        features_LDA = subclass_utils.apply_LDA(
            features=all_features_dataset,
            targets=all_targets_r,
            num_components=num_components,
            directory=directory)
            
        self.logger.info(f'==> Apply Divisive Clustering to dataset')
        
        prefix = f'centers_subclass_{num_subclass}_{num_components}'
        filename = os.path.join(directory, prefix +'.npy')
        
        if not os.path.exists(filename):
        
            num_classes = len(np.unique(all_targets_r))
            all_predictions = np.zeros(all_targets_r.shape)
            all_centers = np.zeros((num_classes*num_subclass,features_LDA.shape[1]))
            
            for cl in range(num_classes):
                index = np.where(all_targets_r == cl)[0]
                all_elements = features_LDA[index]
                predictions, centers = subclass_utils.apply_KMeans(
                    features=all_elements,
                    k_centers=num_subclass)
                predictions = num_subclass*cl + predictions
                all_predictions[index] = predictions
                all_centers[cl*num_subclass:cl*num_subclass+num_subclass] = centers

            with open(filename, 'wb') as f:        
                np.save(f, all_centers)
                
        else:
            
            with open(filename, 'rb') as f:        
                all_centers = np.load(f)
                
            all_predictions = np.zeros(all_targets_r.shape)
            num_classes = len(np.unique(all_targets_r))
                
            iter_number = 10000
            K = int(features_LDA.shape[0]/iter_number)
    
            for i in range(iter_number):
        
                related_features = features_LDA[i*K:i*K+K]
                labels = all_targets_r[i*K:i*K+K]
                distances = (np.sum(related_features**2, axis=1, keepdims=True)
                            + np.sum(all_centers**2, axis=1)
                            - 2 * np.matmul(related_features, all_centers.T))
                
                row_max = np.max(distances, axis=-1)
                
                label_mask = np.zeros((labels.size, num_classes))
                label_mask[np.arange(labels.size), labels.astype('int')] = 1
            
                label_mask = label_mask.repeat(num_subclass,axis=1)
            
                masked_distances = label_mask*(np.subtract(np.expand_dims(row_max,1),distances))
                encoding_targets = np.argmax(masked_distances, axis=1)
                
                all_predictions[i*K:i*K+K] = encoding_targets
                

        if visualize:
            
            num_el_per_class = 20
            
            selected_indexes = subclass_utils.take_balanced(targets=all_targets,
                                                            num_el_per_class=num_el_per_class)
            
            subclass_utils.visualizeSubClassInformation_Sep(all_images,
                                                            all_targets,
                                                            all_predictions,
                                                            selected_indexes,
                                                            num_el_per_class,
                                                            directory)
            
            subclass_utils.visualizePerPixelSubClass(all_images,
                                                      all_targets,
                                                      all_predictions,
                                                      selected_indexes,
                                                      num_el_per_class,
                                                      directory)
            
            subclass_utils.visualizeColorPatch(all_images,
                                               all_targets,
                                               all_predictions,
                                               selected_indexes,
                                               num_el_per_class,
                                               directory)
            
            
        self.logger.info(f'==> Apply Average Features') 
        
        prefix = f'final_scores_{num_subclass}_{num_components}'
        filename = os.path.join(directory, prefix +'.npy')
        
        if os.path.exists(filename):
        
            with open(filename, 'rb') as f:        
                final_scores = np.load(f)
                
        else:
            
            num_subclasses = len(np.unique(all_predictions))
                
            averaged_features = np.zeros((num_subclasses,features_LDA.shape[1]))
            
            for s_cl in range(num_subclasses):
                
                index = np.where(all_predictions == s_cl)[0]
                all_elements = features_LDA[index]
                averaged_features[s_cl] = np.mean(all_elements, axis=0)
                
            del all_features_dataset
            del all_centers
            
            
    
            final_scores = subclass_utils.apply_NNCenter(
                all_features=features_LDA,
                subclass_centers=averaged_features,
                predictions=all_predictions)

            with open(filename, 'wb') as f:
    
                np.save(f, final_scores)
                
  
        if visualize and len(np.unique(all_targets)) == 10:
                
            subclass_utils.visualizeBarHistogram(final_scores,
                                                 all_targets,
                                                 directory)
        
        
        
    def apply_sublass_incremental_sklearn(
        self,
        dataloader,
        num_components,
        num_subclass,
        PCA,
        which_stages=None,
        directory=None):
        feature_extractor = self.networks['feature_extractor']
        feature_extractor.eval()

        self.dloader = dataloader
        dataloader_iterator = dataloader.get_iterator()

        self.logger.info(f'==> Extract features from dataset.')

        
        all_features_dataset, all_targets_r = subclass_utils_inc.extract_inner_features_from_dataset(
            feature_extractor=feature_extractor,
            dataloader_iterator=dataloader_iterator,
            which_stages=which_stages,
            downscale=False,  # cross models
            which_blocks=0,
            logger=self.logger)

        
        if PCA>0:
            
            self.logger.info(f'==> Apply shuffle')
            
            all_features_dataset, all_targets_r = shuffle(all_features_dataset, all_targets_r, random_state=0)
            
            self.logger.info(f'==> Apply PCA {PCA}')
        
            all_features_dataset = subclass_utils_inc.apply_PCA(features=all_features_dataset,
                                                            n_components=PCA,
                                                            directory=directory)
        
        self.logger.info(f'==> Apply LDA to dataset {num_components}')
        
        features_LDA = subclass_utils_inc.apply_LDA(
            features=all_features_dataset,
            targets=all_targets_r,
            num_components=num_components,
            directory=directory)
            
        self.logger.info(f'==> Apply Divisive Clustering to dataset')
        
        prefix = f'centers_subclass_{num_subclass}_{num_components}'
        filename = os.path.join(directory, prefix +'.npy')
        
        if not os.path.exists(filename):
        
            num_classes = len(np.unique(all_targets_r))
            all_predictions = np.zeros(all_targets_r.shape)
            all_centers = np.zeros((num_classes*num_subclass,features_LDA.shape[1]))
            
            for cl in tqdm(range(num_classes)):
                index = np.where(all_targets_r == cl)[0]
                all_elements = features_LDA[index]
                predictions, centers = subclass_utils_inc.apply_KMeans(
                    features=all_elements,
                    k_centers=num_subclass)
                predictions = num_subclass*cl + predictions
                all_predictions[index] = predictions
                all_centers[cl*num_subclass:cl*num_subclass+num_subclass] = centers

            with open(filename, 'wb') as f:        
                np.save(f, all_centers)
                
        else:
            
            with open(filename, 'rb') as f:        
                all_centers = np.load(f)
                
            all_predictions = np.zeros(all_targets_r.shape)
            num_classes = len(np.unique(all_targets_r))
                
            iter_number = 10000
            K = int(features_LDA.shape[0]/iter_number)
    
            for i in range(iter_number):
        
                related_features = features_LDA[i*K:i*K+K]
                labels = all_targets_r[i*K:i*K+K]
                distances = (np.sum(related_features**2, axis=1, keepdims=True)
                            + np.sum(all_centers**2, axis=1)
                            - 2 * np.matmul(related_features, all_centers.T))
                
                row_max = np.max(distances, axis=-1)
                
                label_mask = np.zeros((labels.size, num_classes))
                label_mask[np.arange(labels.size), labels.astype('int')] = 1
            
                label_mask = label_mask.repeat(num_subclass,axis=1)
            
                masked_distances = label_mask*(np.subtract(np.expand_dims(row_max,1),distances))
                encoding_targets = np.argmax(masked_distances, axis=1)
                
                all_predictions[i*K:i*K+K] = encoding_targets
 
        self.logger.info(f'==> Apply Average Features') 
        
        prefix = f'final_scores_{num_subclass}_{num_components}'
        filename = os.path.join(directory, prefix +'.npy')
        
        if os.path.exists(filename):
        
            with open(filename, 'rb') as f:        
                final_scores = np.load(f)
                
        else:
            
            num_subclasses = len(np.unique(all_predictions))
                
            averaged_features = np.zeros((num_subclasses,features_LDA.shape[1]))
            
            for s_cl in range(num_subclasses):
                
                index = np.where(all_predictions == s_cl)[0]
                all_elements = features_LDA[index]
                averaged_features[s_cl] = np.mean(all_elements, axis=0)
                
            del all_features_dataset
            del all_centers
            
            self.logger.info(f'==> Calculate Score') 
    
            final_scores = subclass_utils_inc.apply_NNCenter(
                all_features=features_LDA,
                subclass_centers=averaged_features,
                predictions=all_predictions)

            
            with open(filename, 'wb') as f:
    
                np.save(f, final_scores)        
        

