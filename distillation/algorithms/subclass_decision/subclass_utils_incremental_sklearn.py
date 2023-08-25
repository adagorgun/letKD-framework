import time

import random
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.decomposition import PCA, IncrementalPCA
import os
import pickle as pk

from tqdm import tqdm

colors_per_class = {
    0 : [254, 202, 87],
    1 : [255, 107, 107],
    2 : [10, 189, 227],
    3 : [255, 159, 243],
    4 : [16, 172, 132],
    5 : [128, 80, 128],
    6 : [255, 150, 100],
    7 : [52, 31, 151],
    8 : [100, 100, 255],
    9 : [102, 255, 102]
}


def split_features_block(model, features, which_stages, which_blocks):
    
        
    features = model[0:which_stages](features)
    
    if len(model[which_stages]._modules.keys()) > 1:
    
        features = model[which_stages][0:which_blocks+1](features)
        
    else:
        
        features = model[which_stages].layer[0:which_blocks+1](features)
            
    return features
        



def extract_inner_features_from_dataset(
    feature_extractor,
    dataloader_iterator,
    which_stages=None,
    downscale=False,
    which_blocks=None,
    logger=None):

    if isinstance(which_stages, (list, tuple)):
        assert len(which_stages) == 1

    feature_extractor.eval()

    all_features_dataset = None
    count = 0
    count_im = 0
    for i, batch in enumerate(tqdm(dataloader_iterator)):
        with torch.no_grad():
            
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            targets = batch[1]
            
            images = images.cuda()
            targets = targets.cuda()
            
            assert images.dim()==4

            features = split_features_block(feature_extractor, images, which_stages, which_blocks)
            
            batch_size = features.size()[0]
            
            # features = F.pad(features, (1,2,1,2))
              
            # features = F.avg_pool2d(features, (4,4), stride=(1,1))
            
            if downscale:
                features = F.avg_pool2d(features, (2,2), stride=(2,2))
            
            repeat_size = features.size()[2]
                                    
            features = features.permute(0, 2, 3, 1).contiguous()
            
            features = features.view(-1, features.size(3))
            
            all_data_per_iter = features.size()[0]
            channel_size = features.size()[1]

            features = features.cpu().numpy()
            targets = targets.cpu().numpy()
            
            

        if all_features_dataset is None:
            dataset_shape = len(dataloader_iterator) * all_data_per_iter
            all_features_dataset = np.zeros((dataset_shape,channel_size), dtype='float32')
            all_targets_repeated = np.zeros((dataset_shape), dtype='float32')


        all_features_dataset[count:(count + all_data_per_iter)] = features
        
        all_targets_repeated[count:(count + all_data_per_iter)] = np.repeat(targets, repeat_size**2, axis=0)
        count += all_data_per_iter
        count_im += batch_size

    all_features_dataset = all_features_dataset[:count]
    all_targets_repeated = all_targets_repeated[:count]

    if logger:
        logger.info(f'Shape of extracted dataset: {all_features_dataset.shape}')

    return all_features_dataset, all_targets_repeated

def apply_LDA(
        features,
        targets,
        num_components,
        directory):
    
    # filename_model = os.path.join(directory, 'lda_model.pk')
    filename_scalings = os.path.join(directory, 'lda_model_scalings.npy')
    filename_xbar = os.path.join(directory, 'lda_model_xbar.npy')
    
    if not os.path.exists(filename_scalings):
    
        # pca = PCA(n_components=64)
        # reduced_features = pca.fit_transform(features, targets)
        
        lda = LDA(n_components=num_components)
        transformed_features = lda.fit_transform(features, targets)
        
        # with open(filename_model,"wb") as f:
        #     pickle.dump(lda, f)
        
        
        with open(filename_scalings, 'wb') as f:
    
            np.save(f, lda.scalings_)
            
            
        with open(filename_xbar, 'wb') as f:
    
            np.save(f, lda.xbar_)
        
    else:
        
        with open(filename_scalings, 'rb') as f:
    
            scalings = np.load(f)
            
            
        with open(filename_xbar, 'rb') as f:
    
            xbar = np.load(f)
            
        # transformed_features = lda.transform(features)
        # transformed_features = np.dot(features - xbar, scalings)
        
        
        transformed_features = np.zeros((features.shape[0],num_components))
        
        iter_number = 10000
        K = int(features.shape[0]/iter_number)

        for i in range(iter_number):
    
            related_features = features[i*K:i*K+K]
            
            t_features = np.dot(related_features - xbar, scalings)
            
            transformed_features[i*K:i*K+K] = t_features[:,0:num_components]
    
        
    return transformed_features

def apply_KMeans(
        features,
        k_centers):
    
    # kmeans = MiniBatchKMeans(n_clusters=k_centers,
    #                                  random_state=0,
    #                                  batch_size=256).fit(features)
    
    kmeans = KMeans(n_clusters=k_centers,
                    random_state=0).fit(features)

    centers = kmeans.cluster_centers_
   
    points2ids = []
        
    iter_number = 10
    K = int(features.shape[0]/iter_number)

    for i in range(iter_number):

        related_features = features[i*K:i*K+K]
        
        indexes = kmeans.predict(related_features)
        
        points2ids.append(indexes)
                
    predictions = np.hstack(points2ids)
    
    return predictions, centers

def apply_PCA(
        features,
        n_components,
        directory):
    
    filename_components = os.path.join(directory, 'pca_model_components.npy')
    filename_mean = os.path.join(directory, 'pca_model_mean.npy')
    
    if not os.path.exists(filename_components):
    
    
        ipca = IncrementalPCA(n_components=n_components, batch_size=256).fit(features)
        
        components = ipca.components_
        mean = ipca.mean_
        
        with open(filename_components, 'wb') as f:
    
            np.save(f, components)            
            
        with open(filename_mean, 'wb') as f:
    
            np.save(f, mean)
        
    else:
        
        with open(filename_components, 'rb') as f:
    
            components = np.load(f)            
            
        with open(filename_mean, 'rb') as f:
    
            mean = np.load(f)

    transformed_features = np.zeros((features.shape[0],n_components))
        
    iter_number = 10000
    K = int(features.shape[0]/iter_number)

    for i in range(iter_number):

        related_features = features[i*K:i*K+K]
        
        t_features = np.dot(related_features - mean, np.transpose(components))
        
        transformed_features[i*K:i*K+K] = t_features[:,0:n_components]
    
    return transformed_features


def apply_NNCenter(
        all_features,
        subclass_centers,
        predictions):
    
    nearest_class = np.zeros((all_features.shape[0],1))
    iter_number = 10000
    K = int(all_features.shape[0]/iter_number)
    
    for i in tqdm(range(iter_number)):
        
        related_features = all_features[i*K:i*K+K]
        distances = (np.sum(related_features**2, axis=1, keepdims=True)
                    + np.sum(subclass_centers**2, axis=1)
                    - 2 * np.matmul(related_features, subclass_centers.T))
    
        nearest_class[i*K:i*K+K] = np.argpartition(distances, 1)[:,:1]
    

    # num_classes = len(np.unique(targets))
    num_subclasses = len(np.unique(predictions))
    
    scores = np.zeros((num_subclasses, num_subclasses))
    
    
    for s_cl in tqdm(range(num_subclasses)):
        
        index = np.where(predictions==s_cl)[0]
        related_class_centers = nearest_class[index]
        num_elements = related_class_centers.shape[0]
        # print(f'For SubClass {s_cl}, there are {num_elements} items')
        
        for s_cl_i in range(num_subclasses):
            
            num_items = np.sum(related_class_centers==s_cl_i)
            scores[s_cl, s_cl_i] = num_items/num_elements
            
    return scores
