import time

import random
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import AgglomerativeClustering, KMeans
import distillation.algorithms.clustering.cluster_utils as cluster_utils
import imageio
from skimage import img_as_ubyte
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


import pickle
import os
import cv2
from skimage import color

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

def calculate_linkage(model):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    return linkage_matrix

def extract_features(model, images, which_stages):
    
    total_length_network = len(model._modules.keys())
    
    features = model[0:3](images)
    
        
    for i in range(3, total_length_network): 
    
        features = model[i](features)
            
        if i == which_stages :
            
            break
            
    return features

def split_features_block(model, features, which_stages, which_blocks):
    
        
    features = model[0:which_stages](features)
    
    if len(model[which_stages]._modules.keys()) > 1:
    
        features = model[which_stages][0:which_blocks+1](features)
        
    else:
        
        features = model[which_stages].layer[0:which_blocks+1](features)
            
    return features
        


def extract_features_from_dataset(
    feature_extractor,
    dataloader_iterator,
    which_stages=None,
    logger=None):

    if isinstance(which_stages, (list, tuple)):
        assert len(which_stages) == 1

    feature_extractor.eval()

    all_features_dataset = None
    count = 0
    for i, batch in enumerate(tqdm(dataloader_iterator)):
        with torch.no_grad():
            
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            targets = batch[1]
            
            images = images.cuda()
            
            assert images.dim()==4

            features = extract_features(feature_extractor, images, which_stages)
                        
            features = F.avg_pool2d(features, (4,4), stride=(1,1))
            
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
            all_targets = np.zeros((dataset_shape), dtype='float32')


        all_features_dataset[count:(count + all_data_per_iter)] = features
        
        all_targets[count:(count + all_data_per_iter)] = np.repeat(targets, repeat_size**2, axis=0)
        count += all_data_per_iter

    all_features_dataset = all_features_dataset[:count]
    all_targets = all_targets[:count]

    if logger:
        logger.info(f'Shape of extracted dataset: {all_features_dataset.shape}')

    return all_features_dataset, all_targets

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
            
            # images = F.pad(images, (4,8,4,8))
            im_shape = images.size()[2]
            images = images.permute(0, 2, 3, 1).contiguous()
            images = images.cpu().numpy()

        if all_features_dataset is None:
            dataset_shape = len(dataloader_iterator) * all_data_per_iter
            all_features_dataset = np.zeros((dataset_shape,channel_size), dtype='float32')
            all_targets = np.zeros((batch_size*len(dataloader_iterator)), dtype='float32')
            all_targets_repeated = np.zeros((dataset_shape), dtype='float32')
            all_images = np.zeros((batch_size*len(dataloader_iterator),im_shape,im_shape,3), dtype='float32')


        all_features_dataset[count:(count + all_data_per_iter)] = features
        all_images[count_im:(count_im + batch_size)] = images
        
        all_targets[count_im:(count_im + batch_size)] = targets 
        all_targets_repeated[count:(count + all_data_per_iter)] = np.repeat(targets, repeat_size**2, axis=0)
        count += all_data_per_iter
        count_im += batch_size

    all_features_dataset = all_features_dataset[:count]
    all_images = all_images[:count_im]
    all_targets = all_targets[:count_im]
    all_targets_repeated = all_targets_repeated[:count]

    if logger:
        logger.info(f'Shape of extracted dataset: {all_features_dataset.shape}')

    return all_features_dataset, all_images, all_targets, all_targets_repeated

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
   
    deepcluster = cluster_utils.Kmeans(
                   k_centers, preprocess=False)
    clustering_loss = deepcluster.cluster(
        features, verbose=True)
    
    predictions = np.array(deepcluster.point2ids)
    centers = deepcluster.centroids
    
    return predictions, centers


def apply_NNCenter(
        all_features,
        subclass_centers,
        predictions):
    
    nearest_class = np.zeros((all_features.shape[0],1))
    iter_number = 10000
    K = int(all_features.shape[0]/iter_number)
    
    for i in range(iter_number):
        
        related_features = all_features[i*K:i*K+K]
        distances = (np.sum(related_features**2, axis=1, keepdims=True)
                    + np.sum(subclass_centers**2, axis=1)
                    - 2 * np.matmul(related_features, subclass_centers.T))
    
        nearest_class[i*K:i*K+K] = np.argpartition(distances, 1)[:,:1]
    

    # num_classes = len(np.unique(targets))
    num_subclasses = len(np.unique(predictions))
    
    scores = np.zeros((num_subclasses, num_subclasses))
    
    
    for s_cl in range(num_subclasses):
        
        index = np.where(predictions==s_cl)[0]
        related_class_centers = nearest_class[index]
        num_elements = related_class_centers.shape[0]
        # print(f'For SubClass {s_cl}, there are {num_elements} items')
        
        for s_cl_i in range(num_subclasses):
            
            num_items = np.sum(related_class_centers==s_cl_i)
            scores[s_cl, s_cl_i] = num_items/num_elements
            
    return scores


def draw_rectangle_class_pred(image, label):
    
      
    # image[0:1,0:1] = np.array(colors_per_subclass[label])
    image[1:3,1:3] = np.array(colors_per_class[label])

    return image
          
def visualizeSubClassInformation(
        images,
        targets,
        predictions,
        directory):

    num_classes = len(np.unique(targets)) 
    total_claases = len(np.unique(predictions))
    sub_classes = int((total_claases/num_classes))
    size = 16
    stride = 4
    
    # num_patches = images.shape[1]//4
    
    for cl in range(num_classes):
        
        index = np.where(targets==cl)[0]
        c_ind = np.random.choice(index,1)
        im = images[c_ind[0]]
        
        arr_min = np.min(im)
        arr_max = np.max(im)
        
        im = (im- arr_min) / (arr_max - arr_min)
        
        im = np.uint8(im*255)
        
        # im = np.int8(np.round(cv2.normalize(im, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)))
        
        patches = extract_patches(im, size,stride)
        related_predictions = predictions[c_ind[0]*64:c_ind[0]*64+64]%sub_classes

        # patches = im.reshape(num_patches, size, num_patches, size,3).swapaxes(1, 2).reshape(-1, size, size, 3)
        print(patches.shape)
        
        construct_image = np.zeros((im.shape[0],im.shape[1],3))
        num_patches = int((im.shape[0] - size)/stride + 1)
        
        prefix = f'im/original_image_class_{cl}.png'
        filename = os.path.join(directory, prefix)
        
        imageio.imwrite(filename,im)
        
        for p_r in range(num_patches):
            for p_c in range(num_patches):
                
                # if p_c*stride%size == 0 and p_r*stride%size == 0:
                
                patches_used = np.uint8(draw_rectangle_class_pred(patches[num_patches*p_r+p_c], related_predictions[num_patches*p_r+p_c]))
                construct_image[p_r*stride:p_r*stride+size, p_c*stride:p_c*stride+size,:] = patches_used
        
                # imageio.imwrite(f'{num_patches*p_r+p_c}.png',patches_used)
        prefix = f'im/construct_image_class_{cl}.png'
        filename = os.path.join(directory, prefix)
        
        imageio.imwrite(filename,construct_image)
        print('done') 
        
def take_balanced(targets,
                  num_el_per_class):
    
    num_classes = len(np.unique(targets))
    ind_list = []

    
    for i in range(num_classes):
        
        index = np.where(targets == i)[0]
        c_ind = np.random.choice(index,num_el_per_class)
        ind_list = np.concatenate((ind_list, c_ind))
    
    
    
    
    return ind_list.astype('int')

def color_patch(image, label):
    
    b = image
    g = image
    r = image
    
    r_m, g_m, b_m = colors_per_class[label]

    b_new = np.multiply(b, b_m, casting="unsafe")
    g_new = np.multiply(g, g_m, casting="unsafe")
    r_new = np.multiply(r, r_m, casting="unsafe")

    image_after = cv2.merge([r_new, g_new, b_new])    
    
    return image_after

def visualizeColorPatch(
        images,
        targets,
        predictions,
        selected_indexes,
        num_el_per_class,
        directory):
    
    if not os.path.exists(os.path.join(directory, 'im_color_patch')):
        os.makedirs(os.path.join(directory, 'im_color_patch'))

    num_classes = len(np.unique(targets)) 
    total_claases = len(np.unique(predictions))
    sub_classes = int((total_claases/num_classes))
    size = 4
    stride = 4
        
    for cl in range(num_classes):
        
        c_ind_main = selected_indexes[cl*num_el_per_class:cl*num_el_per_class+num_el_per_class]
        num_im_per_class = 0
        
        for c_ind in c_ind_main:            
             
        
            im = images[c_ind]
            
            arr_min = np.min(im)
            arr_max = np.max(im)
            
            im = (im- arr_min) / (arr_max - arr_min)
            
            im = np.uint8(im*255)
            
            imgGray = color.rgb2gray(im)
                        
            patches = extract_patches(imgGray, size,stride)
            related_predictions = predictions[c_ind*64:c_ind*64+64]%sub_classes
            
            construct_image = np.uint8(np.zeros((im.shape[0],im.shape[1],3)))
            num_patches = int((im.shape[0] - size)/stride + 1)
            
            prefix = f'im_color_patch/original_image_class_{cl}_{num_im_per_class}.png'
            filename = os.path.join(directory, prefix)
            
            imageio.imwrite(filename,im)
            
            for p_r in range(num_patches):
                for p_c in range(num_patches):
                    
                    # if p_c*stride%size == 0 and p_r*stride%size == 0:
                    
                    patches_used = np.uint8(color_patch(patches[num_patches*p_r+p_c], related_predictions[num_patches*p_r+p_c]))
                    construct_image[p_r*stride:p_r*stride+size, p_c*stride:p_c*stride+size,:] = patches_used
            
                    # imageio.imwrite(f'im/{num_patches*p_r+p_c}_class_{cl}.png',patches_used)
            
            prefix = f'im_color_patch/construct_image_class_{cl}_{num_im_per_class}.png'
            filename = os.path.join(directory, prefix)
            
            imageio.imwrite(filename,construct_image)
            # print('done')  
            num_im_per_class += 1
        
        
def visualizeSubClassInformation_Sep(
        images,
        targets,
        predictions,
        selected_indexes,
        num_el_per_class,
        directory):
    
    if not os.path.exists(os.path.join(directory, 'im_point')):
        os.makedirs(os.path.join(directory, 'im_point'))

    num_classes = len(np.unique(targets)) 
    total_claases = len(np.unique(predictions))
    sub_classes = int((total_claases/num_classes))
    size = 4
    stride = 4
        
    for cl in range(num_classes):
        
        c_ind_main = selected_indexes[cl*num_el_per_class:cl*num_el_per_class+num_el_per_class]
        num_im_per_class = 0
        
        for c_ind in c_ind_main:            
             
        
            im = images[c_ind]
            
            arr_min = np.min(im)
            arr_max = np.max(im)
            
            im = (im- arr_min) / (arr_max - arr_min)
            
            im = np.uint8(im*255)
                        
            patches = extract_patches(im, size,stride)
            related_predictions = predictions[c_ind*64:c_ind*64+64]%sub_classes
            
            construct_image = np.uint8(np.zeros((im.shape[0],im.shape[1],3)))
            num_patches = int((im.shape[0] - size)/stride + 1)
            
            prefix = f'im_point/original_image_class_{cl}_{num_im_per_class}.png'
            filename = os.path.join(directory, prefix)
            
            imageio.imwrite(filename,im)
            
            for p_r in range(num_patches):
                for p_c in range(num_patches):
                    
                    # if p_c*stride%size == 0 and p_r*stride%size == 0:
                    
                    patches_used = np.uint8(draw_rectangle_class_pred(patches[num_patches*p_r+p_c], related_predictions[num_patches*p_r+p_c]))
                    construct_image[p_r*stride:p_r*stride+size, p_c*stride:p_c*stride+size,:] = patches_used
            
                    # imageio.imwrite(f'im/{num_patches*p_r+p_c}_class_{cl}.png',patches_used)
            
            prefix = f'im_point/construct_image_class_{cl}_{num_im_per_class}.png'
            filename = os.path.join(directory, prefix)
            
            imageio.imwrite(filename,construct_image)
            # print('done')  
            num_im_per_class += 1
            

def scale_image(image, max_image_size):
    image = np.array(image, dtype='uint8')

    image = cv2.resize(image, (max_image_size, max_image_size), interpolation=cv2.INTER_CUBIC)
    return image
            
def visualizePerPixelSubClass(
        images,
        targets,
        predictions,
        selected_indexes,
        num_el_per_class,
        directory):
    
    if not os.path.exists(os.path.join(directory, 'im_interp')):
        os.makedirs(os.path.join(directory, 'im_interp'))
        

    num_classes = len(np.unique(targets)) 
    total_claases = len(np.unique(predictions))
    sub_classes = int((total_claases/num_classes))
    
    for cl in range(num_classes):
        
        c_ind_main = selected_indexes[cl*num_el_per_class:cl*num_el_per_class+num_el_per_class]

        num_im_per_class = 0
        
        for c_ind in c_ind_main:            
             
        
            im = images[c_ind]
            
            arr_min = np.min(im)
            arr_max = np.max(im)
            
            im = (im- arr_min) / (arr_max - arr_min)
            
            im = np.uint8(im*255)
            
            prefix = f'im_interp/original_image_class_{cl}_{num_im_per_class}.png'
            filename = os.path.join(directory, prefix)
            
            imageio.imwrite(filename,im)

            related_predictions = predictions[c_ind*64:c_ind*64+64]%sub_classes
            related_predictions = np.reshape(related_predictions,(8,8))
            
            prediction_image = np.zeros((8,8,3))
            
            for w in range(8):
                
                for h in range(8):
                    
                    prediction_image[w,h,:] = colors_per_class[related_predictions[w,h]]
    
            prediction_image = scale_image(prediction_image,32)
            
            prefix = f'im_interp/predict_image_class_{cl}_{num_im_per_class}.png'
            filename = os.path.join(directory, prefix)
            
            imageio.imwrite(filename,prediction_image)
            num_im_per_class += 1

        
def extract_patches(
        image,
        size,
        stride):
    

    num_patches = int((image.shape[0] - size)/stride + 1)
    
    try: 
        
        channel_size = image.shape[2]
    
        patches = np.zeros((num_patches**2, size, size, channel_size))
        
    except:
        
        patches = np.zeros((num_patches**2, size, size))
    
    for p_r in range(num_patches):
        for p_c in range(num_patches):
        
            patches[num_patches*p_r+p_c] = image[p_r*stride:p_r*stride+size, p_c*stride:p_c*stride+size]
            
    return patches




def lerp(a, b, t):
    return a*(1 - t) + b*t


            
def visualizeScoreHistogram(all_scores,targets,directory):
    
    if not os.path.exists(os.path.join(directory, 'preds')):
        os.makedirs(os.path.join(directory, 'preds'))
    
    num_classes = len(np.unique(targets)) 
    num_sub_classes = int(all_scores.shape[0]/num_classes)
    white = np.array([255, 255, 255])
    
    length_per_sub = 4
    
    for cl in range(num_classes):
        
        template = np.zeros((num_sub_classes*length_per_sub, all_scores.shape[0]*length_per_sub+num_classes-1, 3))
        
        class_info = all_scores[cl*num_sub_classes:cl*num_sub_classes+num_sub_classes,cl*num_sub_classes:cl*num_sub_classes+num_sub_classes]
        class_info = np.max(np.max(class_info))
        
        for s_cl_row in range(num_sub_classes):
            color = np.array(colors_per_class[s_cl_row])
            
            for s_cl_column in range(all_scores.shape[1]):   
                
                score = all_scores[cl*num_sub_classes+s_cl_row,s_cl_column]/class_info #np.max(all_scores[cl*num_sub_classes+s_cl_row])
                lightened = lerp(white, color, score)
                
                increment = int(np.floor(s_cl_column/num_sub_classes))
                template[s_cl_row*length_per_sub:s_cl_row*length_per_sub+length_per_sub,
                         s_cl_column*length_per_sub+increment:s_cl_column*length_per_sub+length_per_sub+increment] = np.uint8(lightened)
            
            
        prefix = f'scores/score_image_class_{cl}.png'
        filename = os.path.join(directory, prefix)
        imageio.imwrite(filename,template)
        

class_id = ['plane',
            'car',
            'bird',
            'cat',
            'deer',
            'dog',
            'frog',
            'horse',
            'ship',
            'truck']        
def visualizeBarHistogram(all_scores,targets,directory):
    
    if not os.path.exists(os.path.join(directory, 'preds')):
        os.makedirs(os.path.join(directory, 'preds'))
        
    num_classes = len(np.unique(targets))  
    num_sub_classes = int(all_scores.shape[0]/num_classes)
        
    x_ticks = np.arange(0,num_classes*num_sub_classes,num_sub_classes)
    x_axis = np.arange(0,num_classes*num_sub_classes)
    fig, ax = plt.subplots()
    
    for s_cl in range(all_scores.shape[0]):
        
        color = np.array(colors_per_class[s_cl%num_sub_classes]) / 255
        
        
        scores = all_scores[s_cl]
        ax.bar(x_axis, scores, color = color.reshape(1,-1))
        ax.set_xticks(x_ticks)
        
        prefix = f'preds/score_image_class_{s_cl}.png'
        filename = os.path.join(directory, prefix)
        
        fig.savefig(filename, dpi=300)
        plt.cla()
    plt.close(fig)
        
        
    
    