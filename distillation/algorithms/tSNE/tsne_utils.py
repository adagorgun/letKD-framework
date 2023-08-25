import time

import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE
import torch.nn.functional as F
from tqdm import tqdm


plt.ioff()

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

colors_per_superclass = {
    0 : [254, 202, 87],
    1 : [255, 107, 107],
    2 : [10, 189, 227],
    3 : [255, 159, 243],
    4 : [16, 172, 132],
}

markers = ["o", "v", "s", "*", "^", "p", "P", "D"]

# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range

def extract_tSNE_features_with_value(
    dataloader,
    feature_extractor,
    BoW_predictor,
    valueLayer):


    feature_extractor.eval()
    BoW_predictor.eval()
    valueLayer.eval()
    
    my_testiter = iter(dataloader)
    images, target = my_testiter.next()
    
    if torch.cuda.is_available():
        images = images.cuda()
        target = target.cuda()
    
    features_before = feature_extractor(images)
    
    scores = BoW_predictor(features_before)
    
    features_after, value = valueLayer(scores, features_before)
    
    scores = F.softmax(scores, dim=1)    
    
    features_before = features_before.permute(0, 2, 3, 1).contiguous()
    scores = scores.permute(0, 2, 3, 1).contiguous()
    value = value.permute(0, 2, 3, 1).contiguous()    
    features_after = features_after.permute(0, 2, 3, 1).contiguous()
    
    features_before = features_before.view(-1, features_before.size(3))
    scores = scores.view(-1, scores.size(3))
    value = value.view(-1, value.size(3))
    features_after = features_after.view(-1, features_after.size(3))
    
    features_before = features_before.cpu().detach().numpy()
    scores = scores.cpu().detach().numpy()
    value = value.cpu().detach().numpy()
    features_after = features_after.cpu().detach().numpy()
    target = target.cpu().detach().numpy()

    return features_before, scores, value, features_after, target

def extract_tSNE_features(
    dataloader,
    feature_extractor):

    feature_extractor.eval()
    
    my_testiter = iter(dataloader)
    images, target = my_testiter.next()
    
    if torch.cuda.is_available():
        images = images.cuda()
        target = target.cuda()
    
    features = feature_extractor(images)

    features = features.permute(0, 2, 3, 1).contiguous()
    
    features = features.view(-1, features.size(3))
    
    features = features.cpu().detach().numpy()
    target = target.cpu().detach().numpy()

    return features, target


def extract_tSNE_features_GAP(
    dataloader,
    feature_extractor):

    feature_extractor.eval()
    
    all_features_dataset = None
    count = 0
    num_iters = 20
    
    for i in range(num_iters):
    
        my_testiter = iter(dataloader)
        images, target = my_testiter.next()
        
        if torch.cuda.is_available():
            images = images.cuda()
            target = target.cuda()
        
        features = feature_extractor(images)
        
        features = F.avg_pool2d(features, 8)
        
        features = features.permute(0, 2, 3, 1).contiguous()
        
        features = features.view(-1, features.size(3))
        
        all_data_per_iter = features.size()[0]
        channel_size = features.size()[1]
        
        features = features.cpu().detach().numpy()
        target = target.cpu().detach().numpy()
        
        if all_features_dataset is None:
            dataset_shape = num_iters * all_data_per_iter
            all_features_dataset = np.zeros((dataset_shape,channel_size), dtype='float32')
            all_targets = np.zeros((dataset_shape), dtype='float32')
            
        all_features_dataset[count:(count + all_data_per_iter)] = features
        
        all_targets[count:(count + all_data_per_iter)] = target
        count += all_data_per_iter
        
    all_features_dataset = all_features_dataset[:count]
    all_targets = all_targets[:count]

    return all_features_dataset, all_targets

def extract_tSNE_features_with_subclass_value_v2(
    dataloader,
    feature_extractor,
    BoW_predictor,
    valueLayer):


    feature_extractor.eval()
    for i in range(2):
        BoW_predictor[i].eval()
        valueLayer[i].eval()
    
    all_f_b = None
    count_f = 0
    count_s = 0
    for i, batch in enumerate(tqdm(dataloader())):
    
            
        with torch.no_grad():
            
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            targets = batch[1]
            
            images = images.cuda()
                    
            f_b = feature_extractor[0:5](images)
            f_b = feature_extractor[5][0](f_b)
            
            scores = BoW_predictor[0](f_b)
            
            f_a, v_s, _,_ = valueLayer[0]([],scores, f_b, [], [])
            
            scores = F.softmax(scores, dim=1) 
            
            repeat_size = f_a.size()[2]
            
            f_b_q = feature_extractor[5][1:](f_a)
        
            scores_final_stage = BoW_predictor[1](f_b_q)
            f_a_q, v_q = valueLayer[1](scores_final_stage, f_b_q)
            
            f_b = F.avg_pool2d(f_b, (8,8), stride=(1,1))
            f_a = F.avg_pool2d(f_a, (8,8), stride=(1,1))
            
            f_a = f_a.permute(0, 2, 3, 1).contiguous()
            scores = scores.permute(0, 2, 3, 1).contiguous()
            v_s = v_s.permute(0, 2, 3, 1).contiguous()    
            f_b = f_b.permute(0, 2, 3, 1).contiguous()
            f_b_q = f_b_q.permute(0, 2, 3, 1).contiguous()
            v_q = v_q.permute(0, 2, 3, 1).contiguous()
            f_a_q = f_a_q.permute(0, 2, 3, 1).contiguous()
            
            f_a = f_a.view(-1, f_a.size(3))
            scores = scores.view(-1, scores.size(3))
            v_s = v_s.view(-1, v_s.size(3))
            f_b = f_b.view(-1, f_b.size(3))
            f_b_q = f_b_q.view(-1, f_b_q.size(3))
            v_q = v_q.view(-1, v_q.size(3))
            f_a_q = f_a_q.view(-1, f_a_q.size(3))
            
            all_data_per_iter_f_q = f_a_q.size()[0]
            all_data_per_iter_f_s = f_a.size()[0]
            channel_size_f = f_a_q.size()[1]
            channel_size_s = scores.size()[1]

            f_a = f_a.cpu().detach().numpy()
            scores = scores.cpu().detach().numpy()
            v_s = v_s.cpu().detach().numpy()
            f_b = f_b.cpu().detach().numpy()
            f_b_q = f_b_q.cpu().detach().numpy()
            v_q = v_q.cpu().detach().numpy()
            f_a_q = f_a_q.cpu().detach().numpy()
            targets = targets.numpy()
            
            
            if all_f_b is None:
                dataset_shape_f_q = len(dataloader) * all_data_per_iter_f_q
                dataset_shape_f_s = len(dataloader) * all_data_per_iter_f_s
                all_f_b = np.zeros((dataset_shape_f_s,channel_size_f), dtype='float32')
                all_f_a = np.zeros((dataset_shape_f_s,channel_size_f), dtype='float32')
                all_f_a_q = np.zeros((dataset_shape_f_q,channel_size_f), dtype='float32')
                all_f_b_q = np.zeros((dataset_shape_f_q,channel_size_f), dtype='float32')
                all_v_q = np.zeros((dataset_shape_f_q,channel_size_f), dtype='float32')
                all_v_s = np.zeros((dataset_shape_f_q,channel_size_f), dtype='float32')
                all_softmax = np.zeros((dataset_shape_f_q,channel_size_s), dtype='float32')
                all_targets_f_q = np.zeros((dataset_shape_f_q), dtype='float32')
                all_targets_f_s = np.zeros((dataset_shape_f_s), dtype='float32')
                
            all_f_b[count_s:(count_s + all_data_per_iter_f_s)] = f_b
            all_f_a[count_s:(count_s + all_data_per_iter_f_s)] = f_a
            all_f_a_q[count_f:(count_f + all_data_per_iter_f_q)] = f_a_q
            all_f_b_q[count_f:(count_f + all_data_per_iter_f_q)] = f_b_q
            all_v_q[count_f:(count_f + all_data_per_iter_f_q)] = v_q
            all_v_s[count_f:(count_f + all_data_per_iter_f_q)] = v_s
            all_softmax[count_f:(count_f + all_data_per_iter_f_q)] = scores
        
            all_targets_f_q[count_f:(count_f + all_data_per_iter_f_q)] = np.repeat(targets, repeat_size**2, axis=0)
            all_targets_f_s[count_s:(count_s + all_data_per_iter_f_s)] = targets
            count_f += all_data_per_iter_f_q
            count_s += all_data_per_iter_f_s
            
    all_f_b = all_f_b[:count_s]
    all_f_a = all_f_a[:count_s]
    all_f_a_q = all_f_a_q[:count_f]
    all_f_b_q = all_f_b_q[:count_f]
    all_v_q = all_v_q[:count_f]
    all_v_s = all_v_s[:count_f]
    all_softmax = all_softmax[:count_f]
    all_targets_f_q = all_targets_f_q[:count_f]
    all_targets_f_s = all_targets_f_s[:count_s]

    


    return all_f_b, all_f_a, all_v_s, all_softmax, all_targets_f_q, all_targets_f_s, all_f_b_q, all_f_a_q, all_v_q


def extract_tSNE_features_with_subclass_value_v3(
    dataloader,
    feature_extractor,
    BoW_predictor,
    valueLayer):


    feature_extractor.eval()
    for i in range(2):
        BoW_predictor[i].eval()
        valueLayer[i].eval()
    
    all_f_b = None
    count_f = 0
    count_s = 0
    for i, batch in enumerate(tqdm(dataloader())):
    
            
        with torch.no_grad():
            
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            targets = batch[1]
            
            images = images.cuda()
                    
            f_b = feature_extractor[0:5](images)
            f_b = feature_extractor[5][0](f_b)
            
            scores = BoW_predictor[0](f_b)
            
            f_a, v_s, weight_value,_,_ = valueLayer[0]([],scores, f_b, [], [])
            
            scores = F.softmax(scores, dim=1) 
            
            repeat_size = f_a.size()[2]
            
            f_b_q = feature_extractor[5][1:](f_a)
        
            scores_final_stage = BoW_predictor[1](f_b_q)
            f_a_q, v_q = valueLayer[1](scores_final_stage, f_b_q)
            
            # f_b = F.avg_pool2d(f_b, (8,8), stride=(1,1))
            # f_a = F.avg_pool2d(f_a, (8,8), stride=(1,1))
            
            f_a = f_a.permute(0, 2, 3, 1).contiguous()
            scores = scores.permute(0, 2, 3, 1).contiguous()
            v_s = v_s.permute(0, 2, 3, 1).contiguous()    
            f_b = f_b.permute(0, 2, 3, 1).contiguous()
            f_b_q = f_b_q.permute(0, 2, 3, 1).contiguous()
            v_q = v_q.permute(0, 2, 3, 1).contiguous()
            f_a_q = f_a_q.permute(0, 2, 3, 1).contiguous()
            
            f_a = f_a.view(-1, f_a.size(3))
            scores = scores.view(-1, scores.size(3))
            v_s = v_s.view(-1, v_s.size(3))
            f_b = f_b.view(-1, f_b.size(3))
            f_b_q = f_b_q.view(-1, f_b_q.size(3))
            v_q = v_q.view(-1, v_q.size(3))
            f_a_q = f_a_q.view(-1, f_a_q.size(3))
            
            all_data_per_iter_f_q = f_a_q.size()[0]
            all_data_per_iter_f_s = f_a.size()[0]
            channel_size_f = f_a_q.size()[1]
            channel_size_s = scores.size()[1]

            f_a = f_a.cpu().detach().numpy()
            scores = scores.cpu().detach().numpy()
            v_s = v_s.cpu().detach().numpy()
            f_b = f_b.cpu().detach().numpy()
            f_b_q = f_b_q.cpu().detach().numpy()
            v_q = v_q.cpu().detach().numpy()
            f_a_q = f_a_q.cpu().detach().numpy()
            targets = targets.numpy()
            
            
            
            if all_f_b is None:
                dataset_shape_f_q = len(dataloader) * all_data_per_iter_f_q
                dataset_shape_f_s = len(dataloader) * all_data_per_iter_f_s
                all_f_b = np.zeros((dataset_shape_f_s,channel_size_f), dtype='float32')
                all_f_a = np.zeros((dataset_shape_f_s,channel_size_f), dtype='float32')
                all_f_a_q = np.zeros((dataset_shape_f_q,channel_size_f), dtype='float32')
                all_f_b_q = np.zeros((dataset_shape_f_q,channel_size_f), dtype='float32')
                all_v_q = np.zeros((dataset_shape_f_q,channel_size_f), dtype='float32')
                all_v_s = np.zeros((dataset_shape_f_q,channel_size_f), dtype='float32')
                all_softmax = np.zeros((dataset_shape_f_q,channel_size_s), dtype='float32')
                all_targets = np.zeros((dataset_shape_f_q), dtype='float32')
                
                
            all_f_b[count_s:(count_s + all_data_per_iter_f_s)] = f_b
            all_f_a[count_s:(count_s + all_data_per_iter_f_s)] = f_a
            all_f_a_q[count_f:(count_f + all_data_per_iter_f_q)] = f_a_q
            all_f_b_q[count_f:(count_f + all_data_per_iter_f_q)] = f_b_q
            all_v_q[count_f:(count_f + all_data_per_iter_f_q)] = v_q
            all_v_s[count_f:(count_f + all_data_per_iter_f_q)] = v_s
            all_softmax[count_f:(count_f + all_data_per_iter_f_q)] = scores
        
            all_targets[count_f:(count_f + all_data_per_iter_f_q)] = np.repeat(targets, repeat_size**2, axis=0)
            count_f += all_data_per_iter_f_q
            count_s += all_data_per_iter_f_s
            
    all_f_b = all_f_b[:count_s]
    all_f_a = all_f_a[:count_s]
    all_f_a_q = all_f_a_q[:count_f]
    all_f_b_q = all_f_b_q[:count_f]
    all_v_q = all_v_q[:count_f]
    all_v_s = all_v_s[:count_f]
    all_softmax = all_softmax[:count_f]
    all_targets = all_targets[:count_f]
    weight_value = weight_value.cpu().detach().numpy()

    


    return all_f_b, all_f_a, all_v_s, all_softmax, all_targets, all_f_b_q, all_f_a_q, all_v_q, weight_value


def take_specific_class(targets,
                        num_el,
                        which_class):
    
    ind_list = []
    
    for i in range(len(which_class)):
        
        index = np.where(targets == which_class[i])[0]
        c_ind = index[:num_el]
    
        ind_list = np.concatenate((ind_list, c_ind))
    

    return ind_list

def take_balanced(targets,
                  num_el_per_class):
    
    num_classes = 10
    ind_list = []

    
    for i in range(num_classes):
        
        index = np.where(targets == i)[0]
        c_ind = np.random.choice(index,num_el_per_class)
        ind_list = np.concatenate((ind_list, c_ind))
    
    
    
    
    return ind_list
    


def visualizeEmbeddingPoints(features, y_test_patches, labels, file_dir, name):
    
    tsne = TSNE(n_components=2).fit_transform(features)
    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]
    
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.gca().set_axis_off()
    num_classes = 0
    
    # for every class, we'll add a scatter plot separately
    for label in labels:
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(y_test_patches) if l == label]
    
        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)
    
        # convert the class color to matplotlib format
        color = np.array(colors_per_class[num_classes], dtype=np.float) / 255
        num_classes += 1
    
        # add a scatter plot with the corresponding color and label
        ax.scatter(current_tx, current_ty, c=color.reshape(1,-1), s = 12, label=label, alpha=0.5, edgecolors =color.reshape(1,-1))
    
    # build a legend using the labels we set previously
    # ax.legend(loc='best')
    
    # finally, show the plot
    # plt.show()
    
    filename = os.path.join(file_dir, '%s.png' % name)
    
    fig.savefig(filename, dpi=300)
    
def extract_tSNE_features_with_subclass_value_v1(
    dataloader,
    feature_extractor,
    BoW_predictor,
    valueLayer):


    feature_extractor.eval()
    for i in range(2):
        BoW_predictor[i].eval()
        valueLayer[i].eval()
    
    my_testiter = iter(dataloader)
    images, target = my_testiter.next()
    
    if torch.cuda.is_available():
        images = images.cuda()
        target = target.cuda()
    
    
    features_before_sub = feature_extractor[0:5](images)
    features_before_sub = feature_extractor[5][0](features_before_sub)
    
    scores = BoW_predictor[0](features_before_sub)
    
    features_after_sub, value_sub, _,_ = valueLayer[0]([], scores, features_before_sub, [], [])
    
    features_before_quest = feature_extractor[5][1:](features_after_sub)
    
    scores_final_stage = BoW_predictor[1](features_before_quest)
    features_after_quest, value_quest = valueLayer[1](scores_final_stage, features_before_quest)
        
    
    features_before_sub = features_before_sub.permute(0, 2, 3, 1).contiguous()
    scores = scores.permute(0, 2, 3, 1).contiguous()
    value_sub = value_sub.permute(0, 2, 3, 1).contiguous()    
    features_after_sub = features_after_sub.permute(0, 2, 3, 1).contiguous()
    
    features_before_quest = features_before_quest.permute(0, 2, 3, 1).contiguous()
    value_quest = value_quest.permute(0, 2, 3, 1).contiguous()    
    features_after_quest = features_after_quest.permute(0, 2, 3, 1).contiguous()
    
    
    features_before_sub = features_before_sub.view(-1, features_before_sub.size(3))
    scores = scores.view(-1, scores.size(3))
    value_sub = value_sub.view(-1, value_sub.size(3))
    features_after_sub = features_after_sub.view(-1, features_after_sub.size(3))
    
    features_before_quest = features_before_quest.view(-1, features_before_quest.size(3))
    value_quest = value_quest.view(-1, value_quest.size(3))
    features_after_quest = features_after_quest.view(-1, features_after_quest.size(3))
    
    features_before_sub = features_before_sub.cpu().detach().numpy()
    scores = scores.cpu().detach().numpy()
    value_sub = value_sub.cpu().detach().numpy()
    features_after_sub = features_after_sub.cpu().detach().numpy()
    
    features_before_quest = features_before_quest.cpu().detach().numpy()
    value_quest = value_quest.cpu().detach().numpy()
    features_after_quest = features_after_quest.cpu().detach().numpy()
    
    target = target.cpu().detach().numpy()

    return features_before_sub, scores, value_sub, features_after_sub, features_before_quest, value_quest, features_after_quest, target
    
def visualizeEmbeddingPointsSuperClass_CIFAR100(features, all_labels, file_dir, name):
    
    num_subclasses = 8
    
    selected_superclasses = np.array([1,6,11,14,35,54,70,73])
    num_superclasses = len(selected_superclasses)
    
    
    selected_subclasses = selected_superclasses*num_subclasses
    selected_subclasses = np.repeat(selected_subclasses, num_subclasses, axis=0)
    
    dumy_sub = np.array(list(np.arange(num_subclasses))*num_superclasses)
    
    selected_subclasses = selected_subclasses + dumy_sub
    
    tsne = TSNE(n_components=2).fit_transform(features)
    # extract x and y coordinates representing the positions of the images on T-SNE plot
    features = features[selected_subclasses]
    
    
    tx = tsne[:, 0]
    ty = tsne[:, 1]
    
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)
    
    tx = tx[selected_subclasses]
    ty = ty[selected_subclasses]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.gca().set_axis_off()
    
    all_labels = all_labels[selected_subclasses]
    
    unique_labels = np.unique(all_labels)
    num_classes = 0

    # for every class, we'll add a scatter plot separately
    for label in unique_labels:
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(all_labels) if l == label]
        color = np.array(colors_per_class[num_classes], dtype=np.float) / 255
        num_classes +=1
        
        for s_cl in range(num_subclasses):
            
            marker = markers[s_cl]
    
            # extract the coordinates of the points of this class only
            current_tx = np.take(tx, indices[s_cl])
            current_ty = np.take(ty, indices[s_cl])
            
            ax.scatter(current_tx, current_ty, c=color.reshape(1,-1), s = 12, label=label, marker=marker)
        
          
            # add a scatter plot with the corresponding color and label
            
    
    # build a legend using the labels we set previously
    # ax.legend(loc='best')
    
    # finally, show the plot
    # plt.show()
    
    filename = os.path.join(file_dir, '%s.png' % name)
    
    fig.savefig(filename, dpi=300)
    
def visualizeEmbeddingPointsSuperClass(features, all_labels, file_dir, name):
    
    num_subclasses = 8    

    tsne = TSNE(n_components=2).fit_transform(features)
    # extract x and y coordinates representing the positions of the images on T-SNE plot
        
    tx = tsne[:, 0]
    ty = tsne[:, 1]
    
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.gca().set_axis_off()
    
    
    unique_labels = np.unique(all_labels)
    num_classes = 0

    # for every class, we'll add a scatter plot separately
    for label in unique_labels:
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(all_labels) if l == label]
        color = np.array(colors_per_class[num_classes], dtype=np.float) / 255
        num_classes +=1
        
        for s_cl in range(num_subclasses):
            
            marker = markers[s_cl]
    
            # extract the coordinates of the points of this class only
            current_tx = np.take(tx, indices[s_cl])
            current_ty = np.take(ty, indices[s_cl])
            
            ax.scatter(current_tx, current_ty, c=color.reshape(1,-1), s = 12, label=label, marker=marker)
        
    
    filename = os.path.join(file_dir, '%s.png' % name)
    
    fig.savefig(filename, dpi=300)
    
    
def visualizeEmbeddingPoints_all(features, y_test_patches, file_dir, name):
    
    tsne = TSNE(n_components=2).fit_transform(features)
    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]
    
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.gca().set_axis_off()
    
    # for every class, we'll add a scatter plot separately
    for label in colors_per_class:
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(y_test_patches) if l == label]
    
        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)
    
        # convert the class color to matplotlib format
        color = np.array(colors_per_class[label], dtype=np.float) / 255
    
        # add a scatter plot with the corresponding color and label
        ax.scatter(current_tx, current_ty, c=color.reshape(1,-1), s = 12, label=label, alpha=0.5, edgecolors =color.reshape(1,-1))
    
    # build a legend using the labels we set previously
    # ax.legend(loc='best')
    
    # finally, show the plot
    # plt.show()
    
    filename = os.path.join(file_dir, '%s.png' % name)
    
    fig.savefig(filename, dpi=300)