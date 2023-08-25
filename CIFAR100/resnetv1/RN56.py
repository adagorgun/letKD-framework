config = {}
# set the parameters related to the training and testing set
data_train_opt = {}
data_train_opt['dataset_name'] = 'CIFAR100'
data_train_opt['dataset_args'] = {'split': 'train'}
data_train_opt['epoch_size'] = None
data_train_opt['batch_size'] = 64

data_test_opt = {}
data_test_opt['dataset_name'] = 'CIFAR100'
data_test_opt['dataset_args'] = {'split': 'val'}
data_test_opt['batch_size'] = 100

config['data_train_opt'] = data_train_opt
config['data_test_opt'] = data_test_opt

## Learning rate schedule
LUT_lr = [(150, 0.05), (180, 0.005), (210, 0.0005), (240, 0.00005)]
max_num_epochs = 240
config['max_num_epochs'] = max_num_epochs

num_classes = 100
depth = 56
num_filters = [16, 16, 32, 64]


pretrained = 'saved_models/resnet56_vanilla/ckpt_epoch_240.pth'

networks = {}

net_optim_paramsC = {
    'optim_type': 'sgd', 'lr': 0.1, 'momentum':0.9, 'weight_decay': 5e-4,
    'nesterov': True, 'LUT_lr':LUT_lr}

net_optim_paramsF = {
    'optim_type': 'sgd', 'lr': 0.1, 'momentum':0.9, 'weight_decay': 5e-4,
    'nesterov': True, 'LUT_lr':LUT_lr, 'classifier_optim':net_optim_paramsC}

net_optionsF = {'depth': depth, 'num_filters': num_filters,
                'num_classes':num_classes, 'extract_from':['layer3'], 'extract_after_relu':[False]}

networks['student_net'] = {'def_file': 'feature_extractors.resnet',
                           'pretrained': pretrained, 'opt': net_optionsF, 
                           'optim_params': None, 'force': True}


config['networks'] = networks

criterions = {}
criterions['loss'] = {'ctype':'CrossEntropyLoss', 'opt':None}
config['criterions'] = criterions

config['algorithm_type'] = 'classification'
