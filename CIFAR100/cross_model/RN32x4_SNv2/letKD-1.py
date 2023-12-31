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

LUT_lr = [(150, 0.01), (180, 0.001), (210, 0.0001), (240, 0.00001)]
config['max_num_epochs'] = 240


num_classes = 100
num_filters_teacher = [32, 64, 128, 256]
depth_teacher = 32
num_filters_student = None
num_embeddings = 4096

networks = {}
pretrained_teacher = "./saved_models/resnet32x4_vanilla/ckpt_epoch_240.pth"

net_optionsF = {'depth': depth_teacher, 'num_filters': num_filters_teacher,
    'num_classes':num_classes, 'extract_from':['avgpool_2x2'], 'extract_after_relu':[False], 'downscale': True} ##  no relu after layer3 so, extract_after_relu should always be False

networks['teacher_net'] = {
    'def_file': 'feature_extractors.resnet',
    'pretrained': pretrained_teacher, 'opt': net_optionsF,
    'optim_params': None, 'force':True}

num_epochs='*'
pretrained_VQ = f'./experiments/VQ/CIFAR100/resnetv1/RN32x4_downscale/vector_quantizer_kmeansK{num_embeddings}_net_epoch{num_epochs}'

net_optionsVQ = {
    'num_embeddings': num_embeddings,
    'embedding_dim': num_filters_teacher[-1],
    'commitment_cost': 0.25,
    'decay': 0.99,
    'epsilon': 1e-5,
    'temperature': 1}

networks['vector_quantizer_target'] = {
    'def_file': 'miscellaneous.vector_quantization',
    'pretrained': pretrained_VQ,
    'opt': net_optionsVQ,
    'optim_params': None}

net_optim_paramsC = {
    'optim_type': 'sgd', 'lr': 0.1, 'momentum':0.9, 'weight_decay': 5e-4,
    'nesterov': True, 'LUT_lr':LUT_lr}

net_optionsC = {
    'classifier_type': 'linear', 'num_classes': num_classes,
    'num_channels': num_filters_teacher[3], 'global_pooling': True}

net_optim_paramsF = {
    'optim_type': 'sgd', 'lr': 0.1, 'momentum':0.9, 'weight_decay': 5e-4,
    'nesterov': True, 'LUT_lr':LUT_lr, 'classifier_optim':net_optim_paramsC}

net_optionsF = {'num_classes':num_classes, 'extract_from':['relu2'], 'extract_after_relu':[False],
                'classifier_options':net_optionsC}

networks['student_net'] = {'def_file': 'feature_extractors.shufflev2',
    'pretrained': None, 'opt': net_optionsF, 'optim_params': net_optim_paramsF}

net_optim_paramsCP = {
    'optim_type': 'sgd', 'lr': 0.1, 'momentum':0.9, 'weight_decay': 5e-4,
    'nesterov': True, 'LUT_lr':LUT_lr}

net_optionsCP = {
    'classifier_type': 'conv_cosine',
    'num_classes': num_embeddings,
    'num_channels': 1024,
    'scale_cls': 3.0,
    'learn_scale': True}

networks['BoW_predictor'] = {
    'def_file': 'classifiers.classifier',
    'pretrained': None, 'opt': net_optionsCP, 'optim_params': net_optim_paramsCP}

networks['valueLayer'] = {
    'def_file': 'miscellaneous.valueLayer',
    'pretrained': None, 'opt': {'emb_size':num_embeddings, 'inp_size':1024},
    'optim_params': net_optim_paramsCP}

config['networks'] = networks

criterions = {}
criterions['loss'] = {'ctype':'CrossEntropyLoss', 'opt':None}
config['criterions'] = criterions

config['algorithm_type'] = 'quest_value'
config['quest_loss_coef'] = 1.0
