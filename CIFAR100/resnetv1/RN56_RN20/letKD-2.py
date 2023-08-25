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
config['max_num_epochs'] = 240

num_classes = 100
num_filters_teacher = [16, 16, 32, 64]
depth_teacher = 56
num_filters_student = [16, 16, 32, 64]
depth_student = 20

num_embeddings = 4096
num_subclasses = 8*num_classes
num_epochs='*'


networks = {}
pretrained_teacher = './saved_models/resnet56_vanilla/ckpt_epoch_240.pth'


net_optionsF = {
    'depth': depth_teacher, 'num_filters': num_filters_teacher,
    'num_classes':num_classes, 'extract_from':['layer3'], 'extract_after_relu':[False]}

networks['teacher_net'] = {
    'def_file': 'feature_extractors.resnet',
    'pretrained': pretrained_teacher, 'opt': net_optionsF,
    'optim_params': None, 'force':True}


centers_filename =  f'./experiments/LDA/CIFAR100/resnetv1/RN56/centers_subclass_{int(num_subclasses/num_classes)}.npy'
lda_filename = './experiments/LDA/CIFAR100/resnetv1/RN56/lda_model'
scores_filename = f'./experiments/LDA/CIFAR100/resnetv1/RN56/final_scores_{int(num_subclasses/num_classes)}.npy'


pretrained_VQ = f'./experiments/VQ/CIFAR100/resnetv1/RN56/vector_quantizer_kmeansK{num_embeddings}_net_epoch{num_epochs}'

net_optionsVQ = {
    'num_embeddings': num_embeddings,
    'embedding_dim': num_filters_teacher[3],
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

net_optionsF = {'depth': depth_student, 'num_filters': num_filters_student,
                'num_classes':num_classes, 'extract_from':['layer3'], 'extract_after_relu':[False],
                'classifier_options':net_optionsC}

networks['student_net'] = {'def_file': 'feature_extractors.resnet',
    'pretrained': None, 'opt': net_optionsF, 'optim_params': net_optim_paramsF}

net_optim_paramsCP = {
    'optim_type': 'sgd', 'lr': 0.1, 'momentum':0.9, 'weight_decay': 5e-4,
    'nesterov': True, 'LUT_lr':LUT_lr}

net_optionsCP = {
    'classifier_type': 'conv_cosine',
    'num_classes': num_subclasses,
    'num_channels': num_filters_student[3],
    'scale_cls': 3.0,
    'learn_scale': True}

networks['BoW_predictor_0'] = {
    'def_file': 'classifiers.classifier',
    'pretrained': None, 'opt': net_optionsCP, 'optim_params': net_optim_paramsCP}

net_optionsCP = {
    'classifier_type': 'conv_cosine',
    'num_classes': num_embeddings,
    'num_channels': num_filters_student[3],
    'scale_cls': 3.0,
    'learn_scale': True}

networks['BoW_predictor_1'] = {
    'def_file': 'classifiers.classifier',
    'pretrained': None, 'opt': net_optionsCP, 'optim_params': net_optim_paramsCP}


networks['valueLayer_0'] = {
    'def_file': 'miscellaneous.valueLayer_subclass',
    'pretrained': None, 'opt': {'emb_size':num_subclasses, 'inp_size':num_filters_student[3], 'centers_filename':centers_filename,
                                'lda_filename':lda_filename, 'scores_filename': scores_filename, 'num_classes':num_classes},
    'optim_params': net_optim_paramsCP}

networks['valueLayer_1'] = {
    'def_file': 'miscellaneous.valueLayer',
    'pretrained': None, 'opt': {'emb_size':num_embeddings, 'inp_size':num_filters_student[3]},
    'optim_params': net_optim_paramsCP}

config['networks'] = networks

criterions = {}
criterions['loss'] = {'ctype':'CrossEntropyLoss', 'opt':None}
config['criterions'] = criterions

config['algorithm_type'] = 'subclass_quest_value'
config['sub_loss_coef'] = 1.0
config['quest_loss_coef'] = 1.0
config['which_stage_t'] = 5
config['which_stage_s'] = 5


