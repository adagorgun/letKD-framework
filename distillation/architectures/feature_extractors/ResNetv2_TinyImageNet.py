import torchvision.models as models
import distillation.architectures.tools as tools
from collections import OrderedDict
import torch.nn as nn




def create_model(opt):


    model = models.resnet18(pretrained=True)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    # model.fc.out_features = 200
    
    key_names = list(model._modules.keys())
    od = OrderedDict()
    for k in key_names[:-1]:
        od[k] = model._modules[k]
    od['reshape'] = tools.Reshape(-1, 512)
    od[key_names[-1]] = nn.Linear(512 , 200)
    new_model = nn.Sequential(od)


    return new_model
