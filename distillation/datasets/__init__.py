from distillation.datasets.imagenet_dataset import ImageNet
from distillation.datasets.tiny_imagenet_dataset import TinyImageNet
from distillation.datasets.cifar_dataset import CIFAR100
from distillation.datasets.cifar_dataset import CIFAR10
from distillation.datasets.mit67_datasets import MITScenes

def dataset_factory(dataset_name, *args, **kwargs):
    datasets_collection = {}
    datasets_collection['ImageNet'] = ImageNet
    datasets_collection['TinyImageNet'] = TinyImageNet
    datasets_collection['CIFAR100'] = CIFAR100
    datasets_collection['CIFAR10'] = CIFAR10
    datasets_collection['MITScenes'] = MITScenes
    return datasets_collection[dataset_name](*args, **kwargs)
