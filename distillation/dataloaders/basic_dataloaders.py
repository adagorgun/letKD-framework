from __future__ import print_function

import numpy as np
import torch.utils.data

from torch.utils.data.sampler import BatchSampler
from torch.utils.data import DataLoader

def generate_element_list(list_size, dataset_size):
    if list_size == dataset_size:
        return list(range(dataset_size))
    elif list_size < dataset_size:
        return np.random.choice(
            dataset_size, list_size, replace=False).tolist()
    else: # list_size > list_size
        num_times = list_size // dataset_size
        residual = list_size % dataset_size
        assert((num_times * dataset_size + residual) == list_size)
        elem_list = list(range(dataset_size)) * num_times
        if residual:
            elem_list += np.random.choice(
                dataset_size, residual, replace=False).tolist()

        return elem_list


class SimpleDataloader:
    def __init__(
        self, dataset, batch_size, train, num_workers=4, epoch_size=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_size = len(dataset)
        self.epoch_size = epoch_size if epoch_size else self.dataset_size
        self.train = train

    def get_iterator(self, epoch=0):
        if self.epoch_size != self.dataset_size:
            elem_list = generate_element_list(self.epoch_size, self.dataset_size)
            dataset = torch.utils.data.Subset(self.dataset, elem_list)
        else:
            dataset = self.dataset
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=self.train,
            num_workers=self.num_workers, drop_last=self.train)
        """
        def load_fun_(idx):
            return self.dataset[idx % len(self.dataset)]
        tnt_dataset = tnt.dataset.ListDataset(elem_list=elem_list, load=load_fun_)
        data_loader = tnt_dataset.parallel(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.train,
            drop_last=self.train)
        """
        return data_loader

    def __call__(self, epoch=0):
        return self.get_iterator()

    def __len__(self):
        return self.epoch_size // self.batch_size


class DataloaderSampler:
    def __init__(
        self,
        dataset,
        batch_size,
        train,
        num_workers=4):
        self.dataset = dataset
        self.sampler = dataset.sampler
        self.dataset_size = len(self.sampler)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train = train

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            self.batch_size,
            sampler=self.sampler,
            num_workers=self.num_workers)

    def get_iterator(self, epoch=0):
        return self.dataloader

    def __call__(self, epoch=0):
        return self.get_iterator()

    def __len__(self):
        return self.dataset_size // self.batch_size


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset, n_classes, n_samples):
        loader = DataLoader(dataset)
        self.labels_list = []
        for _, label in loader:
            self.labels_list.append(label)
        self.labels = torch.LongTensor(self.labels_list)
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size