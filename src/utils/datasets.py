import numpy as np
import pandas as pd
import torch 
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

import utils.data_utils as data_utils

class BinaryDataset(Dataset):
    def __init__(self, source_file, threshold, label_col='label', transform=None,
             target_transform=None, random_state=-1, feature_type='continuous', add_noise = False):
        data = pd.read_csv(source_file)
        self.labels = pd.DataFrame(data[label_col])
        self.data = data.drop(columns=[label_col])
        self.data = np.asarray(self.data)
        self.feature_type = feature_type
        self.transform = transform
        self.target_transform = target_transform
        if transform is not None:
            self.data = self.transform.transform(self.data)
        else:
            self.data = np.array(self.data)
        if target_transform is not None:
            self.labels = self.target_transform(self.labels)

        if add_noise:
            if self.feature_type == 'continuous':
                self.data = data_utils.perturb_X(self.data, random_state, threshold, min = 0, add_noise = add_noise)
            elif self.feature_type == 'mixed':
                self.data = data_utils.perturb_X_mixed(self.data, threshold, random_state, min = 0, add_noise = add_noise)
            else:
                self.data = data_utils.perturb_X_discrete(self.data, threshold, random_state, add_noise = add_noise)
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if type(self.data) == type(np.ones(10)):
            data = torch.tensor(self.data[idx])
        else:
            data = self.data[idx].clone().detach()
        try:
            label = self.labels.iloc[idx, 0]
        except:
            label = self.labels[idx]
        return data, label

    def num_features(self):
        return self.data.shape[1]

    def num_classes(self):
        try:
            num = len(self.labels.label.unique())
        except:
            num = len(self.labels.unique())
        return num

class WhoDataset(Dataset):
    def __init__(self, source_file, threshold = 0, label_col='label', transform=None,
             target_transform=None, random_state=-1, feature_type='continuous', add_noise = False):
        data = pd.read_csv(source_file)
        self.labels = pd.DataFrame(data[label_col])
        self.data = data.drop(columns=[label_col])
        self.data = np.asarray(self.data)
        self.transform = transform
        self.target_transform = target_transform
        self.feature_type = feature_type
        if transform is not None:
            self.data = self.transform.transform(self.data)
        if target_transform is not None:
            self.labels = self.target_transform.transform(self.labels)

        if add_noise:
            if self.feature_type == 'continuous':
                self.data = data_utils.perturb_X(self.data, random_state, threshold, add_noise=add_noise)
            else:
                self.data = data_utils.perturb_X_discrete(self.data, threshold, random_state, add_noise = add_noise)

        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.data[idx]
        try:
            label = self.labels.iloc[idx, 0]
        except:
            label = self.labels[idx]
        return data, label

    def num_features(self):
        return self.data.shape[1]

    def num_classes(self):
        return 1 

def load_mnist_data(random_state, threshold, add_noise = False):
    train = datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor()
        )
    test = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    idx = (train.targets == 1) | (train.targets == 7)
    train.targets = train.targets[idx]
    train.data = train.data[idx]
    train.targets[train.targets == 7] = 0

    idx = (test.targets == 1) | (test.targets == 7)
    test.targets = test.targets[idx]
    test.data = test.data[idx]
    test.targets[test.targets == 7] = 0
    if add_noise:
        train.data = data_utils.perturb_X(train.data, random_state, threshold, 0, 255, add_noise=add_noise)
        test.data = data_utils.perturb_X(test.data, 0, threshold, 0, 255, add_noise = add_noise)
    return train, test

def load_data(file_base, dataset, random_state, threshold, add_noise = False, label_col="label"):
    '''
    TODO
    remove unused datasets?
    '''
    scaler, scaler_labels = None, None
    if 'whobin' in dataset:
        scaler = data_utils.get_scaler(pd.read_csv(file_base + '_train.csv').drop(
                columns=[label_col]), threshold, random_state=random_state, add_noise=add_noise)
        scaler_labels = None
        train = BinaryDataset(file_base + '_train.csv', threshold, transform=scaler, random_state=random_state, add_noise=add_noise, label_col= label_col)
        test = BinaryDataset(file_base + '_test.csv', threshold, transform=scaler, random_state=-1, add_noise=add_noise, label_col = label_col)
    elif dataset in ['compas', 'income', 'german', 'german_cor', 'student','heloc', 'adult']:
        train = BinaryDataset(file_base + '_train.csv', threshold, transform=scaler, random_state=random_state, feature_type='discrete', add_noise=add_noise, label_col= label_col)
        test = BinaryDataset(file_base + '_test.csv', threshold, transform=scaler, random_state=-1, feature_type='discrete', add_noise=add_noise, label_col = label_col)
    elif dataset == 'who':
        train = WhoDataset(file_base + '_train.csv', threshold, transform=scaler, target_transform=scaler_labels, random_state=random_state, add_noise=add_noise, label_col= label_col)
        test = WhoDataset(file_base + '_test.csv', threshold, transform=scaler,  target_transform=scaler_labels, random_state=-1, add_noise=add_noise, label_col = label_col)
    else:
        train, test = load_mnist_data(random_state, threshold, add_noise=add_noise)
    return train, test

def load_data_simple(file_base, dataset, scaler, random_state, threshold, label_col='label'):
    '''
    Use with ART (adversarial training) when we want entire dataset as a tensor/np array, 
    rather than a dataloader object
    '''
    if dataset == 'mnist':
        data_train, data_test = load_mnist_data(random_state, threshold)
        X_train = data_train.data
        X_test = data_test.data
        y_train = data_train.targets
        y_test = data_test.targets
    else: 
        data_train = pd.read_csv(file_base + "_train.csv")
        y_train = np.array(data_train[label_col])
        X_train = data_train.drop(columns=[label_col])

        data_test = pd.read_csv(file_base + "_test.csv")
        y_test = np.array(data_test[label_col])
        X_test = data_test.drop(columns=[label_col]) 

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)
    return X_train, np.array(y_train), X_test, np.array(y_test)

def load_secondary_dataset(dataset, file_base, r, threshold, add_noise=False, label_col="label"):
    if 'orig' in dataset:
        sec_name = dataset.replace('orig', 'shift')
        file_base = file_base.replace('orig', 'shift')
        _, secondary_dataset = load_data(file_base, sec_name, r, threshold, add_noise=add_noise)
    else:
        sec_name = dataset.replace('shift', 'orig')
        file_base = file_base.replace('shift', 'orig')
        _, secondary_dataset = load_data(file_base, sec_name, r, threshold, add_noise=add_noise)
    return secondary_dataset