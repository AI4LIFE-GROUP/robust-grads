import numpy as np
import pandas as pd
import torch 
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

import data_utils

class PerturbParams():
    '''
    Stores data necessary to randomly perturb a dataset subject to constraints on what
    samples can change.

    strategy - current random or targeted
    threshold - fraction of eligible samples to modify
    
    target_indices and target_vals store the criteria for changing a data point; i.e., a sample
        x is only eligible to be modified if x[target_indices] == target_vals
    
    indices_to_change and new_vals represent how to modify (a subset of) samples that are eligible,
        i.e., we set x[indices_to_change] := new_vals
    '''
    def __init__(self, strategy, threshold = 0.1, target_indices = [], 
                 target_vals = [], indices_to_change = [], new_vals = []):
        self.strategy = strategy
        self.threshold = threshold
        self.target_indices = target_indices
        self.target_vals = target_vals
        self.indices_to_change = indices_to_change
        self.new_vals = new_vals


class BinaryDataset(Dataset):
    def __init__(self, source_file, perturb_params, label_col='label', transform=None,
             target_transform=None, random_state=-1):
        data = pd.read_csv(source_file)
        self.labels = pd.DataFrame(data[label_col])
        self.data = data.drop(columns=[label_col])
        if random_state >= 0:
            #self.data = data_utils.perturb_X_discrete(self.data, perturb_params, random_state)
            #print(random_state, perturb_params.threshold)
            self.data = data_utils.perturb_X(self.data, random_state, perturb_params.threshold)
            
        self.transform = transform
        self.target_transform = target_transform
        if transform is not None:
            self.data = self.transform.transform(self.data)
        else:
            self.data = np.array(self.data)
        if target_transform is not None:
            self.labels = self.target_transform(self.labels)
        
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = torch.tensor(self.data[idx])
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
    def __init__(self, source_file, perturb_params, label_col='label', transform=None,
             target_transform=None, random_state=-1):
        data = pd.read_csv(source_file)
        self.labels = pd.DataFrame(data[label_col])
        self.data = data.drop(columns=[label_col])
        self.data = np.asarray(self.data)
        self.transform = transform
        self.target_transform = target_transform
        if transform is not None:
            self.data = self.transform.transform(self.data)
        if target_transform is not None:
            self.labels = self.target_transform.transform(self.labels)

        if random_state >= 0:
            self.data = data_utils.perturb_X(self.data, random_state, perturb_params.threshold)

        
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

def load_mnist_data(random_state, perturb_params):
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
    if random_state >= 0:
        train.data = data_utils.perturb_X(train.data, random_state, perturb_params.threshold, 0, 255)
        test.data = data_utils.perturb_X(test.data, 0, perturb_params.threshold, 0, 255)
    return train, test

def load_data(file_base, dataset, scaler, scaler_labels, random_state, params, noisy_test=False):
    r = random_state if noisy_test else -1
    if (dataset in ['compas', 'income', 'german', 'german_cor', 'heloc', 'adult', 'gmsc']) or ('whobin' in dataset):
        train = BinaryDataset(file_base + '_train.csv', params, transform=scaler, random_state=random_state)
        test = BinaryDataset(file_base + '_test.csv', params, transform=scaler, random_state=r)
    elif dataset in ['who']:
        train = WhoDataset(file_base + '_train.csv', params, transform=scaler, target_transform=scaler_labels, random_state=random_state)
        test = WhoDataset(file_base + '_test.csv', params, transform=scaler,  target_transform=scaler_labels, random_state=r)
    else:
        train, test = load_mnist_data(random_state, params)
    return train, test

def load_data_simple(file_base, dataset, scaler, random_state, perturb_params, label_col='label'):
    '''
    Use with ART when we want entire dataset as a tensor/np array, rather than a dataloader object
    '''
    if dataset == 'mnist':
        data_train, data_test = load_mnist_data(random_state, perturb_params)
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
    # return np.matrix(X_train), y_train, np.matrix(X_test), y_test
