import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def create_train_test_split(source_file, label_col='label', random_state=1129):
    ''' 
    Separate data into train and test sets based on the provided random state. 
    Save as csv files.
    '''
    data = pd.read_csv(source_file)
    y = data[label_col]
    X = data.drop(columns=[label_col])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    X_train[label_col] = y_train
    X_test[label_col] = y_test
    filename_root = source_file.split(".csv")[0]
    X_train.to_csv(filename_root + "_train.csv", index=False)
    X_test.to_csv(filename_root + "_test.csv", index=False)
    
# Perturb the dataset -- assumes X is a tensor
def perturb_X(X, random_state, std_dv, min = None, max = None, add_noise = False):
    if type(X) == type(pd.DataFrame()):
        X = torch.tensor(np.matrix(X))
    elif type(X) == type(np.ones(X.shape)):
        X = torch.tensor(X)
    if add_noise:
        torch.manual_seed(random_state)
        perturbation = torch.normal(torch.zeros(X.shape), std_dv * torch.ones_like(X))
        data_new = X + perturbation
        if (min is not None) or (max is not None):
            X = torch.clamp(data_new, min, max)
        
    return X

def satisfies(X, row, target_ind, target_val):
    for i in range(len(target_ind)):
        if X[row, target_ind[i]] != target_val[i]:
            return False
    return True

def perturb_X_mixed(X, params, random_state, min=None, max=None, add_noise = False):
    ''' 
    Use when some features of X are binary while others are continuous
    '''
    random.seed = random_state
    torch.manual_seed(random_state)
    X = np.matrix(X)

    if add_noise:
        if params.strategy == 'random':
            for col in range(X.shape[1]):
                
                unique =  np.unique(np.round(X[:,col],5).T.tolist()[0])
                if len(unique) == 2:
                    binary = True
                else:
                    binary = False
                for row in range(X.shape[0]):
                    if random.random() < params.threshold:
                        if binary:
                            if X[row,col] == 0:
                                X[row,col] = 1
                            else:
                                X[row,col] = 0
                        else:
                            perturbation = torch.normal(torch.zeros(1), params.threshold * torch.ones_like(torch.tensor(1)))
                            X[row,col] = X[row,col] + perturbation

                            if (min is not None) or (max is not None):
                                X[row,col] = torch.clamp(torch.tensor(X[row,col]), min, max)
        else:
            print("ERROR: perturbation strategies other than random are not implemented yet for mixed datasets ")
    return np.array(X)

def perturb_X_discrete(X, params, random_state, add_noise = False):
    # Perturb X when all features are binary (0/1)
    # threshold in [0,1] is the fraction of features we will change
    random.seed(random_state)
    X = np.matrix(X)
    if add_noise:
        if params.strategy == 'random':
            for row in range(X.shape[0]):
                for col in range(X.shape[1]):
                    if random.random() < params.threshold:
                        if X[row,col] == 0:
                            X[row,col] = 1
                        else:
                            X[row,col] = 0
        elif params.strategy == 'untargeted':
            indices_to_change, new_vals = params.indices_to_change, params.new_vals

            for row in range(X.shape[0]):
                if random.random() < params.threshold:
                    for i in range(len(indices_to_change)):
                        X[row,indices_to_change[i]] = new_vals[i]
        elif params.strategy == 'targeted-random':
            target_indices, target_vals = params.target_indices, params.target_vals

            for row in range(X.shape[0]):
                if satisfies(X, row, target_indices, target_vals):
                    for col in range(X.shape[1]):
                        if random.random() < params.threshold:
                            if X[row,col] == 0:
                                X[row,col] = 1
                            else:
                                X[row,col] = 0
        elif params.strategy == 'targeted':
            target_indices, target_vals = params.target_indices, params.target_vals
            indices_to_change, new_vals = params.indices_to_change, params.new_vals
            
            for row in range(X.shape[0]):
                if satisfies(X, row, target_indices, target_vals):
                    if random.random() < params.threshold:
                        for i in range(len(indices_to_change)):
                            X[row,indices_to_change[i]] = new_vals[i]
        else:
            print("ERROR: perturbation strategy must be one of 'random', 'targeted', or 'untargeted' ")
    return np.array(X)

def get_scaler(X, perturb_params, random_state=0, min_val = None, max_val = None, add_noise = False):
    ''' 
    Returns a StandardScaler object fitted to the perturbed version of the
    dataset X, where X's features are continuous.
    '''
    if type(X) == type(pd.DataFrame()):
        X = torch.tensor(np.matrix(X))
    if add_noise:
        X = perturb_X(X, random_state, perturb_params.threshold, min=min_val, max=max_val, add_noise = add_noise)
    scaler = StandardScaler().fit(X)

    return scaler

def get_scaler_discrete(X, perturb_params, random_state=0, add_noise=False):
    '''
    Returns a StandardScaler object fitted to the perturbed version of the 
    dataset X, where X's features are discrete.
    '''
    if type(X) == type(pd.DataFrame()):
        X = torch.tensor(np.matrix(X))
    if add_noise and not (perturb_params.strategy == 'none'):
        X = perturb_X_discrete(X, perturb_params, random_state, add_noise = add_noise)
    scaler = StandardScaler().fit(X)
    return scaler, None

def get_scaler_mixed(X, perturb_params, random_state = 0, min_val=None, max_val=None, add_noise = False):
    '''
    Returns a StandardScaler object fitted to the perturbed version of the 
    dataset X, where X's features are mixed continuous and discrete.
    '''
    if type(X) == type(pd.DataFrame()):
        X = torch.tensor(np.matrix(X))
    if add_noise and not (perturb_params.strategy == 'none'):
        X = perturb_X_mixed(X, perturb_params, random_state, min=min_val, max=max_val, add_noise=add_noise)
    scaler = StandardScaler().fit(X)
    return scaler, None

def transform_y(y):
    y = np.array(y)
    y_neg = np.ones(y.shape)-y
    df = pd.DataFrame([y, y_neg])
    y2 = df.T
    return y2