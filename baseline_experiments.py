import argparse
import numpy as np
import pandas as pd
from torch import nn

import data_utils
import datasets
import training

# LINEAR MODELS - see linear_model.py
# NEURAL ARCHITECTURE - see neural_net.py


def main(args):
    random_states = [x-1 for x in range(100)]
    random_states = [x-1 for x in range(5)]
    test_accuracy, train_accuracy = [], []
    perturb_params = datasets.PerturbParams(args.strategy, args.threshold, args.target_indices, args.target_vals,
                    args.indices_to_change, args.new_vals)

    for r in random_states:
        if args.fixed_seed:
            seed = 0
        else:
            seed = r
        if args.dataset in ['income', 'compas']:
            scaler, scaler_labels = data_utils.get_scaler_discrete(pd.read_csv(args.file_base + '_train.csv').drop(
                columns=[args.label_col]), perturb_params, random_state=r)
        elif args.dataset in ['who']:
            scaler = data_utils.get_scaler(pd.read_csv(args.file_base + '_train.csv').drop(
                columns=[args.label_col]), perturb_params, random_state=r)
            scaler_labels = data_utils.get_scaler(np.array(pd.read_csv(args.file_base + '_train.csv')[args.label_col]).reshape(-1, 1), 
                perturb_params, random_state = -1, min_val=0)
        elif args.dataset in ['jordan', 'kuwait']:
            scaler, scaler_labels = data_utils.get_scaler_mixed(pd.read_csv(args.file_base + '_train.csv').drop(
                columns=[args.label_col]), perturb_params, random_state=r, min_val=0)
        elif 'whobin' in args.dataset:
            scaler = data_utils.get_scaler(pd.read_csv(args.file_base + '_train.csv').drop(
                columns=[args.label_col]), perturb_params, random_state=r)
            scaler_labels = None
        else:  # if data is already scaled in a preprocessing step, as with german, skip scaling step 
            scaler, scaler_labels = None, None
        
        train, test = datasets.load_data(args.file_base, args.dataset, scaler, scaler_labels, r, perturb_params)
        secondary_dataset = None
        if args.dataset_shift and 'orig' in args.dataset:
            sec_name = args.dataset.replace('orig', 'shift')
            file_base = args.file_base.replace('orig', 'shift')
            _, secondary_dataset = datasets.load_data(file_base, sec_name, scaler, scaler_labels, r, perturb_params)
        elif args.dataset_shift:
            sec_name = args.dataset.replace('shift', 'orig')
            file_base = args.file_base.replace('shift', 'orig')
            _, secondary_dataset = datasets.load_data(file_base, sec_name, scaler, scaler_labels, r, perturb_params)

        if args.dataset in ['mnist']:
            #MNIST
            num_feat = 28*28
            num_classes = 2
        else:
            num_feat = train.num_features()
            num_classes = train.num_classes()

        params = training.Params(args.lr, args.lr_decay, args.epochs, args.batch_size, loss_fn=args.loss, num_feat=num_feat, 
                        num_classes=num_classes, activation=args.activation, nodes_per_layer=args.nodes_per_layer,
                        num_layers=args.num_layers, optimizer=args.optimizer, seed=seed, epsilon = args.epsilon, dropout= args.dropout)
        
        if args.linear:
            training.train_linear_models(args, train, test, r)
        elif args.adversarial:
            model, test_acc, train_acc, preds = training.train_adv_nn(params, train, test, r, args.dataset)
        else:
            model, test_acc, train_acc, preds = training.train_nn(params, train, test, r, args.dataset, secondary_dataset)
        test_accuracy.append(test_acc)
        train_accuracy.append(train_acc)

    if type(args.activation) == type(nn.ReLU()):
        act = "relu"
    elif type(args.activation) == type(nn.Softplus(beta=5)):
        act = "soft"
    else:
        act = "leak"
    
    np.save(args.output_dir + "/accuracy_" + args.dataset + "_nn" + str(args.num_layers) + "_"+ act +
                ".npy", np.matrix([test_accuracy, train_accuracy]))
    return 1

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('file_base', type=str)
    parser.add_argument('--dataset_shift', type=bool, default=False)
    parser.add_argument('--adversarial', type=bool, default=False)
    parser.add_argument('--output_dir', type=str, default='.')
    parser.add_argument('--label_col', default='label', type=str)

    parser.add_argument('--linear', type=bool, default=False)

    parser.add_argument('--lr', type=float, default=0.2)
    parser.add_argument('--lr_decay', type=float, default=0.8)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--nodes_per_layer', type=int, default=50)
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--optimizer', type=str, default=None)
    parser.add_argument('--loss', type=str, default=None)
    parser.add_argument('--fixed_seed', type=bool, default=False) # if true, use seed 0 for all random states

    parser.add_argument('--target_indices', type=str, default='') # if only changing a subset of data, what criteria to use
    parser.add_argument('--target_vals', type=str, default='')
    parser.add_argument('--indices_to_change', type=str, default='') # what indices to change?
    parser.add_argument('--new_vals', type=str, default='') # what can we change modified values to?
    parser.add_argument('--threshold', type=float, default = 0.0)
    parser.add_argument('--strategy', type=str, default='random')
    parser.add_argument('--epsilon', type=float, default=0.5) # epsilon for finding adv. examples
    parser.add_argument('--dropout', type=float, default=0.0) # dropout rate

    args = parser.parse_args()
    if args.target_indices == '':
        args.target_indices = []
        args.target_vals = []
    else:
        args.target_indices = list(map(int, args.target_indices.split("_")))
        args.target_vals = list(map(float, args.target_vals.split("_")))
    if args.indices_to_change == '':
        args.indices_to_change = []
        args.new_vals = []
    else:    
        args.indices_to_change = list(map(int, args.indices_to_change.split("_")))
        args.new_vals = list(map(float, args.new_vals.split("_")))

    assert len(args.target_indices) == len(args.target_vals), 'target_indices and target_vals must be the same length'
    assert len(args.indices_to_change) == len(args.new_vals), 'indices_to_change and new_vals must be the same length'
    strategies = ['random', 'targeted', 'untargeted', 'targeted-random', 'none']
    args.strategy = str.strip(args.strategy)
    assert args.strategy in strategies, 'strategy must be in [' + ' '.join(strategies) + '] but is ' + args.strategy

    if args.dataset in ['mnist']: 
        args.loss = nn.MSELoss() 
    else:
        args.loss = nn.CrossEntropyLoss()         

    if args.activation is not None:
        if args.activation == 'leak':
            args.activation = nn.LeakyReLU()
        elif args.activation == 'soft':
            args.activation = nn.Softplus(beta=5)
        else:
            args.activation = nn.ReLU()
    else:
        args.activation = nn.ReLU()

    # if we are testing dataset shift (rather than random perturbation), 
    # call the remaining code 2x, once on original data and once on shifted data
    if args.dataset_shift:
        args.dataset = args.dataset + '_orig'
        args.file_base = args.file_base + '_orig'
        main(args)
        args.dataset = args.dataset[:-5] + "_shift"
        args.file_base = args.file_base[:-5] + "_shift"
        main(args)
    else:
        main(args)
