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
    if args.fixed_seed == False:
        random_states = [x for x in range(args.variations + args.base_repeats)]
    elif args.dataset_shift == True:
        random_states = [x for x in range(args.variations)] # no ``base models'', exactly
    else:
        random_states = [x for x in range((args.variations + 1)*args.base_repeats)]

    test_accuracy, train_accuracy, secondary_accuracy = [], [], []
    perturb_params = datasets.PerturbParams(args.strategy, args.threshold, args.target_indices, args.target_vals,
                    args.indices_to_change, args.new_vals)

    base_iter = 0
    finetune = args.finetune
    for r in random_states:
        if (not args.dataset_shift) and (args.fixed_seed == False) and (r < args.base_repeats):
            add_noise = False
            baseline_model = True
        elif (r % (args.variations + 1) == 0):
            baseline_model = True
            add_noise = False
        else: 
            if args.dataset_shift:
                add_noise = False
            else:
                add_noise = True
            baseline_model = False
        if (args.fixed_seed) and not args.dataset_shift:
            seed = (r) // (args.variations + 1)
        else:
            seed = r

        if args.dataset in ['income', 'compas']:
            scaler, scaler_labels = data_utils.get_scaler_discrete(pd.read_csv(args.file_base + '_train.csv').drop(
                columns=[args.label_col]), perturb_params, random_state=r)
        elif args.dataset in ['who']:
            scaler = data_utils.get_scaler(pd.read_csv(args.file_base + '_train.csv').drop(
                columns=[args.label_col]), perturb_params, random_state=r, add_noise=add_noise)
            scaler_labels = data_utils.get_scaler(np.array(pd.read_csv(args.file_base + '_train.csv')[args.label_col]).reshape(-1, 1), 
                perturb_params, random_state = -1, min_val=0)
        elif args.dataset in ['jordan', 'kuwait']:
            scaler, scaler_labels = data_utils.get_scaler_mixed(pd.read_csv(args.file_base + '_train.csv').drop(
                columns=[args.label_col]), perturb_params, random_state=r, min_val=0)
        elif 'whobin' in args.dataset:
            scaler = data_utils.get_scaler(pd.read_csv(args.file_base + '_train.csv').drop(
                columns=[args.label_col]), perturb_params, random_state=r, add_noise=add_noise)
            scaler_labels = None
        else:  # if data is already scaled in a preprocessing step, as with german, skip scaling step 
            scaler, scaler_labels = None, None
        train, test = datasets.load_data(args.file_base, args.dataset, scaler, scaler_labels, r, perturb_params, add_noise=add_noise)
        secondary_dataset = None
        if args.dataset_shift and 'orig' in args.dataset:
            sec_name = args.dataset.replace('orig', 'shift')
            file_base = args.file_base.replace('orig', 'shift')
            _, secondary_dataset = datasets.load_data(file_base, sec_name, scaler, scaler_labels, r, perturb_params, add_noise=add_noise)
        elif args.dataset_shift:
            sec_name = args.dataset.replace('shift', 'orig')
            file_base = args.file_base.replace('shift', 'orig')
            _, secondary_dataset = datasets.load_data(file_base, sec_name, scaler, scaler_labels, r, perturb_params, add_noise=add_noise)

        if args.dataset in ['mnist']:
            #MNIST
            num_feat = 28*28
            num_classes = 2
        else:
            num_feat = train.num_features()
            num_classes = train.num_classes()

        params = training.Params(args.lr, args.lr_decay, args.epochs, args.batch_size, loss_fn=args.loss, num_feat=num_feat, 
                        num_classes=num_classes, activation=args.activation, nodes_per_layer=args.nodes_per_layer,
                        num_layers=args.num_layers, optimizer=args.optimizer, seed=seed, epsilon = args.epsilon, dropout= args.dropout,
                        weight_decay = args.weight_decay)
        
        if args.linear:
            training.train_linear_models(args, train, test, r)
        elif args.adversarial:
            model, test_acc, train_acc, sec_acc, test_loss, train_loss = training.train_adv_nn(params, train, test, r, args.dataset, args.output_dir, args.run_id, secondary_dataset, (finetune and (not baseline_model)), (finetune and baseline_model))
        else:
            model, test_acc, train_acc, sec_acc, test_loss, train_loss = training.train_nn(params, train, test, r, args.dataset, args.output_dir, args.run_id, secondary_dataset, (finetune and (not baseline_model)), (finetune and baseline_model))
        test_accuracy.append(test_acc)
        train_accuracy.append(train_acc)
        if sec_acc is not None:
            secondary_accuracy.append(sec_acc)

    trainacc = [round(train_accuracy[i][-1]*100, 2) for i in range(len(train_accuracy))]
    print("Train acc: ", trainacc)
    print("avg: ", sum(trainacc)/len(trainacc), " min: ", min(trainacc))
    testacc = [round(test_accuracy[i][-1]*100, 2) for i in range(len(test_accuracy))]
    print("Test acc: ", testacc)
    print("avg: ", sum(testacc)/len(testacc), " min: ", min(testacc))

    if type(args.activation) == type(nn.ReLU()):
        act = "relu"
    elif type(args.activation) == type(nn.Softplus(beta=5)):
        act = "soft"
    
    np.save(args.output_dir + "/accuracy_train_" + args.run_id + ".npy", train_accuracy)
    np.save(args.output_dir + "/accuracy_test_" + args.run_id + ".npy", test_accuracy)
    np.save(args.output_dir + "/loss_train_" + args.run_id + ".npy", train_loss)
    np.save(args.output_dir + "/loss_test_" + args.run_id + ".npy", test_loss)

    params = [args.dataset, args.threshold, args.adversarial, args.dataset_shift, args.fixed_seed, 
                args.base_repeats, args.variations, act, args.lr, args.lr_decay, args.weight_decay, 
                args.epochs, args.nodes_per_layer, args.num_layers, args.epsilon, args.beta, args.finetune]
    np.save(args.output_dir + "/params_" + args.run_id + ".npy", params)
    if args.dataset_shift:
        if 'orig' in args.dataset:
            np.save(args.output_dir + "/accuracy_test_orig_" + args.run_id + ".npy", secondary_accuracy)
        else:
            np.save(args.output_dir + "/accuracy_test_shift_" + args.run_id + ".npy", secondary_accuracy)
    return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('file_base', type=str, help='file path of dataset through _train or _test')
    parser.add_argument('run_id', type=str)
    parser.add_argument('--base_repeats', type=int, default=5) # how many base models do we need to compare with (and average over?)
    parser.add_argument('--variations', type=int, default=100) # how many models to compare, total?
    parser.add_argument('--dataset_shift', type=bool, default=False)
    parser.add_argument('--adversarial', type=bool, default=False)
    parser.add_argument('--output_dir', type=str, default='.')
    parser.add_argument('--label_col', default='label', type=str)

    parser.add_argument('--linear', type=bool, default=False)

    parser.add_argument('--lr', type=float, default=0.2)
    parser.add_argument('--lr_decay', type=float, default=0.8)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--nodes_per_layer', type=int, default=50)
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--optimizer', type=str, default=None)
    parser.add_argument('--loss', type=str, default=None)
    parser.add_argument('--fixed_seed', type=bool, default=False) # if true, use seed 0 for all random states
    parser.add_argument('--finetune', type=bool, default = False)

    parser.add_argument('--target_indices', type=str, default='') # if only changing a subset of data, what criteria to use
    parser.add_argument('--target_vals', type=str, default='')
    parser.add_argument('--indices_to_change', type=str, default='') # what indices to change?
    parser.add_argument('--new_vals', type=str, default='') # what can we change modified values to?
    parser.add_argument('--threshold', type=float, default = 0.0)
    parser.add_argument('--strategy', type=str, default='random')
    parser.add_argument('--epsilon', type=float, default=0.5) # epsilon for finding adv. examples
    parser.add_argument('--dropout', type=float, default=0.0) # dropout rate
    parser.add_argument('--beta', type=float, default=5) # beta for softplus

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
            args.activation = nn.Softplus(beta=args.beta)
        else:
            args.activation = nn.ReLU()
    else:
        args.activation = nn.ReLU()

    args.orig_dataset_shift = False
    args.shifted_dataset_shift = False
    # if we are testing dataset shift (rather than random perturbation), 
    # call the remaining code 2x, once on original data and once on shifted data
    if args.dataset_shift:
        args.orig_dataset_shift = True
        args.dataset = args.dataset + '_orig'
        args.run_id = args.run_id + '_orig'
        args.file_base = args.file_base + '_orig'
        main(args)
        args.orig_dataset_shift = False
        args.shifted_dataset_shift = True
        args.dataset = args.dataset[:-5] + "_shift"
        args.file_base = args.file_base[:-5] + "_shift"
        args.run_id = args.run_id[:-5] + "_shift"
        main(args)
    else:
        main(args)
