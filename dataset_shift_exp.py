import argparse
import numpy as np
import pandas as pd
from torch import nn

import data_utils
import datasets
import training

'''
    Run fine-tuning experiments on a real-world dataset shift.

    To run retraining experiments, use baseline_experiments.py instead.
    To run experiments with synthetic noise, use baseline_experiments.py instead.
'''

def main(args):

    random_states = [x for x in range(args.variations)] # no ``base models'', exactly

    test_accuracy, train_accuracy, secondary_accuracy = [], [], []
    test_acc_shift, train_acc_shift, sec_acc_shift = [], [], []
    all_train_loss, all_test_loss, all_train_shift_loss, all_test_shift_loss = [], [], [], []


    finetune = args.finetune
    orig_run_id = args.run_id

    for r in random_states:
        args.run_id = orig_run_id

        seed = r
        print("Working with r=",str(r))

        scaler = data_utils.get_scaler(pd.read_csv(args.file_base + '_train.csv').drop(
                columns=[args.label_col]), args.threshold, random_state=r, add_noise=False)
        scaler_labels = None
        train, test = datasets.load_data(args.file_base, args.dataset, scaler, scaler_labels, r, args.threshold, add_noise=False)

        sec_name = args.dataset.replace('orig', 'shift')
        file_base = args.file_base.replace('orig', 'shift')
        shifted_train, shifted_test = datasets.load_data(file_base, sec_name, scaler, scaler_labels, r, args.threshold, add_noise=False)

        num_feat = train.num_features()
        num_classes = train.num_classes()

        params = training.Params(args.lr, args.lr_decay, args.epochs, args.batch_size, loss_fn=args.loss, num_feat=num_feat, 
                        num_classes=num_classes, activation=args.activation, nodes_per_layer=args.nodes_per_layer,
                        num_layers=args.num_layers, optimizer=args.optimizer, seed=seed, epsilon = args.epsilon, dropout= args.dropout,
                        weight_decay = args.weight_decay)
        
        if args.adversarial:
            _, test_acc, train_acc, sec_acc, test_loss, train_loss = training.train_adv_nn(params, train, test, r, args.dataset, args.output_dir, args.run_id, shifted_test, finetune_base=finetune)
        else:
            _, test_acc, train_acc, sec_acc, test_loss, train_loss = training.train_nn(params, train, test, r, args.dataset, args.output_dir, args.run_id, shifted_test, finetune_base=finetune)
        test_accuracy.append(test_acc)
        train_accuracy.append(train_acc)
        secondary_accuracy.append(sec_acc)
        all_train_loss.append(train_loss)
        all_test_loss.append(test_loss)
        

        params.epochs = args.finetune_epochs
        params.learning_rate = args.lr * (0.4)
        args.run_id += "_shifted"
        if args.adversarial:
            _, test_acc, train_acc, sec_acc, test_loss, train_loss = training.train_adv_nn(params, shifted_train, shifted_test, r, args.dataset, args.output_dir, args.run_id, test, finetune=finetune)
        else:
            _, test_acc, train_acc, sec_acc, test_loss, train_loss = training.train_nn(params, shifted_train, shifted_test, r, args.dataset, args.output_dir, args.run_id, test, finetune=finetune)

        test_acc_shift.append(test_acc)
        train_acc_shift.append(train_acc)
        sec_acc_shift.append(sec_acc_shift)
        all_train_shift_loss.append(train_loss)
        all_test_shift_loss.append(test_loss)


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
    np.save(args.output_dir + "/accuracy_train_shifted_" + args.run_id + ".npy", train_acc_shift)
    np.save(args.output_dir + "/accuracy_test_shifted_" + args.run_id + ".npy", test_acc_shift)
    np.save(args.output_dir + "/loss_train_" + args.run_id + ".npy", all_train_loss)
    np.save(args.output_dir + "/loss_test_" + args.run_id + ".npy", all_test_loss)
    np.save(args.output_dir + "/loss_train_shifted_" + args.run_id + ".npy", all_train_shift_loss)
    np.save(args.output_dir + "/loss_test_shifted_" + args.run_id + ".npy", all_test_shift_loss)
    params = [args.dataset, 0, args.adversarial, args.dataset_shift, args.fixed_seed, 
                1, args.variations, act, args.lr, args.lr_decay, args.weight_decay, 
                args.epochs, args.nodes_per_layer, args.num_layers, args.epsilon, args.beta, args.finetune]
    np.save(args.output_dir + "/params_" + args.run_id + ".npy", params)


    return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('file_base', type=str, help='file path of dataset through _train or _test')
    parser.add_argument('run_id', type=str)
    parser.add_argument('--variations', type=int, default=10) # how many models to compare, total?
    parser.add_argument('--adversarial', type=bool, default=False)
    parser.add_argument('--output_dir', type=str, default='.')
    parser.add_argument('--label_col', default='label', type=str)

    parser.add_argument('--lr', type=float, default=0.2)
    parser.add_argument('--lr_decay', type=float, default=0.8)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--finetune_epochs', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--nodes_per_layer', type=int, default=50)
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--optimizer', type=str, default=None)
    parser.add_argument('--fixed_seed', type=bool, default=False) # if true, use seed 0 for all random states

    parser.add_argument('--epsilon', type=float, default=0.5) # epsilon for finding adv. examples
    parser.add_argument('--dropout', type=float, default=0.0) # dropout rate
    parser.add_argument('--beta', type=float, default=5) # beta for softplus

    args = parser.parse_args()
    args.finetune = True
    args.dataset_shift = True
    args.threshold = 0 # do not add random noise to data

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

    args.shifted_dataset_shift = False
    args.orig_dataset_shift = True
    args.dataset = args.dataset + '_orig'
    args.file_base = args.file_base + '_orig'
    main(args)

