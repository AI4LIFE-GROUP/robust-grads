import argparse
import numpy as np
import pandas as pd
from torch import nn

import utils.data_utils as data_utils
import utils.datasets as datasets
import utils.parser_utils as parser_utils
import training

# NEURAL ARCHITECTURE - see neural_net.py

def main(args):
    if args.dataset_shift:
        if args.fixed_seed:
            random_states = [x for x in range(args.variations)]
        else:
            if 'orig' in args.dataset:
                random_states = [x for x in range(args.variations)]
            else:
                random_states = [x + args.variations for x in range(args.variations)]
    else:
        if args.fixed_seed:
            # we need args.variations # of base models and then args.base_repeats # of comparison models for each
            random_states = [x for x in range((args.variations + 1)*args.base_repeats)]     
        if args.fixed_seed == False:
            # we need args.variations # of base models and then args.base_repeats # of comparison models total
            random_states = [x for x in range(args.variations + args.base_repeats)]        

    test_accuracy, train_accuracy, secondary_accuracy = [], [], []

    finetune = args.finetune
    for r in random_states:
        if (not args.dataset_shift) and (args.fixed_seed == False) and (r < args.base_repeats):
            baseline_model = True           
            add_noise = False
        elif (r % (args.variations + 1) == 0):
            baseline_model = True
            add_noise = False
        else: 
            baseline_model = False
            if args.dataset_shift:
                add_noise = False
            else:
                add_noise = True
        if (args.fixed_seed) and not args.dataset_shift:
            seed = (r) // (args.variations + 1)
        else:
            seed = r

        if 'whobin' in args.dataset:
            scaler = data_utils.get_scaler(pd.read_csv(args.file_base + '_train.csv').drop(
                columns=[args.label_col]), args.threshold, random_state=r, add_noise=add_noise)
            scaler_labels = None
        else:  # if data is already scaled in a preprocessing step, skip scaling step 
            scaler, scaler_labels = None, None
        train, test = datasets.load_data(args.file_base, args.dataset, scaler, scaler_labels, r, args.threshold, add_noise=add_noise)
        secondary_dataset = None
        if args.dataset_shift and 'orig' in args.dataset:
            sec_name = args.dataset.replace('orig', 'shift')
            file_base = args.file_base.replace('orig', 'shift')
            _, secondary_dataset = datasets.load_data(file_base, sec_name, scaler, scaler_labels, r, args.threshold, add_noise=add_noise)
        elif args.dataset_shift:
            sec_name = args.dataset.replace('shift', 'orig')
            file_base = args.file_base.replace('shift', 'orig')
            _, secondary_dataset = datasets.load_data(file_base, sec_name, scaler, scaler_labels, r, args.threshold, add_noise=add_noise)

        num_feat = train.num_features()
        num_classes = train.num_classes()

        params = training.Params(args.lr, args.lr_decay, args.epochs, args.lime_epochs, args.batch_size, loss_fn=args.loss, num_feat=num_feat, 
                        num_classes=num_classes, activation=args.activation, nodes_per_layer=args.nodes_per_layer,
                        num_layers=args.num_layers, optimizer=args.optimizer, seed=seed, epsilon = args.epsilon, dropout= args.dropout,
                        weight_decay = args.weight_decay)
        
        ### TODO insert other model architectures here
        if args.linear:
            training.train_linear_models(args, train, test, r)
        elif args.adversarial:
            _, test_acc, train_acc, sec_acc, test_loss, train_loss = training.train_adv_nn(params, train, test, r, args.dataset, args.output_dir, args.run_id, secondary_dataset, (finetune and (not baseline_model)), (finetune and baseline_model))
        else:
            _, test_acc, train_acc, sec_acc, test_loss, train_loss = training.train_nn(params, train, test, r, args.dataset, args.output_dir, args.run_id, secondary_dataset, (finetune and (not baseline_model)), (finetune and baseline_model))
        test_accuracy.append(test_acc)
        train_accuracy.append(train_acc)
        if sec_acc is not None:
            secondary_accuracy.append(sec_acc)

    print(len(train_accuracy))
    print(train_accuracy[0].shape)
    # print accuracy info
    trainacc = [round(train_accuracy[i][-1]*100, 2) for i in range(len(train_accuracy))]
    testacc = [round(test_accuracy[i][-1]*100, 2) for i in range(len(test_accuracy))]
    print("Train acc: ", trainacc)
    print("avg: ", sum(trainacc)/len(trainacc), " min: ", min(trainacc))
    print("Test acc: ", testacc)
    print("avg: ", sum(testacc)/len(testacc), " min: ", min(testacc))


    # save accuracy/loss
    np.save(args.output_dir + "/accuracy_train_" + args.run_id + ".npy", train_accuracy)
    np.save(args.output_dir + "/accuracy_test_" + args.run_id + ".npy", test_accuracy)
    np.save(args.output_dir + "/loss_train_" + args.run_id + ".npy", train_loss)
    np.save(args.output_dir + "/loss_test_" + args.run_id + ".npy", test_loss)

    # save params
    if type(args.activation) == type(nn.ReLU()):
        act = "relu"
    elif type(args.activation) == type(nn.Softplus(beta=5)):
        act = "soft"
    params = [args.dataset, args.threshold, args.adversarial, args.dataset_shift, args.fixed_seed, 
                args.base_repeats, args.variations, act, args.lr, args.lr_decay, args.weight_decay, 
                max(args.epochs), args.nodes_per_layer, args.num_layers, args.epsilon, args.beta, args.finetune]
    np.save(args.output_dir + "/params_" + args.run_id + ".npy", params)
    if args.dataset_shift:
        if 'orig' in args.dataset:
            np.save(args.output_dir + "/accuracy_test_orig_" + args.run_id + ".npy", secondary_accuracy)
        else:
            np.save(args.output_dir + "/accuracy_test_shift_" + args.run_id + ".npy", secondary_accuracy)

    return 1

if __name__ == "__main__":
    parser = parser_utils.create_parser()
    parser = parser_utils.add_retraining_args(parser)

    args = parser.parse_args()
    args = parser_utils.process_args(args)

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
