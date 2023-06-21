import numpy as np
from torch import nn

import utils.datasets as datasets
import training
import utils.parser_utils as parser_utils

'''
    finetune_experiments.py
    Run fine-tuning experiments on a real-world dataset shift

    To run retraining experiments, use retraining_experiments.py instead
    To run fine-tuning experiments with synthetic noise, use finetune_synth_experiments.py instead
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

        train, test = datasets.load_data(args.file_base, args.dataset,  r, args.threshold, add_noise=False, label_col = args.label_col)

        sec_name = args.dataset.replace('orig', 'shift')
        file_base = args.file_base.replace('orig', 'shift')
        shifted_train, shifted_test = datasets.load_data(file_base, sec_name, r, args.threshold, add_noise=False, label_col = args.label_col)

        num_feat = train.num_features()
        num_classes = train.num_classes()

        params = training.Params(args.lr, args.lr_decay, args.epochs, args.lime_epochs, args.batch_size, loss_fn=args.loss, num_feat=num_feat, 
                        num_classes=num_classes, activation=args.activation, nodes_per_layer=args.nodes_per_layer,
                        num_layers=args.num_layers, optimizer=args.optimizer, seed=seed, epsilon = args.epsilon, dropout= args.dropout,
                        weight_decay = args.weight_decay)
        
        _, test_acc, train_acc, sec_acc, test_loss, train_loss = training.train_nn(params, train, test, r, args.dataset, args.output_dir, args.run_id, shifted_test, finetune_base=finetune)
        
        test_accuracy.append(test_acc)
        train_accuracy.append(train_acc)
        secondary_accuracy.append(sec_acc)
        all_train_loss.append(train_loss)
        all_test_loss.append(test_loss)
        

        params.epochs = args.finetune_epochs
        params.lime_epochs = args.lime_finetune_epochs
        params.learning_rate = args.lr * (0.4)
        args.run_id += "_shifted"
        
        _, test_acc, train_acc, sec_acc, test_loss, train_loss = training.train_nn(params, shifted_train, shifted_test, r, sec_name, args.output_dir, args.run_id, test, finetune=finetune)

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
    
    args.run_id = orig_run_id
    np.save(args.output_dir + "/accuracy_train_base_" + args.run_id + ".npy", train_accuracy)
    np.save(args.output_dir + "/accuracy_test_base_" + args.run_id + ".npy", test_accuracy)
    np.save(args.output_dir + "/accuracy_train_ft_" + args.run_id + ".npy", train_acc_shift)
    np.save(args.output_dir + "/accuracy_test_ft_" + args.run_id + ".npy", test_acc_shift)
    np.save(args.output_dir + "/loss_train_base_" + args.run_id + ".npy", all_train_loss)
    np.save(args.output_dir + "/loss_test_base_" + args.run_id + ".npy", all_test_loss)
    np.save(args.output_dir + "/loss_train_ft_" + args.run_id + ".npy", all_train_shift_loss)
    np.save(args.output_dir + "/loss_test_ft_" + args.run_id + ".npy", all_test_shift_loss)
    params = [args.dataset, 0, 0, args.dataset_shift, args.fixed_seed, 
                1, args.variations, act, args.lr, args.lr_decay, args.weight_decay, 
                max(args.epochs), args.nodes_per_layer, args.num_layers, args.epsilon, args.beta, args.finetune]
    np.save(args.output_dir + "/params_" + args.run_id + ".npy", params)


    return 1

if __name__ == "__main__":
    parser = parser_utils.create_nn_parser()

    args = parser.parse_args()

    args.finetune = True
    args.dataset_shift = True
    args.threshold = 0 # do not add random noise to data

    args = parser_utils.process_args_nn(args)
    
    args.shifted_dataset_shift = False
    args.orig_dataset_shift = True
    args.dataset = args.dataset + '_orig'
    args.file_base = args.file_base + '_orig'
    main(args)

