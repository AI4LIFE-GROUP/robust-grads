import numpy as np

import utils.datasets as datasets
import utils.parser_utils as parser_utils
import training
import utils.exp_utils as exp_utils

'''
finetune_synth_experiments.py

Run fine-tuning experiments on datasets with synthetic noise.
For fine-tuning experiments on datasets with real-world data shifts, use finetune_experiments.py instead.
For retraining experiments on datasets with synthetic noise, use retrain_experiments.py instead.
'''
def main(args):
    dataset, run_id, output_dir = args.dataset, args.run_id, args.output_dir
    base_repeats, variations = args.base_repeats, args.variations
    file_base = args.file_base
    threshold = args.threshold
    fixed_seed, dataset_shift = args.fixed_seed, args.dataset_shift # True, False
    finetune = args.finetune # True

    test_accuracy_base, train_accuracy_base, secondary_accuracy_base = [], [], []
    test_loss_all_base, train_loss_all_base = [], []

    test_accuracy_ft, train_accuracy_ft, secondary_accuracy_ft = [], [], []
    test_loss_all_ft, train_loss_all_ft = [], []

    for r in range(base_repeats):
        base_seed = r * (variations + 1)

        train, test = datasets.load_data(file_base, dataset, r, threshold, add_noise=False, label_col=args.label_col)
        num_feat = train.num_features()
        num_classes = train.num_classes()

        params = training.Params(args.lr, args.lr_decay, args.epochs, args.lime_epochs, args.batch_size, loss_fn=args.loss, num_feat=num_feat, 
                        num_classes=num_classes, activation=args.activation, nodes_per_layer=args.nodes_per_layer,
                        num_layers=args.num_layers, optimizer=args.optimizer, seed=base_seed, epsilon = args.epsilon, dropout= args.dropout,
                        weight_decay = args.weight_decay)
        
        _, test_acc, train_acc, sec_acc, test_loss, train_loss = training.train_nn(params, train, test, base_seed, dataset, output_dir, run_id, 
                                                                                   secondary_dataset = None, finetune=False, finetune_base = True)
        test_accuracy_base.append(test_acc)
        train_accuracy_base.append(train_acc)
        test_loss_all_base.append(test_loss)
        train_loss_all_base.append(train_loss)
        if sec_acc is not None:
            secondary_accuracy_base.append(sec_acc)


        for v in range(variations):
            seed = r * (variations + 1) + (v + 1)
            params_ft = training.Params(args.lr, args.lr_decay, args.finetune_epochs, args.lime_finetune_epochs, args.batch_size, loss_fn=args.loss, num_feat=num_feat, 
                        num_classes=num_classes, activation=args.activation, nodes_per_layer=args.nodes_per_layer,
                        num_layers=args.num_layers, optimizer=args.optimizer, seed=base_seed, epsilon = args.epsilon, dropout= args.dropout,
                        weight_decay = args.weight_decay)
            # params_ft.seed determines the base model that we start with in train_nn()

            train, test = datasets.load_data(file_base, dataset, seed, threshold, add_noise=True, label_col=args.label_col)
            _, test_acc, train_acc, sec_acc, test_loss, train_loss = training.train_nn(params_ft, train, test, seed, dataset, output_dir, run_id, 
                                                                                       secondary_dataset = None, finetune = True, finetune_base = False)

            test_accuracy_ft.append(test_acc)
            train_accuracy_ft.append(train_acc)
            test_loss_all_ft.append(test_loss)
            train_loss_all_ft.append(train_loss)


    # save accuracy/loss
    np.save(output_dir + "/accuracy_train_base_" + run_id + ".npy", train_accuracy_base)
    np.save(output_dir + "/accuracy_test_base_" + run_id + ".npy", test_accuracy_base)
    np.save(output_dir + "/loss_train_base_" + run_id + ".npy", np.array(train_loss_all_base))
    np.save(output_dir + "/loss_test_base_" + run_id + ".npy", np.array(test_loss_all_base))
    np.save(output_dir + "/accuracy_train_ft_" + run_id + ".npy", train_accuracy_ft)
    np.save(output_dir + "/accuracy_test_ft_" + run_id + ".npy", test_accuracy_ft)
    np.save(output_dir + "/loss_train_ft_" + run_id + ".npy", np.array(train_loss_all_ft))
    np.save(output_dir + "/loss_test_ft_" + run_id + ".npy", np.array(test_loss_all_ft))

    # save params
    act = exp_utils.get_activation(args.activation)
    params = [dataset, threshold, args.adversarial, dataset_shift, fixed_seed, 
                base_repeats, variations, act, args.lr, args.lr_decay, args.weight_decay, 
                max(args.epochs), args.nodes_per_layer, args.num_layers, args.epsilon, args.beta, finetune]
    np.save(output_dir + "/params_" + run_id + ".npy", params)
    return 1

if __name__ == "__main__":
    parser = parser_utils.create_nn_parser()

    args = parser.parse_args()
    args = parser_utils.process_args_nn(args)

    args.finetune = True
    args.fixed_seed = True # fixed seed should be true because we only compare like models
    args.dataset_shift = False

    main(args)
