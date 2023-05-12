import numpy as np

import utils.datasets as datasets
import utils.parser_utils as parser_utils
import training
import utils.exp_utils as exp_utils

def shift_main(args):
    random_states = [x for x in range(args.variations)] # no ``base models'', exactly

def main(args):
    dataset, run_id, output_dir = args.dataset, args.run_id, args.output_dir
    dataset_shift, fixed_seed, finetune = args.dataset_shift, args.fixed_seed, True
    base_repeats, variations = args.base_repeats, args.variations
    file_base = args.file_base
    threshold = args.threshold

    random_states_base = [x for x in range(args.base_repeats)]
    random_states_var = [x for x in range(args.variations)]

    test_accuracy_base, train_accuracy_base, secondary_accuracy_base = [], [], []
    test_loss_all_base, train_loss_all_base = [], []

    test_accuracy_ft, train_accuracy_ft, secondary_accuracy_ft = [], [], []
    test_loss_all_ft, train_loss_all_ft = [], []

    for r in random_states_base:
        if args.dataset_shift:
            dataset = args.dataset + '_orig'
            file_base = args.file_base + '_orig'
        # only add synthetic noise before fine-tuning.
        add_noise = False
        seed = r

        train, test = datasets.load_data(file_base, dataset, r, threshold, add_noise=add_noise, label_col=args.label_col)
        secondary_dataset = None
        if dataset_shift:
            secondary_dataset = datasets.load_secondary_dataset(dataset, file_base, r, threshold, add_noise=add_noise, label_col=args.label_col)

        num_feat = train.num_features()
        num_classes = train.num_classes()

        params = training.Params(args.lr, args.lr_decay, args.epochs, args.lime_epochs, args.batch_size, loss_fn=args.loss, num_feat=num_feat, 
                        num_classes=num_classes, activation=args.activation, nodes_per_layer=args.nodes_per_layer,
                        num_layers=args.num_layers, optimizer=args.optimizer, seed=seed, epsilon = args.epsilon, dropout= args.dropout,
                        weight_decay = args.weight_decay)
        
        if args.adversarial:
            _, test_acc, train_acc, sec_acc, test_loss, train_loss = training.train_adv_nn(params, train, test, r, dataset, output_dir, run_id, secondary_dataset, False, True)
        else:
            _, test_acc, train_acc, sec_acc, test_loss, train_loss = training.train_nn(params, train, test, r, dataset, output_dir, run_id, secondary_dataset, False, True)
        test_accuracy_base.append(test_acc)
        train_accuracy_base.append(train_acc)
        test_loss_all_base.append(test_loss)
        train_loss_all_base.append(train_loss)
        if sec_acc is not None:
            secondary_accuracy_base.append(sec_acc)

        # train 
        if dataset_shift:
            dataset = dataset.replace('orig', 'shift')
            file_base = file_base.replace('orig', 'shift')
        params_ft = training.Params(args.lr, args.lr_decay, args.finetune_epochs, args.lime_finetune_epochs, args.batch_size, loss_fn=args.loss, num_feat=num_feat, 
                        num_classes=num_classes, activation=args.activation, nodes_per_layer=args.nodes_per_layer,
                        num_layers=args.num_layers, optimizer=args.optimizer, seed=seed, epsilon = args.epsilon, dropout= args.dropout,
                        weight_decay = args.weight_decay)
        for r2 in random_states_var:
            add_noise = (not dataset_shift)
            seed = r2

            if add_noise:
                train, test = datasets.load_data(file_base, dataset, r2, threshold, add_noise=add_noise, label_col=args.label_col)
            elif dataset_shift:
                train, test = datasets.load_data(file_base, dataset, r2, threshold, add_noise=add_noise, label_col=args.label_col)
                secondary_dataset = datasets.load_secondary_dataset(dataset, file_base, r2, threshold, add_noise=add_noise, label_col=args.label_col)

            if args.adversarial:
                _, test_acc, train_acc, sec_acc, test_loss, train_loss = training.train_adv_nn(params_ft, train, test, r, dataset, output_dir, run_id, secondary_dataset, True, False)
            else:
                _, test_acc, train_acc, sec_acc, test_loss, train_loss = training.train_nn(params_ft, train, test, r, dataset, output_dir, run_id, secondary_dataset, True, False)

            test_accuracy_ft.append(test_acc)
            train_accuracy_ft.append(train_acc)
            test_loss_all_ft.append(test_loss)
            train_loss_all_ft.append(train_loss)
            if sec_acc is not None:
                secondary_accuracy_ft.append(sec_acc)

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
                max(args.epochs), args.nodes_per_layer, args.num_layers, args.epsilon, args.beta, True]
    np.save(output_dir + "/params_" + run_id + ".npy", params)
    if dataset_shift:
            np.save(output_dir + "/accuracy_test_sec_origmodel_" + run_id + ".npy", secondary_accuracy_base)
            np.save(output_dir + "/accuracy_test_sec_ftmodel_" + run_id + ".npy", secondary_accuracy_ft)
    return 1

if __name__ == "__main__":
    parser = parser_utils.create_nn_parser()
    parser = parser_utils.add_retraining_args(parser)

    args = parser.parse_args()
    args = parser_utils.process_args_nn(args)

    args.finetune = True
    args.fixed_seed = False # fixed seed doesn't make sense -- we compare models starting with the base model weights

    main(args)
