import numpy as np

import utils.datasets as datasets
import utils.parser_utils as parser_utils
import training
import utils.exp_utils as exp_utils

def main(args):
    dataset, run_id, output_dir = args.dataset, args.run_id, args.output_dir
    dataset_shift, fixed_seed, finetune = args.dataset_shift, args.fixed_seed, args.finetune
    base_repeats, variations = args.base_repeats, args.variations
    threshold = args.threshold

    random_states = exp_utils.get_random_states(dataset, dataset_shift, fixed_seed, variations, base_repeats)

    test_accuracy, train_accuracy, secondary_accuracy = [], [], []

    for r in random_states:
        add_noise, baseline_model, seed = exp_utils.find_seed(r, dataset_shift, fixed_seed, base_repeats, variations)
        train, test = datasets.load_data(args.file_base, dataset, r, threshold, add_noise=add_noise, label_col=args.label_col)

        secondary_dataset = None
        if dataset_shift:
            secondary_dataset = datasets.load_secondary_dataset(dataset, args.file_base, r, threshold, add_noise=add_noise, label_col=args.label_col)

        num_feat = train.num_features()
        num_classes = train.num_classes()

        params = training.Params(args.lr, args.lr_decay, args.epochs, args.lime_epochs, args.batch_size, loss_fn=args.loss, num_feat=num_feat, 
                        num_classes=num_classes, activation=args.activation, nodes_per_layer=args.nodes_per_layer,
                        num_layers=args.num_layers, optimizer=args.optimizer, seed=seed, epsilon = args.epsilon, dropout= args.dropout,
                        weight_decay = args.weight_decay)
        
        ### TODO move linear to its own file?
        if args.linear:
            training.train_linear_models(args, train, test, r)
        elif args.adversarial:
            _, test_acc, train_acc, sec_acc, test_loss, train_loss = training.train_adv_nn(params, train, test, r, dataset, output_dir, run_id, secondary_dataset, (finetune and (not baseline_model)), (finetune and baseline_model))
        else:
            _, test_acc, train_acc, sec_acc, test_loss, train_loss = training.train_nn(params, train, test, r, dataset, output_dir, run_id, secondary_dataset, (finetune and (not baseline_model)), (finetune and baseline_model))
        test_accuracy.append(test_acc)
        train_accuracy.append(train_acc)
        if sec_acc is not None:
            secondary_accuracy.append(sec_acc)

    # print accuracy info
    trainacc = [round(train_accuracy[i][-1]*100, 2) for i in range(len(train_accuracy))]
    testacc = [round(test_accuracy[i][-1]*100, 2) for i in range(len(test_accuracy))]
    print("Train acc: ", trainacc)
    print("avg: ", sum(trainacc)/len(trainacc), " min: ", min(trainacc))
    print("Test acc: ", testacc)
    print("avg: ", sum(testacc)/len(testacc), " min: ", min(testacc))

    # save accuracy/loss
    np.save(output_dir + "/accuracy_train_" + run_id + ".npy", train_accuracy)
    np.save(output_dir + "/accuracy_test_" + run_id + ".npy", test_accuracy)
    np.save(output_dir + "/loss_train_" + run_id + ".npy", train_loss)
    np.save(output_dir + "/loss_test_" + run_id + ".npy", test_loss)

    # save params
    act = exp_utils.get_activation(args.activation)

    params = [dataset, threshold, args.adversarial, dataset_shift, fixed_seed, 
                base_repeats, variations, act, args.lr, args.lr_decay, args.weight_decay, 
                max(args.epochs), args.nodes_per_layer, args.num_layers, args.epsilon, args.beta, finetune]
    np.save(output_dir + "/params_" + run_id + ".npy", params)
    if dataset_shift:
        if 'orig' in dataset:
            np.save(output_dir + "/accuracy_test_orig_" + run_id + ".npy", secondary_accuracy)
        else:
            np.save(output_dir + "/accuracy_test_shift_" + run_id + ".npy", secondary_accuracy)

    return 1

if __name__ == "__main__":
    parser = parser_utils.create_nn_parser()
    parser = parser_utils.add_retraining_args(parser)

    args = parser.parse_args()
    args = parser_utils.process_args_nn(args)

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
