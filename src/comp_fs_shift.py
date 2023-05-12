import pandas as pd
import numpy as np
import os
 
import argparse

import utils.metrics as metrics

def get_filename(filebase, run_id, epoch, shift = None, full = False):
    filename = filebase + "/results_" 
    filename = filename + run_id + "_e" + str(epoch) + "_"
    if full:
        filename = filename + "full_"
    return filename

def get_filename_acc_shift(filebase, run_id):
    train_shift = filebase + "/" + 'accuracy_train_base_' + run_id + '.npy'
    test_shift = filebase + "/" + 'accuracy_test_base_' + run_id + '.npy'
    return train_shift, test_shift

def get_filename_loss_shift(filebase, run_id):
    ''' on orig model, just get loss on orig datasets'''
    train_shift = filebase + "/" + 'loss_train_ft_' + run_id + '.npy'
    train_orig = filebase + "/" + 'loss_train_base_' + run_id + '.npy'
    test_shift = filebase + "/" + 'loss_test_ft_' + run_id + '.npy'
    test_orig_full = filebase + "/" + 'loss_test_ft_' + run_id + '.npy'
    return train_shift, train_orig, test_shift, test_orig_full

def get_grads_base(filename, n, base_index, version = 'gradients', fixed_seed = True):
    idx = str((n+1)*base_index)

    grads_base = np.load(filename + version + idx + ".npy")
    return grads_base



def add_tops_shift(filename_orig, filename_shift, n):
    full_res = []
    metric_names = ['gradients'] 
    for name in metric_names: # 5 or 3
        new_res = []
        grads_raw, grads_normed, grads_angle = [], [], []

        for idx in range(n):
            g1 = np.load(filename_orig + name + str(idx) + ".npy")
            g2 = np.load(filename_shift + name + str(idx) + ".npy")

            grads_raw.append(metrics.gradnorm_raw(g1, g2, 2))
            grads_normed.append(metrics.gradnorm_norm(g1, g2, 2))
            grads_angle.append(metrics.gradient_angle(g1, g2, 2))
        for res in [grads_raw, grads_normed, grads_angle]: 
            new_res.append(np.mean(res))
            for perc in [5, 10, 25, 50, 75, 90, 95]:
                new_res.append(np.percentile(res, perc))
            # append standard deviation
            new_res.append(np.std(res))
        full_res.extend(new_res)
    return full_res


def get_columns():
    columns = ['dataset', 'iteration', 'epochs', 'finetune_epochs', 'threshold', 'adversarial', 'fixed_seed', 'variations', 'activation',
                'lr', 'lr_decay', 'weight_decay', 'nodes_per_layer', 'num_layers', 'epsilon', 'beta', 'finetune']

    options = ['train_shift', 'test_shift']

    for t in options:
        columns.append(t + "_acc")
        for perc in [10,25,50,75,90]:
            columns.append(t + '_acc_p' + str(perc))
        columns.append(t + '_acc_std')

    columns.extend(['train_loss_orig', 'test_loss_orig', 'train_loss_shift', 'test_loss_shift'])

    options = ['part_grad_', 'full_grad_']

    for o in options:
        for res in ['gradient_raw', 'gradient_normed', 'gradient_angle']:
            columns.append(o + res)
            for perc in [5,10,25,50,75,90,95]:
                columns.append(o + res + "_p" + str(perc))
            columns.append(o + res + "_std")
    return columns


def collect_params(filepath, run_id):
    try:
        params = np.load(filepath + "/params_" + run_id + ".npy")
    except:
        try:
            params = np.load(filepath + "/params_" + run_id + "_orig.npy")
        except:
            params = np.load(filepath + "/params_" + run_id + "_shifted.npy")
    return params

def main(args):
    # Note: fixed seed is always true (doesn't make sense to start from different seed when we're starting with the same pre-trained model)
    filebase = args.filebase
    columns = get_columns()
    all_res = []
    for run_id in args.run_id:
        params = collect_params(args.filebase, run_id)
        dataset, threshold, adversarial = params[0], params[1], params[2]
        n = int(params[6]) # n = num variations
        activation = params[7]
        lr, lr_decay, weight_decay = params[8], params[9], params[10]
        nodes_per_layer, num_layers = params[12], params[13]
        epsilon, beta = params[14], params[15]

        train_shift, test_shift = get_filename_acc_shift(filebase, run_id) 
        train_loss_orig, test_loss_orig, train_loss_shift, test_loss_shift = get_filename_loss_shift(filebase, run_id)
        print("filebase is ",filebase)
        print("test_shift is ",test_shift)
        if test_shift.split("/")[-1] not in os.listdir(filebase):
            print("HERE", test_shift.split("/")[-1])
            continue

        train_shift_acc = np.matrix(np.load(train_shift))
        test_shift_acc = np.matrix(np.load(test_shift))
        avg_train_shift = np.average(train_shift_acc, axis=0)
        avg_test_shift = np.average(test_shift_acc, axis=0)
        train_loss_orig, test_loss_orig, train_loss_shift, test_loss_shift = np.load(train_loss_orig), np.load(test_loss_orig), np.load(train_loss_shift), np.load(test_loss_shift)
        
        for epoch in args.epochs:
            for ft_epoch in args.finetune_epochs:
                filename = get_filename(filebase, run_id, epoch, shift = 'orig')
                filename_shift = get_filename(filebase, run_id, ft_epoch, shift = 'shift')
                filename_orig_full = get_filename(filebase, run_id, epoch, shift = 'orig', full=1)
                filename_shift_full = get_filename(filebase, run_id,  ft_epoch, shift = 'shift', full=1)

            
                new_res = [dataset, 0, epoch, ft_epoch, threshold, adversarial, 1, n, activation, lr, lr_decay, weight_decay,
                            nodes_per_layer, num_layers, epsilon, beta, params[16]] # 16 columns

                # add accuracy metrics
                targets_avg = [avg_train_shift, avg_test_shift]
                targets = [train_shift_acc, test_shift_acc]
                for avg, tar in zip(targets_avg, targets): # 24 columns
                    new_res.append(avg[0,0])
                    for perc in [10,25,50,75,90]:
                        new_res.append(np.percentile(np.array(tar.T[0])[0],perc))
                    new_res.append(np.std(np.array(tar.T[0])))
                targets = [train_loss_orig, test_loss_orig, train_loss_shift, test_loss_shift]
                for tar in targets: # 4 columns
                    new_res.append(tar[0])

                new_res.extend(add_tops_shift(filename, filename_shift_full, n))
                new_res.extend(add_tops_shift(filename_orig_full, filename_shift, n))
                all_res.append(new_res)
    df = pd.DataFrame(all_res,columns=columns)
    df.to_csv(args.outputfile + ".csv", index=False)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('filebase', type=str, help='directory where .npy file outputs are stored')
    parser.add_argument('outputfile', type=str, help='name of output csv file -- do not include .csv extension') # don't include .csv extension
    parser.add_argument('--run_id', type=str, nargs='+')
    parser.add_argument('--epochs',type=int, nargs='+')
    parser.add_argument('--finetune_epochs',type=int, nargs='+')


    args = parser.parse_args()
    main(args)
