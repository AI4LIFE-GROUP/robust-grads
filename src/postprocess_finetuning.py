import pandas as pd
import numpy as np
import os
 
import argparse

import utils.metrics as metrics

def get_filename(filebase, run_id, epoch, full = False, finetune = False):
    filename = filebase + "/results_" 
    filename = filename + run_id 
    if finetune:
        filename = filename + "_shifted"
    filename = filename + "_e" + str(epoch) + "_"
    if full:
        filename = filename + "full_"
    return filename

def get_filename_acc(filebase, run_id):
    train_base = filebase + "/" + 'accuracy_train_base_' + run_id + '.npy'
    test_base = filebase + "/" + 'accuracy_test_base_' + run_id + '.npy'
    train_ft = filebase + "/" + 'accuracy_train_ft_' + run_id + '.npy'
    test_ft = filebase + "/" + 'accuracy_test_ft_' + run_id + '.npy'
    return train_base, test_base, train_ft, test_ft

def get_filename_loss(filebase, run_id):
    ''' on orig model, just get loss on orig datasets'''
    train_ft = filebase + "/" + 'loss_train_ft_' + run_id + '.npy'
    train_base = filebase + "/" + 'loss_train_base_' + run_id + '.npy'
    test_ft = filebase + "/" + 'loss_test_ft_' + run_id + '.npy'
    test_base = filebase + "/" + 'loss_test_base_' + run_id + '.npy'
    return train_base, test_base, train_ft, test_ft
    
def get_grads_base(filename, n, base_index, version = 'gradients', fixed_seed = True):
    idx = str((n+1)*base_index)

    grads_base = np.load(filename + version + idx + ".npy")
    return grads_base



def add_tops(filename_base, filename_ft, n):
    full_res = []
    metric_names = ['gradients'] 
    for name in metric_names: # 5 or 3
        new_res = []
        grads_raw, grads_normed, grads_angle = [], [], []

        for idx in range(n):
            g1 = np.load(filename_base + name + str(idx) + ".npy")
            g2 = np.load(filename_ft + name + str(idx) + ".npy")

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

    options = ['train_ft', 'test_ft', 'train_base', 'test_base']

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
            params = np.load(filepath + "/params_" + run_id + "_base.npy")
        except:
            params = np.load(filepath + "/params_" + run_id + "_ft.npy")
    return params

def main(args):
    # Note: fixed seed is always true (doesn't make sense to start from different seed when we're starting with the same pre-trained model)
    filebase = args.filebase
    columns = get_columns()
    all_res = []
    for run_id in args.run_id:
        params = collect_params(args.filebase, run_id)
        dataset, threshold, adversarial = params[0], params[1], params[2]
        base_repeats = int(params[5])
        activation = params[7]
        lr, lr_decay, weight_decay = params[8], params[9], params[10]
        nodes_per_layer, num_layers = params[12], params[13]
        epsilon, beta = params[14], params[15]

        train_base, test_base, train_ft, test_ft = get_filename_acc(filebase, run_id) 
        train_loss_base, test_loss_base, train_loss_ft, test_loss_ft = get_filename_loss(filebase, run_id)
        if test_ft.split("/")[-1] not in os.listdir(filebase):
            print("HERE (error, couldn't find file)", test_ft.split("/")[-1])
            continue

        train_base_acc = np.matrix(np.load(train_base))
        test_base_acc = np.matrix(np.load(test_base))
        train_ft_acc = np.matrix(np.load(train_ft))
        test_ft_acc = np.matrix(np.load(test_ft))
        avg_train_base = np.average(train_base_acc, axis=0)
        avg_test_base = np.average(test_base_acc, axis=0)
        avg_train_ft = np.average(train_ft_acc, axis=0)
        avg_test_ft = np.average(test_ft_acc, axis=0)
        train_loss_base, test_loss_base, train_loss_ft, test_loss_ft = np.load(train_loss_base), np.load(test_loss_base), np.load(train_loss_ft), np.load(test_loss_ft)
        
        for epoch in args.epochs:
            for ft_epoch in args.finetune_epochs:
                filename_base = get_filename(filebase, run_id, epoch)
                filename_ft = get_filename(filebase, run_id, ft_epoch, finetune=True)
                filename_base_full = get_filename(filebase, run_id, epoch, full=1)
                filename_ft_full = get_filename(filebase, run_id,  ft_epoch, full=1, finetune=True)

            
                new_res = [dataset, 0, epoch, ft_epoch, threshold, adversarial, 1, base_repeats, activation, lr, lr_decay, weight_decay,
                            nodes_per_layer, num_layers, epsilon, beta, params[16]] # 16 columns

                # add accuracy metrics
                targets_avg = [avg_train_ft, avg_test_ft, avg_train_base, avg_test_base]
                targets = [train_ft_acc, test_ft_acc, train_base_acc, test_base_acc]
                for avg, tar in zip(targets_avg, targets): # 24 columns
                    new_res.append(avg[0,0])
                    for perc in [10,25,50,75,90]:
                        new_res.append(np.percentile(np.array(tar.T[0])[0],perc))
                    new_res.append(np.std(np.array(tar.T[0])))
                targets = [train_loss_base, test_loss_base, train_loss_ft, test_loss_ft]
                for tar in targets: # 4 columns
                    new_res.append(tar[0][-1]) # just add the final loss

                new_res.extend(add_tops(filename_base, filename_ft, base_repeats))
                new_res.extend(add_tops(filename_base_full, filename_ft_full, base_repeats))
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
