import pandas as pd
import numpy as np
import os
 
import argparse
import utils.metrics as metrics

def get_filename(filebase, run_id, epoch, shift = None, full = False):
    filename = filebase + "/results_" 
    if 'adv' in run_id:
        if (shift is not None) and (shift == 'orig'):
            filename += 'orig_'
        elif (shift is not None) and (shift == 'shift'):
            filename += 'shift_'
    if shift is not None:
        if shift == 'shift':
            filename = filename + run_id + "_shift_e" + str(epoch) + "_"
        elif shift == 'orig':
            filename = filename + run_id + "_orig_e" + str(epoch) + "_"
    else:
        filename = filename + run_id + "_e" + str(epoch) + "_"
    if full:
        filename = filename + "full_"
    return filename
    
def get_filename_acc(filebase, run_id):
    test = filebase + "/" + 'accuracy_test_'
    train = filebase + "/" + 'accuracy_train_'
    test = test + run_id + ".npy"
    train = train + run_id + ".npy"
    return test, train

def get_filename_loss(filebase, run_id):
    test = filebase + "/loss_test_" + run_id + ".npy"
    train = filebase + "/loss_train_" + run_id + ".npy"
    return test, train

def get_filename_acc_shift(filebase, run_id):
    train_shift = filebase + "/" + 'accuracy_train_' + run_id + '_shift.npy'
    test_shift = filebase + "/" + 'accuracy_test_' + run_id + '_shift.npy'
    train_orig = filebase + "/" + 'accuracy_train_' + run_id + '_orig.npy'
    test_orig = filebase + "/" + 'accuracy_test_' + run_id + '_orig.npy'
    return train_shift, test_shift, train_orig, test_orig

def get_filename_loss_shift(filebase, run_id):
    ''' on orig model, just get loss on orig datasets'''
    train_shift = filebase + "/" + 'loss_train_' + run_id + '_shift.npy'
    train_orig = filebase + "/" + 'loss_train_' + run_id + '_orig.npy'
    test_shift = filebase + "/" + 'loss_test_' + run_id + '_shift.npy'
    test_orig_full = filebase + "/" + 'loss_test_' + run_id + '_orig.npy'
    return train_shift, train_orig, test_shift, test_orig_full

def get_grads_base(filename, n, base_index, version = 'gradients', fixed_seed = True):
    if fixed_seed:
        idx = str((n+1)*base_index)
    else:
        idx = str(base_index)
    grads_base = np.load(filename + version + idx + ".npy")
    tops = []
    for i in range(5):
        tops.append(metrics.get_top_k(i+1, grads_base))
    return grads_base, tops








def add_tops_shift(filename_orig, filename_shift, n, lime_epoch, fixed_seed):
    full_res = []
    if lime_epoch:
        metric_names = ['gradients', 'salience', 'smoothgrad', 'lime', 'shap']
    else:
        metric_names = ['gradients', 'salience', 'smoothgrad']
    for name in metric_names: # 5 or 3
        new_res = []
        top_f = [[], [], [], [], []]
        top_sa = [[], [], [], [], []]
        top_cdc = [[], [], [], [], []]
        top_ssd = [[], [], [], [], []]
        grads_raw, grads_normed, grads_angle = [], [], []

        for idx in range(n):
            g1 = np.load(filename_orig + name + str(idx) + ".npy")
            if fixed_seed:
                g2 = np.load(filename_shift + name + str(idx) + ".npy")
            else:
                g2 = np.load(filename_shift + name + str(idx+n) + ".npy")
            
            s1 = np.sign(g1)
            s2 = np.sign(g2)
            tops1, tops2 = [], []
            for i in range(5):
                tops1.append(metrics.get_top_k(i+1, g1))
                tops2.append(metrics.get_top_k(i+1, g2))
            for i in range(5):
                top_f[i].append(metrics.top_k_overall(i+1, tops1[i], tops2[i]))
                top_sa[i].append(metrics.top_k_sa(i+1, tops1[i], tops2[i], s1, s2))
                top_cdc[i].append(metrics.top_k_cdc(i+1, tops1[i], tops2[i], s1, s2))
                top_ssd[i].append(metrics.top_k_ssd(i+1, tops1[i], tops2[i], s1, s2))
            grads_raw.append(metrics.gradnorm_raw(g1, g2, 2))
            grads_normed.append(metrics.gradnorm_norm(g1, g2, 2))
            grads_angle.append(metrics.gradient_angle(g1, g2, 2))
        for k in range(5):    
            for res in [top_f, top_sa, top_cdc, top_ssd]:
                new_res.append(np.mean(res[k]))
                for perc in [5,10, 25, 50, 75, 90,95]:
                    new_res.append(np.percentile(res[k], perc))
                new_res.append(np.std(res[k]))
        for res in [grads_raw, grads_normed, grads_angle]: 
            new_res.append(np.mean(res))
            for perc in [5,10, 25, 50, 75, 90,95]:
                new_res.append(np.percentile(res, perc))
            new_res.append(np.std(res))
        full_res.extend(new_res)
    if not lime_epoch:
        for i in range(2):
            new_res = []
            tops = [[[] for i in range(5)] for j in range(5)]
            grads = [[],[],[]]
            for t in tops:
                for j in range(4):
                    new_res.append([])
                    for p in range(5):
                        new_res.append([])
            for t in grads:
                new_res.append([])
                for p in range(5):
                    new_res.append([])
            full_res.extend(new_res)
    return full_res


def add_tops(filename, tops, grads_base, tops_sal, salience_base, tops_smooth, smooth_base, n, base_index, 
                tops_lime = None, lime_base = None, tops_shap = None, shap_base = None, fixed_seed = True):
    full_res = []
    if lime_base is not None:
        metric_names = ['gradients', 'salience', 'smoothgrad', 'lime', 'shap']
        metric_vals = [grads_base, salience_base, smooth_base, lime_base, shap_base]
        all_topk = [tops, tops_sal, tops_smooth, tops_lime, tops_shap]
    else:
        metric_names = ['gradients', 'salience', 'smoothgrad']
        metric_vals = [grads_base, salience_base, smooth_base]
        all_topk = [tops, tops_sal, tops_smooth]
    for name, met, top in zip(metric_names, metric_vals, all_topk):
        new_res = []
        signs = np.sign(met)

        top_f = [[], [], [], [], []]
        top_sa = [[], [], [], [], []]
        top_cdc = [[], [], [], [], []]
        top_ssd = [[], [], [], [], []]

        grads_raw, grads_normed, grads_angle = [], [], []
        for idx in range(n):
            if fixed_seed:
                new_idx = str((n+1)*base_index + idx + 1)
            else:
                new_idx = str(idx + base_index)
            g = np.load(filename + name + new_idx + ".npy")
            s = np.sign(g)
            top_temps = []
            for i in range(5):
                top_temps.append(metrics.get_top_k(i+1, g))
            for i in range(5):
                top_f[i].append(metrics.top_k_overall(i+1, top[i], top_temps[i]))
                top_sa[i].append(metrics.top_k_sa(i+1, top[i], top_temps[i], signs, s))
                top_cdc[i].append(metrics.top_k_cdc(i+1, top[i], top_temps[i], signs, s))
                top_ssd[i].append(metrics.top_k_ssd(i+1, top[i], top_temps[i], signs, s))

            grads_raw.append(metrics.gradnorm_raw(met, g, 2))
            grads_normed.append(metrics.gradnorm_norm(met, g, 2))
            grads_angle.append(metrics.gradient_angle(met, g, 2))
        for k in range(5):
            for res in [top_f, top_sa, top_cdc, top_ssd]:
                new_res.append(sum(res[k])/len(res[k]))
                for perc in [5,10,25,50,75,90,95]:
                    new_res.append(np.percentile(res[k], perc))
                new_res.append(np.std(res[k]))
        for res in [grads_raw, grads_normed, grads_angle]:
            new_res.append(sum(res)/len(res))
            for perc in [5,10, 25, 50, 75, 90,95]:
                new_res.append(np.percentile(res, perc))
            new_res.append(np.std(res))
        full_res.extend(new_res)
    if lime_base is None:
        # do this so that all columns line up
        for i in range(2):
            new_res = []
            tops = [[[], [], [], [], []] for j in range(5)]
            grads = [[], [], []]
            for t in tops:
                for j in range(4):
                    new_res.append([])
                    for p in range(5):
                        new_res.append([])
            for t in grads:
                new_res.append([])
                for p in range(5):
                    new_res.append([])
            full_res.extend(new_res)

    return full_res

def get_columns_common():
    columns = ['dataset', 'iteration', 'epochs', 'threshold', 'adversarial', 'fixed_seed', 'variations', 'activation',
                'lr', 'lr_decay', 'weight_decay', 'nodes_per_layer', 'num_layers', 'epsilon', 'beta', 'finetune']
    return columns

def add_pcts_columns(options, suffix):
    columns = []
    for t in options:
        columns.append(t + "_" + suffix)
        for perc in [5,10,25,50,75,90,95]:
            columns.append(t + '_' + suffix + '_p' + str(perc))
        columns.append(t + '_' + suffix + '_std')
    return columns

def add_pcts_columns_k(options):
    columns = []
    for o in options:
        for k in range(5): 
            for res in ['top_a_', 'top_sa_', 'top_cdc_', 'top_ssd_']:
                columns.append(o + res + str(k))
                for perc in [5,10,25,50,75,90,95]:
                    columns.append(o + res + str(k) + "_p" + str(perc))
                columns.append(o + res + str(k) + "_std")
        for res in ['gradient_raw', 'gradient_normed', 'gradient_angle']:
            columns.append(o + res)
            for perc in [5,10,25,50,75,90,95]:
                columns.append(o + res + "_p" + str(perc))
            columns.append(o + res + "_std")
    return columns

def get_columns_dataset_shift():
    columns = get_columns_common()
    options = ['train_shift', 'test_shift', 'train_orig', 'test_orig']
    columns.extend(add_pcts_columns(options, 'acc'))
    columns.extend(['train_loss_orig', 'test_loss_orig', 'train_loss_shift', 'test_loss_shift'])
    options = ['part_grad_', 'part_sal_', 'part_sg_', 'part_lime_', 'part_shap_', 'full_grad_',  'full_sal_',  'full_sg_',  'full_lime_',  'full_shap_']
    columns.extend(add_pcts_columns_k(options))
    return columns

def get_columns():
    columns = get_columns_common()
    options = ['train','test']
    columns.extend(add_pcts_columns(options, 'acc'))
    columns.extend(['train_loss', 'test_loss'])
    options = ['grad_', 'sal_', 'sg_', 'lime_', 'shap_']
    columns.extend(add_pcts_columns_k(options))
    return columns

def process_random_seed(args):
    ''' Random seed, synthetic noise '''
    filebase = args.filebase
    dataset_shift = False
    columns = get_columns()
    all_res = []
    for run_id in args.run_id:
        params = collect_params(args.filebase, run_id)
        dataset, threshold, adversarial = params[0], params[1], params[2]
        base_repeats, variations = int(params[5]), int(params[6])
        activation = params[7]
        lr, lr_decay, weight_decay = params[8], params[9], params[10]
        nodes_per_layer, num_layers = params[12], params[13]
        epsilon, beta, finetune = params[14], params[15], params[16]


        test, train = get_filename_acc(filebase, run_id)
        test_loss, train_loss = get_filename_loss(filebase, run_id)
        if test.split("/")[-1] not in os.listdir(filebase):
            continue
        test_acc, train_acc = np.matrix(np.load(test)), np.matrix(np.load(train))
        avg_test, avg_train = np.average(test_acc, axis=0), np.average(train_acc, axis=0)
        train_loss, test_loss = np.load(train_loss), np.load(test_loss)

        for iter in range(variations):
            for ep_idx in range(len(args.epochs)):
                epoch = args.epochs[ep_idx]
                filename = get_filename(filebase, run_id, epoch)
                grads_base, tops = get_grads_base(filename, base_repeats, iter, fixed_seed=False)
                
                salience_base, tops_sal = get_grads_base(filename, base_repeats, iter, fixed_seed=False, version='salience')
                smooth_base, tops_smooth = get_grads_base(filename, base_repeats, iter, fixed_seed=False, version='smoothgrad')
                if epoch in args.lime_epochs:
                    lime_base, tops_lime = get_grads_base(filename, base_repeats, iter, fixed_seed=False, version='lime')
                    shap_base, tops_shap = get_grads_base(filename, base_repeats, iter, fixed_seed=False, version='shap')
                else:
                    lime_base, tops_lime, shap_base, tops_shap = None, None, None, None
                new_res = [dataset, iter, epoch, threshold, adversarial, 0, base_repeats, activation, lr, lr_decay, weight_decay,
                            nodes_per_layer, num_layers, epsilon, beta, finetune]

                # add accuracy metrics
                new_res.append(avg_train[0,ep_idx])
                for perc in [5,10,25,50,75,90,95]:
                    new_res.append(np.percentile(np.array(train_acc.T[ep_idx])[0],perc))
                new_res.append(np.std(np.array(train_acc.T[ep_idx])[0]))
                new_res.append(avg_test[0,ep_idx])
                for perc in [5,10,25,50,75,90,95]:
                    new_res.append(np.percentile(np.array(test_acc.T[ep_idx])[0], perc))
                new_res.append(np.std(np.array(test_acc.T[ep_idx])[0]))
                
                # add train and test loss
                targets = [train_loss, test_loss]
                for tar in targets: # 2 columns
                    new_res.append(tar[ep_idx])

                new_res.extend(add_tops(filename, tops, grads_base, tops_sal, salience_base, tops_smooth, smooth_base, base_repeats, 
                                        base_repeats, tops_lime=tops_lime, lime_base = lime_base, tops_shap = tops_shap, shap_base = shap_base,
                                        fixed_seed=False))
                all_res.append(new_res)

    df = pd.DataFrame(all_res, columns=columns)
    df.to_csv(args.outputfile + ".csv", index=False)

def process_fixed_seed(args, finetune):
    filebase = args.filebase
    columns = get_columns()
    all_res = []
    for run_id in args.run_id:
        params = collect_params(args.filebase, run_id)
        dataset, threshold, adversarial = params[0], params[1], params[2]
        base_repeats, variations = int(params[5]), int(params[6]) 
        activation = params[7]
        lr, lr_decay, weight_decay = params[8], params[9], params[10]
        nodes_per_layer, num_layers = params[12], params[13]
        epsilon, beta = params[14], params[15]

        test, train = get_filename_acc(filebase, run_id)
        test_loss, train_loss = get_filename_loss(filebase, run_id)
        if test.split("/")[-1] not in os.listdir(filebase):
            continue
            
        test_acc, train_acc = np.matrix(np.load(test)), np.matrix(np.load(train))
        avg_test, avg_train = np.average(test_acc, axis=0), np.average(train_acc, axis=0)
        train_loss, test_loss = np.load(train_loss), np.load(test_loss)

        
        for iter in range(variations):
            for ep_idx in range(len(args.epochs)):
                epoch = args.epochs[ep_idx]
                if finetune:
                    base_epoch = args.max_epochs
                else:
                    base_epoch = epoch
                filename = get_filename(filebase, run_id, base_epoch)
                grads_base, tops = get_grads_base(filename, base_repeats, iter)
                salience_base, tops_sal = get_grads_base(filename, base_repeats, iter, version='salience')
                smooth_base, tops_smooth = get_grads_base(filename, base_repeats, iter, version='smoothgrad')
                if epoch in args.lime_epochs:
                    lime_base, tops_lime = get_grads_base(filename, base_repeats, iter, version='lime')
                    shap_base, tops_shap = get_grads_base(filename, base_repeats, iter, version='shap')
                else:
                    lime_base, tops_lime, shap_base, tops_shap = None, None, None, None
                    
                new_res = [dataset, iter, epoch, threshold, adversarial, 1, base_repeats, activation, lr, lr_decay, weight_decay,
                            nodes_per_layer, num_layers, epsilon, beta, params[16]] 

                # add accuracy metrics
                targets_avg = [avg_train, avg_test]
                targets = [train_acc, test_acc]
                for avg, tar in zip(targets_avg, targets): 
                    new_res.append(avg[0,ep_idx])
                    for perc in [5,10,25,50,75,90,95]:
                        new_res.append(np.percentile(np.array(tar.T[ep_idx])[0],perc))
                    new_res.append(np.std(np.array(tar.T[ep_idx])[0]))
                targets = [train_loss, test_loss]
                for tar in targets: # 2 columns
                    new_res.append(tar[ep_idx])

                
                # add top-k and gradient norm comparisons
                if finetune:
                    filename = get_filename(filebase, run_id, epoch) # reset for finding finetuned models!
                new_res.extend(add_tops(filename, tops, grads_base, tops_sal, salience_base, tops_smooth, smooth_base, base_repeats, 
                                        iter, tops_lime=tops_lime, lime_base = lime_base, tops_shap = tops_shap, shap_base = shap_base))
                all_res.append(new_res)
    df = pd.DataFrame(all_res,columns=columns)
    df.to_csv(args.outputfile + ".csv", index=False)

def process_shift(args, finetune):
    filebase = args.filebase
    columns = get_columns_dataset_shift()
    all_res = []
    for run_id in args.run_id:
        params = collect_params(args.filebase, run_id)
        dataset, threshold, adversarial = params[0], params[1], params[2]
        fixed_seed = (params[4] == "True")
        base_repeats = int(params[5])
        activation = params[7]
        lr, lr_decay, weight_decay = params[8], params[9], params[10]
        nodes_per_layer, num_layers = params[12], params[13]
        epsilon, beta = params[14], params[15]

        train_shift, test_shift, train_orig, test_orig = get_filename_acc_shift(filebase, run_id) 
        train_loss_orig, test_loss_orig, train_loss_shift, test_loss_shift = get_filename_loss_shift(filebase, run_id)
        if test_shift.split("/")[-1] not in os.listdir(filebase):
            print("failed for ",test_shift.split("/")[-1])
            continue

        train_shift_acc = np.matrix(np.load(train_shift))
        test_shift_acc = np.matrix(np.load(test_shift))
        train_orig_acc = np.matrix(np.load(train_orig))
        test_orig_acc = np.matrix(np.load(test_orig))
        avg_train_shift = np.average(train_shift_acc, axis=0)
        avg_test_shift = np.average(test_shift_acc, axis=0)
        avg_train_orig = np.average(train_orig_acc, axis=0)
        avg_test_orig = np.average(test_orig_acc, axis=0)
        train_loss_orig, test_loss_orig, train_loss_shift, test_loss_shift = np.load(train_loss_orig), np.load(test_loss_orig), np.load(train_loss_shift), np.load(test_loss_shift)
        
        for ep_idx in range(len(args.epochs)):
            epoch = args.epochs[ep_idx]
            if finetune:
                base_epoch = args.max_epochs
            else:
                base_epoch = epoch
            filename = get_filename(filebase, run_id, base_epoch, shift = 'orig')
            filename_shift = get_filename(filebase, run_id, base_epoch, shift = 'shift')
            filename_orig_full = get_filename(filebase, run_id, base_epoch, shift = 'orig', full=1)
            filename_shift_full = get_filename(filebase, run_id, base_epoch, shift = 'shift', full=1)

            
            new_res = [dataset, 0, epoch, threshold, adversarial, 1, base_repeats, activation, lr, lr_decay, weight_decay,
                        nodes_per_layer, num_layers, epsilon, beta, params[16]] 

            # add accuracy metrics
            targets_avg = [avg_train_shift, avg_test_shift, avg_train_orig, avg_test_orig]
            targets = [train_shift_acc, test_shift_acc, train_orig_acc, test_orig_acc]
            for avg, tar in zip(targets_avg, targets): # 24 columns
                new_res.append(avg[0,ep_idx])
                for perc in [5,10,25,50,75,90,95]:
                    new_res.append(np.percentile(np.array(tar.T[ep_idx])[0],perc))
                new_res.append(np.std(np.array(tar.T[ep_idx])[0]))
            targets = [train_loss_orig, test_loss_orig, train_loss_shift, test_loss_shift]
            for tar in targets: # 4 columns
                new_res.append(tar[ep_idx])

            # add top-k and gradient norm comparisons
            if finetune:
                filename_shift = get_filename(filebase, run_id, epoch, shift='shift')
                filename_shift_full = get_filename(filebase, run_id, epoch, shift='shift_full')
            new_res.extend(add_tops_shift(filename, filename_shift, base_repeats, (epoch in args.lime_epochs), fixed_seed))
            new_res.extend(add_tops_shift(filename_orig_full, filename_shift_full, base_repeats, (epoch in args.lime_epochs), fixed_seed))
            all_res.append(new_res)
    df = pd.DataFrame(all_res,columns=columns)
    df.to_csv(args.outputfile + ".csv", index=False)

def collect_params(filepath, run_id):
    try:
        params = np.load(filepath + "/params_" + run_id + ".npy")
    except:
        params = np.load(filepath + "/params_" + run_id + "_orig.npy")
    return params

def main(args):
    params = collect_params(args.filebase, args.run_id[0])
    finetune = (params[16] == "True")
    if params[3] == 'True':
        # real-world dataset shift
        process_shift(args, finetune)
    elif params[4] == "True":
        # fixed seed
        process_fixed_seed(args, finetune)
    else:
        # random seed
        process_random_seed(args)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('filebase', type=str, help='directory where .npy file outputs are stored')
    parser.add_argument('outputfile', type=str, help='name of output csv file -- do not include .csv extension') 
    parser.add_argument('--run_id', type=str, nargs='+')
    parser.add_argument('--epochs', type=int, nargs='+', help="all epochs we collected data for in ascending order")
    parser.add_argument('--lime_epochs', type=int, nargs='+', default=None, help="epochs we collected lime data for in ascending order")

    args = parser.parse_args()
    if args.lime_epochs is None:
        args.lime_epochs = args.epochs 
    args.max_epochs = args.epochs[-1]
    main(args)
