import argparse
import numpy as np
import os
import pandas as pd

# ----------------------
# Accuracy and prediction consistency

def load_file(filename):
    data = np.load(filename)
    return data

def get_accuracy(filename):
    acc = np.load(filename)
    return sum(acc)/len(acc)

def get_differential_acc(filename_base, random_seeds):
    '''
    Given files of predictions for the test set over multiple random seeds, 
    return a (num_samples, num_classes)-dim array where entry [i,j] is the 
    fraction of random seeds that yielded class j for test point i.
    '''
    a = load_file(filename_base + "preds" + str(random_seeds[0]) + '.npy')
    preds = np.zeros((len(random_seeds), a.shape[0]))
    ix = 0
    for r in random_seeds:
        preds[ix] = load_file(filename_base + "preds" + str(r) + '.npy')
        ix += 1
    results = np.zeros((len(np.unique(preds)), a.shape[0]))
    ix = 0
    for i in np.unique(preds):
        results[ix] = np.sum(np.transpose(preds) == i, axis=1)
        ix += 1
    return np.transpose(results) / len(random_seeds)

def confidence(results):
    ''' 
    Returns the fraction of datasets that yielded the most-common prediction
    for each test samples. 
    
    Input is a (num_samples, num_classes)-dim array
    Returns a (num_samples)-dim array 
    '''
    return np.max(results,axis=1)

# ---------------------
# Gradients and top-k features

def get_all_grads(filename_base, random_seeds):
    '''
    Returns a (r, num_samples, num_features)-dim array where (i,j,k) is the gradient
    of feature k for test point j for the model trained on dataset i. 
    '''
    a = load_file(filename_base + "gradients" + str(random_seeds[0]) + ".npy")
    a = np.squeeze(a)
    if len(a.shape) == 3:
        grads = np.zeros((len(random_seeds), a.shape[0], a.shape[1], a.shape[2]))
    else:
        grads = np.zeros((len(random_seeds), a.shape[0], a.shape[1]))
    ix = 0
    for r in random_seeds:
        grads[ix] = np.squeeze(load_file(filename_base + "gradients" + str(r) + ".npy"))
        ix += 1
    return grads

def get_k_feat(grads, k):
    '''
    Input: grads is a (num samples, num features)-dim tensor
    
    Output: returns a tensor of shape (num_samples, k) where row i
            contains the top-k feature indices for sample i.
            
    The top features are calculated via gradient (absolute) magnitude
    '''
    idx = np.transpose(np.argpartition(np.abs(grads), -k, axis=-1))[-k:]
    return np.transpose(idx)

def grad_norm_pairwise(grads1, grads2, norm=2):
    '''
    Given two sets of gradients from the original and shifted models, compute the 
    gradient distances between the two models.
    '''
    assert grads1.shape == grads2.shape
    # mid_result will be [num_samples, num_features]
    mid_result = grads1 - grads2
    mid_result = np.squeeze(mid_result)
    # result will be the average difference across all features, sample-wise
    result = np.linalg.norm(mid_result, ord=norm, axis=(0,2))
    
    # we could compute mean of results (this would be consistent with grad_norm) but 
    # it might be useful for data analysis to return the full vector
    return result

def grad_norm(grads, norm = 2):
    '''
    Given a (r, num_samples, num_feat)-dim array grads, return the norm of average pairwise 
    change in gradients between different runs. 
    
    Returns a num_samples-dim vector where each entry is the average gradient norm for that test point
    across any two versions of the data.
    '''
    new_array = np.zeros((grads.shape[0], grads.shape[0], grads.shape[1]))
    if len(grads.shape) == 4:
        axes = (1,2)
    else:
        axes = 1
    for i in range(grads.shape[0]):
        for j in range(grads.shape[0]):
            if j > i:
                break
            new_array[i][j] = np.linalg.norm(grads[i] - grads[j], ord=norm, axis=axes)
    upper_tri = np.triu(np.transpose(new_array), 1)
    total_entries = grads.shape[0] * (grads.shape[0] - 1) / 2
    total = np.sum(upper_tri, axis=(1,2))
    
    return total / total_entries

def top_k_consistency_shift(top_k1, top_k2):
    '''
    Given two sets of top-k features from the original and shifted models, compute the 
    fraction top-k features that are the same between the two models.

    Input: top_k1 and top_k2 are (r, num_samples, k)-dim arrays
    Output: output is a (r, num_samples)-dim array where (i,j) is the fraction of top-k features that are the
            same between top_k1[i,j] and top_k2[i,j]
    '''
    results = []
    assert top_k1.shape == top_k2.shape
    for r in range(top_k1.shape[0]):
        this_top1 = top_k1[r]
        this_top2 = top_k2[r]
        results.append([len(np.intersect1d(row, row2)) / len(row) for row, row2 in zip(this_top1, this_top2)])
    results = np.array(results)
    # sum along axis 0
    return np.sum(results, axis=0) / top_k1.shape[0]

def top_k_consistency_pair(top_k1, top_k2, idx=None):
    '''
    Given two sets of top-k features from two different datasets models,
    return the element-wise similarity score of the top k features.
    Given k features where l top features are shared across the two datasets, 
    the similarity score is l/k
    '''
    assert top_k1.shape == top_k2.shape
    k = top_k1.shape[1]
    
    if idx is not None:
        top_k1 = top_k1[idx]
        top_k2 = top_k2[idx]

    mask = (top_k1[:,:,None] == top_k2[:,None,:]).any(-1)
    scores = np.sum(mask, axis=-1)/k
    return scores

def top_k_consistency(top_k, idx=None):
    if idx is not None:
        top_k = top_k[idx]
        
    new_array = np.zeros((top_k.shape[0], top_k.shape[0], top_k.shape[1]))
    for i in range(top_k.shape[0]):
        for j in range(top_k.shape[0]):
            if j > i:
                break
            new_array[i][j] = top_k_consistency_pair(top_k[i], top_k[j])
        
    # remove the diagonal (will always be one) and lower triangle (repeat of upper tri)
    upper_tri = np.triu(np.transpose(new_array), 1)
    total_entries = top_k.shape[0] * (top_k.shape[0] - 1) / 2
    return np.sum(upper_tri, axis=(1,2)) / total_entries

def pairwise_comparison(args, random_seeds):
    diff_acc = get_differential_acc(args.dataset + "_", random_seeds)

    grads = get_all_grads(args.dataset + "_", random_seeds)
    grad_norms = grad_norm(grads)
    first_part = args.output_dir + "/" + args.dataset + "_"
    if 'mnist' not in args.dataset:
        top_1 = get_k_feat(grads, 1)
        top_2 = get_k_feat(grads, 2)
        top_3 = get_k_feat(grads, 3)
        top_4 = get_k_feat(grads, 4)
        top_5 = get_k_feat(grads, 5)

        top1_const = top_k_consistency(top_1)
        top2_const = top_k_consistency(top_2)
        top3_const = top_k_consistency(top_3)
        top4_const = top_k_consistency(top_4)
        top5_const = top_k_consistency(top_5)
        
        np.save(first_part + "top1const.npy", top1_const)
        np.save(first_part + "top2const.npy", top2_const)
        np.save(first_part + "top3const.npy", top3_const)
        np.save(first_part + "top4const.npy", top4_const)
        np.save(first_part + "top5const.npy", top5_const)

        np.save(first_part + "top1.npy", top_1)
        np.save(first_part + "top2.npy", top_2)
        np.save(first_part + "top3.npy", top_3)
        np.save(first_part + "top4.npy", top_4)
        np.save(first_part + "top5.npy", top_5)

    np.save(first_part + "gradnorm.npy", grad_norms)
    np.save(first_part + "preds.npy", diff_acc)

def shift_comparison(args, random_seeds):
    dataset_orig_o = args.dataset
    dataset_shift_o = args.dataset.replace("orig", "shift")
    dataset_orig_f = args.dataset + "_full"
    dataset_shift_f = dataset_shift_o + "_full"

    grads = []
    for d in [dataset_orig_o, dataset_shift_o, dataset_orig_f, dataset_shift_f]:
        grads.append(get_all_grads(d + "_", random_seeds))

    first_part = args.output_dir + "/" + args.dataset + "_shiftcompare_"
    # compare orig datasets

    if 'mnist' not in args.dataset:
        top_1_orig = get_k_feat(grads[0], 1)
        top_2_orig = get_k_feat(grads[0], 2)
        top_3_orig = get_k_feat(grads[0], 3)
        top_4_orig = get_k_feat(grads[0], 4)
        top_5_orig = get_k_feat(grads[0], 5)

        top_1_shift = get_k_feat(grads[1], 1)
        top_2_shift = get_k_feat(grads[1], 2)
        top_3_shift = get_k_feat(grads[1], 3)
        top_4_shift = get_k_feat(grads[1], 4)
        top_5_shift = get_k_feat(grads[1], 5)

        top_1_const = top_k_consistency_shift(top_1_orig, top_1_shift)
        top_2_const = top_k_consistency_shift(top_2_orig, top_2_shift)
        top_3_const = top_k_consistency_shift(top_3_orig, top_3_shift)
        top_4_const = top_k_consistency_shift(top_4_orig, top_4_shift)
        top_5_const = top_k_consistency_shift(top_5_orig, top_5_shift)

        np.save(first_part + "top1.npy", top_1_const)
        np.save(first_part + "top2.npy", top_2_const)
        np.save(first_part + "top3.npy", top_3_const)
        np.save(first_part + "top4.npy", top_4_const)
        np.save(first_part + "top5.npy", top_5_const)
    gradnorms = grad_norm_pairwise(grads[0], grads[1])
    np.save(first_part + "gradnorm.npy", gradnorms)

    # compare full datasets
    first_part = first_part + "full_"
    if 'mnist' not in args.dataset:
        top_1_orig = get_k_feat(grads[2], 1)
        top_2_orig = get_k_feat(grads[2], 2)
        top_3_orig = get_k_feat(grads[2], 3)
        top_4_orig = get_k_feat(grads[2], 4)
        top_5_orig = get_k_feat(grads[2], 5)

        top_1_shift = get_k_feat(grads[3], 1)
        top_2_shift = get_k_feat(grads[3], 2)
        top_3_shift = get_k_feat(grads[3], 3)
        top_4_shift = get_k_feat(grads[3], 4)
        top_5_shift = get_k_feat(grads[3], 5)

        top_1_const = top_k_consistency_shift(top_1_orig, top_1_shift)
        top_2_const = top_k_consistency_shift(top_2_orig, top_2_shift)
        top_3_const = top_k_consistency_shift(top_3_orig, top_3_shift)
        top_4_const = top_k_consistency_shift(top_4_orig, top_4_shift)
        top_5_const = top_k_consistency_shift(top_5_orig, top_5_shift)

        np.save(first_part + "top1.npy", top_1_const)
        np.save(first_part + "top2.npy", top_2_const)
        np.save(first_part + "top3.npy", top_3_const)
        np.save(first_part + "top4.npy", top_4_const)
        np.save(first_part + "top5.npy", top_5_const)
    gradnorms = grad_norm_pairwise(grads[2], grads[3])
    np.save(first_part + "gradnorm.npy", gradnorms)

def load_np(filename):
    a = np.load(filename)
    return a

def final_step(dir_base, ):
    all_data = []
    bad = []
    for f in os.listdir(dir_base):
        if "." in f and len(f)<10:
            print(f)
            continue
        
        params = [act, threshold, epoch]

        count = 0
        for g in os.listdir(dir_base + "/" + f):
            count += 1
            if 'preds' in g:
                dataset = g.split("_preds")[0]
        if count == 1:
            bad.append(f)
            continue

        accuracy_file_name = dataset.split("_t")[0]
        if 'adv' in accuracy_file_name:
            accuracy_file_name = accuracy_file_name.split("_adv")[0] + accuracy_file_name.split("adv")[1]
        accuracy = load_np(dir_base + "/" + f + "/accuracy_" + accuracy_file_name + ".npy")
        test_acc = accuracy[0]
        train_acc = accuracy[1]
        params.append(min(test_acc))
        params.append(sum(test_acc)/len(test_acc))
        params.append(max(test_acc))
        params.append(np.std(test_acc))
        params.append(min(train_acc))
        params.append(sum(train_acc)/len(train_acc))
        params.append(max(train_acc))
        params.append(np.std(train_acc))

        # do this almost all the time, except for adv-only tests
        top1 = load_np(dir_base + "/" + f + "/" + dataset + "_top1const.npy")
        top2 = load_np(dir_base + "/" + f + "/" + dataset + "_top2const.npy")
        top3 = load_np(dir_base + "/" + f + "/" + dataset + "_top3const.npy")
        top4 = load_np(dir_base + "/" + f + "/" + dataset + "_top4const.npy")
        top5 = load_np(dir_base + "/" + f + "/" + dataset + "_top5const.npy")
        gradnorm = load_np(dir_base + "/" + f+ "/" + dataset + "_gradnorm.npy")
        if dataset_shift:
            for j in [top1, top2, top3, top4, top5, gradnorm]:
                params.append(sum(j)/len(j))
                for p in [10,25,50,75,90]:
                    params.append(np.percentile(np.array(j), p))
            all_data.append(params)
        else:
            for j in [top1, top2, top3, top4, top5, gradnorm]:
                params.append(sum(j)/len(j))
                for p in [10,25,50,75,90]:
                    params.append(np.percentile(np.array(j), p))
            all_data.append(params)
    return all_data
    
def main(args):
    epochs = [1, 2, 5, 10, 20, 30, 40, 50]
    thresholds = [0.0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    dataset = args.dataset
    filebase = args.filebase

    all_res = []
    for act in ['relu', 'soft', 'adv']:
        for threshold in thresholds:
            for ep_idx in range(len(epochs)):
                epoch = epochs[ep_idx]
                if args.dataset_shift:
                    filename = get_filename(dataset + '_orig', filebase, threshold, epoch, act)
                    filename_shift = get_filename(dataset + '_shift', filebase, threshold, epoch, act)
                    filename_orig_full = get_filename(dataset + "_orig", filebase, threshold, epoch, act, full=1)
                    filename_shift_full = get_filename(dataset + "_shift", filebase, threshold, epoch, act, full=1)
                else:
                    filename = get_filename(dataset, filebase, threshold, epoch, act)
                grads_base, tops = get_grads_base(filename)
                if args.dataset_shift:
                    grads_base_shift, tops_shift = get_grads_base(filename_shift)
                    grads_base_orig_full, tops_orig_full = get_grads_base(filename_orig_full)
                    grads_base_shift_full, tops_shift_full = get_grads_base(filename_shift_full)
                signs = np.sign(grads_base)
                if args.dataset_shift:
                    signs_shift = np.sign(grads_base_shift)
                    signs_orig_full = np.sign(grads_base_orig_full)
                    signs_shift_full = np.sign(grads_base_shift_full)
                new_res = [act, threshold, epoch]

                if args.dataset_shift:
                    new_res.append(avg_train[0,ep_idx])
                    new_res.append(avg_train_shift[0,ep_idx])
                    for perc in [10,25,50,75,90]:
                        new_res.append(np.percentile(np.array(train_acc.T[ep_idx])[0],perc))
                        new_res.append(np.percentile(np.array(train_acc_shift.T[ep_idx])[0],perc))
                    new_res.append(avg_test[0,ep_idx])
                    new_res.append(avg_test_shift[0,ep_idx])
                    new_res.append(avg_test_shift_part[0,ep_idx])
                    new_res.append(avg_test_orig_full[0,ep_idx])
                    for perc in [10,25,50,75,90]:
                        new_res.append(np.percentile(np.array(test_acc.T[ep_idx])[0], perc))
                        new_res.append(np.percentile(np.array(test_acc_shift.T[ep_idx])[0], perc))
                        new_res.append(np.percentile(np.array(test_acc_shift_part.T[ep_idx])[0], perc))
                        new_res.append(np.percentile(np.array(test_acc_orig_full.T[ep_idx])[0], perc))
                else:
                    new_res.append(avg_train[0,ep_idx])
                    for perc in [10,25,50,75,90]:
                        new_res.append(np.percentile(np.array(train_acc.T[ep_idx])[0],perc))
                    new_res.append(avg_test[0,ep_idx])
                    for perc in [10,25,50,75,90]:
                        new_res.append(np.percentile(np.array(test_acc.T[ep_idx])[0], perc))

                new_res.extend(add_tops(filename, tops, signs, grads_base))
                if args.dataset_shift:
                    new_res.extend(add_tops(filename_shift, tops_shift, signs_shift, grads_base_shift))
                    new_res.extend(add_tops(filename_orig_full, tops_orig_full, signs_orig_full, grads_base_orig_full))
                    new_res.extend(add_tops(filename_shift_full, tops_shift_full, signs_shift_full, grads_base_shift_full))

                all_res.append(new_res)

    columns = ['activation', 'threshold', 'epochs']
    if args.dataset_shift:
        options = ['train', 'train_shift', 'test', 'test_shift', 'test_shift_part', 'test_orig_full']
    else:
        options = ['train','test']
    for t in options:
        columns.append(t + "_acc")
        for perc in [10,25,50,75,90]:
            columns.append(t + '_acc_p' + str(perc))
    if args.dataset_shift:
        options = ['orig_part_', 'shift_part_', 'orig_full_', 'shift_full_']
    else:
        options = ['']
    for o in options:
        for k in range(5):
            for res in ['top_a_', 'top_sa_', 'top_cdc_', 'top_ssd_']:
                columns.append(o + res + str(k))
                for perc in [10,25,50,75,90]:
                    columns.append(o + res + str(k) + "_p" + str(perc))
        for res in ['gradient_raw', 'gradient_unit', 'gradient_avg']:
            columns.append(o + res)
            for perc in [10,25,50,75,90]:
                columns.append(o + res + "_p" + str(perc))

    df = pd.DataFrame(all_res,columns=columns)
    df.to_csv(args.outputfile)

    if args.dataset_shift:
        orig_dataset = args.dataset
        # evaluate the original, partial dataset on the partial test set
        pairwise_comparison(args, random_seeds)
        # next, evaluate the shifted dataset on the whole test set
        args.dataset = args.dataset + "_full"
        args.dataset = args.dataset.replace("orig", "shift")
        pairwise_comparison(args, random_seeds)

        # also do comparison between the two datasets
        args.dataset = orig_dataset
        shift_comparison(args, random_seeds)

    else:
        pairwise_comparison(args, random_seeds)    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('output_dir') # where to save output
    parser.add_argument('--dataset_shift', type=bool, default= False)
    args = parser.parse_args()

    main(args)
