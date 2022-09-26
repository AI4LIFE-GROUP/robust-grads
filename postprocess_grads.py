import numpy as np
import argparse

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

def main(args):
    random_seeds = [x-1 for x in range(100)]

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
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('file_base') # location of saved results files
    parser.add_argument('output_dir') # where to save output
    args = parser.parse_args()
    main(args)
