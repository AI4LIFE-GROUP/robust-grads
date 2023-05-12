import torch.nn as nn

def get_activation(activation):
    if type(activation) == type(nn.Softplus(beta=5)):
        return "soft"
    else:
        return "relu"
    

def get_random_states(dataset, dataset_shift, fixed_seed, variations, base_repeats):
    '''
    Return a string of random states to use for this set of experiments.

    Parameters:
        dataset - name of the dataset
        dataset_shift - True if this is a real-world data shift
        fixed_seed - True if the seed is fixed across model comparisons
        variations - How many variations to run for each baseline model (only for synthetic noise)
        base_repeats - Number of baseline models to average over
    '''
    if dataset_shift:
        if fixed_seed:
            random_states = [x for x in range(base_repeats)]
        else:
            if 'orig' in dataset:
                random_states = [x for x in range(base_repeats)]
            else:
                # if shifted w/o fixed seed, train shifted models on completely new seeds
                random_states = [x + base_repeats for x in range(base_repeats)]
    else:
        if fixed_seed:
            # we need variations # of base models and then base_repeats # of comparison models for each
            random_states = [x for x in range((variations + 1)*base_repeats)]     
        else:
            # we need base_repeats # of base models and then variations # of comparison models total
            random_states = [x for x in range(variations + base_repeats)]        
    return random_states


def find_seed(r, dataset_shift, fixed_seed, base_repeats, variations):
    '''
        Returns the experimental settings (add noise, baseline_model, and random_seed)
        for the experiment based on the random state r and additional parameters.

        Parameters:
            r - random state
            dataset_shift - True if this is a real-world data shift
            fixed_seed - True if the seed is fixed across model comparisons
            base_repeats - Number of baseline models to average over
            variations - How many variations to run for each baseline model (only for synthetic noise)
        Returns:
            add_noise - True if we are adding noise to the data
                    True only when dataset_shift is False and baseline_model is False
            baseline_model - True if this is a baseline model
            seed - random seed to use for this experiment
    '''
    if (not dataset_shift) and (not fixed_seed) and (r < base_repeats):
            baseline_model = True           
            add_noise = False
    elif (r % (variations + 1) == 0):
        baseline_model = True
        add_noise = False
    else: 
        baseline_model = False
        if dataset_shift:
            add_noise = False
        else:
            add_noise = True
    if (fixed_seed) and not dataset_shift:
        seed = (r) // (variations + 1)
    else:
        seed = r
    return add_noise, baseline_model, seed
        