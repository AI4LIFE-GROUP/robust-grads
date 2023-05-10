import argparse
import torch.nn as nn

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('file_base', type=str)
    parser.add_argument('--dataset_shift', type=bool, default=False)
    parser.add_argument('--adversarial', type=bool, default=False)
    parser.add_argument('--output_dir', type=str, default='.')
    parser.add_argument('--label_col', default='label', type=str)

    parser.add_argument('--linear', type=bool, default=False)

    parser.add_argument('--lr', type=float, default=0.2)
    parser.add_argument('--lr_decay', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--nodes_per_layer', type=int, default=50)
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--optimizer', type=str, default=None)
    parser.add_argument('--loss', type=str, default=None)
    parser.add_argument('--fixed_seed', type=bool, default=False) # if true, use seed 0 for all random states

    parser.add_argument('--target_indices', type=str, default='') # if only changing a subset of data, what criteria to use
    parser.add_argument('--target_vals', type=str, default='')
    parser.add_argument('--indices_to_change', type=str, default='') # what indices to change?
    parser.add_argument('--new_vals', type=str, default='') # what can we change modified values to?
    parser.add_argument('--threshold', type=float, default = 0.0)
    parser.add_argument('--strategy', type=str, default='random')
    parser.add_argument('--epsilon', type=float, default=0.5) # epsilon for finding adv. examples
    parser.add_argument('--dropout', type=float, default=0.0) # dropout rate

    return parser

def process_args(args):
    # Determine target_indices (if any)
    if args.target_indices == '':
        args.target_indices = []
        args.target_vals = []
    else:
        args.target_indices = list(map(int, args.target_indices.split("_")))
        args.target_vals = list(map(float, args.target_vals.split("_")))

        # Validate target_indices and target_vals
        err_msg = 'target_indices and target_vals must be the same length'
        assert len(args.target_indices) == len(args.target_vals), err_msg
        
    # Determine indices_to_change and new_vals (if any)
    if args.indices_to_change == '':
        args.indices_to_change = []
        args.new_vals = []
    else:    
        args.indices_to_change = list(map(int, args.indices_to_change.split("_")))
        args.new_vals = list(map(float, args.new_vals.split("_")))

        # Validate indices_to_change and new_vals
        err_msg = 'indices_to_change and new_vals must be the same length'
        assert len(args.indices_to_change) == len(args.new_vals), err_msg
    
    # Validate strategy
    strategies = ['random', 'targeted', 'untargeted', 'targeted-random', 'none']
    args.strategy = str.strip(args.strategy)
    assert args.strategy in strategies, 'strategy must be in [' + ' '.join(strategies) + '] but is ' + args.strategy
    
    # Select loss function
    if args.dataset in ['mnist']: 
        args.loss = nn.MSELoss()
    else:
        args.loss = nn.CrossEntropyLoss()

    return args