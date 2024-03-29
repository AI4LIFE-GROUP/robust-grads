import argparse 
import torch.nn as nn

def create_tree_parser():
    parser = create_parser()
    parser.add_argument('--max_depth', type=int, default=[4], nargs='+', help='max depth(s) of trees. will run one test for each input')
    parser.add_argument('--min_samples_leaf', type=int, default=[5], nargs='+', help='minimum samples in leaf. Will run one test per input')
    parser.add_argument('--n_lime_samples', type=int, default=1000, help='number of samples to use for LIME')

    return parser
    
def create_nn_parser():
    parser = create_parser()
    parser.add_argument('--adversarial', type=bool, default=False, help="Included for compatibility with other scripts. Not used currently, is always False")
    
    parser.add_argument('--lr', type=float, default=0.2)
    parser.add_argument('--lr_decay', type=float, default=0.8) 
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--nodes_per_layer', type=int, default=50)
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--optimizer', type=str, default=None)

    parser.add_argument('--epsilon', type=float, default=0.5) # epsilon for finding adv. examples
    parser.add_argument('--dropout', type=float, default=0.0) # dropout rate
    parser.add_argument('--beta', type=float, default=5) # beta for softplus

    parser.add_argument('--epochs', type=int, default=[20], nargs='+', help='epoch(s) at which to calculate stats when training, in ascending order. Final (max) value is the total number of training epochs')
    parser.add_argument('--finetune_epochs', type=int, default=[20], nargs='+', help='epoch(s) at which to calculate stats when finetuning in ascending order. Final value is max number of fine-tuning epochs')
    parser.add_argument('--lime_epochs', type=int, default=None, nargs='+', help='epoch(s) at which to calculate lime and shap when training, in ascending order. If omitted, epochs is used.')
    parser.add_argument('--lime_finetune_epochs', type=int, default=None, nargs='+', help='epoch(s) at which to calculate lime and shap when fine-tuning, in ascending order. If omitted, finetune_epochs is used')

    parser.add_argument('--finetune', type=bool, default = False)
    parser.add_argument('--lr_finetune', type=float, default=None)
    return parser

def create_parser():
    ''' 
        Create parser and arguments that are used in all experiments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('file_base', type=str, help='file path of dataset through _train or _test')
    parser.add_argument('run_id', type=str)
    parser.add_argument('--variations', type=int, default=10, help="How many noisy trials to compare against each base model? not used in dataset shift experiments") # how many models to compare, total?
    parser.add_argument('--base_repeats', type=int, default=10, help="how many trials to average over") # how many base models do we need to compare with (and average over?)
    parser.add_argument('--dataset_shift', type=bool, default=False, help='true if data represents a real-world, not synthetic, shift')

    parser.add_argument('--output_dir', type=str, default='.', help='directory to save results in')
    parser.add_argument('--label_col', default='label', type=str)

    parser.add_argument('--fixed_seed', type=bool, default=False) 
    parser.add_argument('--threshold', type=float, default = 0.0, help='Standard deviation of noise (for gaussian noise on real-valued data) or probability that a feature si flipped (binary data)')

    return parser

def process_args_nn(args):
    args.loss = nn.CrossEntropyLoss()         

    if args.activation is not None:
        if args.activation == 'leak':
            args.activation = nn.LeakyReLU()
        elif args.activation == 'soft':
            args.activation = nn.Softplus(beta=args.beta)
        else:
            args.activation = nn.ReLU()
    else:
        args.activation = nn.ReLU()

    if args.lime_epochs is None:
        args.lime_epochs = args.epochs
    if args.lime_finetune_epochs is None:
        args.lime_finetune_epochs = args.finetune_epochs
    if args.lr_finetune is None:
        args.lr_finetune = args.lr

    if args.dataset_shift and (args.variations > 1):
        print("Variations must be 1 for dataset shift experiments. Setting variations=1 now.")
        args.variations = 1

    return args