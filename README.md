# robust-grads

## Under construction!! 

## Datasets:
### General
* [WHO](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who?resource=download)
* Adult income and COMPAS - no data yet because features are mostly categorical -- i.e., don't work for adversarial training
* MNIST

### Distribution shift
* German credit, [original](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who?resource=download) and [updated](https://archive.ics.uci.edu/ml/datasets/South+German+Credit+%28UPDATE%29) -- no data yet since features are categorical. 
* WHO dataset, again -- create multiple datasets that span different years.

It is easy to add other tabular datasets, just search for wherever 'whobin' appears in the code and update the list to include your dataset. Select the correct branches in `datasets.load_data` depending on if the label is binary or continuous. Right now, we assume continuous features, but binary features can be supported (apart from for adversarial training) by modifying the `continuous` flags in `datasets.load_data`

The code assumes that there exists `<file_base>_train.csv` and `<file_base>_test.csv` files, where `<file_base>` is the second command-line argument and can be, for example, `data/german` if the files `data/german_train.csv` and `data/german_test.csv` exist. 

## To run:
The code is broken into two pieces. `baseline_experiments.py` trains 100 neural networks and saves their accuracy (train and test), test-set predictions (binary predictions and logits), and test-set gradients in `.npy` files.

`python3 baseline_experiments.py <dataset> <file_base>`

The second piece, `postprocess_grads.py` post-processes the above files and computes pairwise similarity scores between all 100 models' top-k (k=1,2,3,4,5) feature scores and overall gradients.

`python3 postprocess_grads.py data_prefix <output_directory>` where `data_prefix` is the `.npy` file name up through the file `_`, e.g., `whobin_nn3_relu_t0.1`.

This will generate more numpy files, where each file stores information about the top-k or gradient norm (L2 norm is the default) for each test point. From there, you can load the .npy files in a script and compute the average across all test points for each metric.

### Random perturbation
Given a single dataset, we can add random perturbations to test similarity of models learned across similar datasets.

When running `baseline_experiments.py` include `--threshold=p`. For binary features, this will flip each feature with probability `p`.  For continuous features, this will add Gaussian noise of mean 0, standard deviation `p` to all data. .  

### Temporal or spatial distribution shift
The code assumes that there will be two data files, named `filename_orig_train.csv` and `filename_shift_train.csv` (and likewise for the test sets). 

To indicate you are testing dataset shift and that the code should not add random perturbations to the data, use the flag `--dataset-shift=1`. When using this option, additional output files will be generated. 
* files with `shiftcompare` in their titles compare the original and updated models directly
* * files with `shiftcompare` and `full` in the title compare the original and shifted model as evaluated on the updated test dataset
* * files without `full` compare the models as evaluated on the original test dataset
* Files without `shiftcompare` are the standard comparison of behavior across multiple random seeds
* *  Files with `shift` in the title use the updated test data to evaluate the models trained on the updated training data 
* * The `orig` files use the original test data to evaluate the models trained using the original training data

### Additional command-line parameters for `baseline_experiments.py`:
* `--adversarial` default False. If true, do adversarial training
* `--output_dir` specify where to save output files, default `.`
* `--label_col` label for output variable in dataset, default `label`
* `--lr` learning rate
* `--lr_decay` learning rate decay
* `epochs` 
* `--batch_size`
* `--activation` activation function (relu is default, `leak` for leaky relu and `soft` for softplus with beta=5 are also accepted)
* `--nodes_per_layer` number of nodes in each hidden layer
* `--num_layers` number of hidden layers
* `--optimizer` what optimizer to use, default is sgd, `amsgrad` and `adam` are also supported
* `--fixed_seed` if true, initialize all 100 neural nets using the same random seed (0). default False
* `--threshold` what fraction of the data to perturb (for binary datasets, this means we flip each feature with this probability)
* `--epsilon` epsilon to use in adversarial training
* `--dropout` how much dropout to use in training? default 0.
* `--dataset_shift` if true, compare an old and updated dataset 

These are not currently used in my experiments, but could be of use if we want to target dataset shifts or uncertainty to a subset of the dataset.
* `--target_indices` if only changing a subset of the data, what columns to look at to determine eligibility? Not currently used in experiments.
* `--target_vals` if only changing a subset of the data, what values should target_indices have to be included? Not currently used in experiments.
* `--indices_to_change` What indices can be modified?
* `--new_vals` If categorical, what values to change these indices to?
* `--strategy` default random. Other options are targeted or targeted-random. Not used so need to check code on definitions of each.
 

## Notes about the results
For most results we only consider parameter settings that result in good accuracy. Intuitively, if one model of the 100 has low accuracy, there is more space for it to disagree with other models, so the discrepancy metrics may be inflated. 

There was previously a bug where continuous data was not modified with Gaussian noise -- instead, the provided propotion (via the threshold argument) was set to 0. These results have been moved to the `old` directory within the `results` folder in case we ever want to come back to them.
