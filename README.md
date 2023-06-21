# robust-grads

## Datasets:
* [WHO](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who?resource=download)
* [Adult Income](https://archive.ics.uci.edu/ml/datasets/Adult)
* [HELOC](https://community.fico.com/s/explainable-machine-learning-challenge)
Our preprocessed data is located in the `data` folder.

The code assumes that there exists `<file_base>_train.csv` and `<file_base>_test.csv` files, where `<file_base>` is the second command-line argument and can be, for example, `data/adult` if the files `data/adult_train.csv` and `data/adult_test.csv` exist. 

## Dependencies
Our code relies on standard python libraries like `numpy`, `pandas`, and `pytorch`. We also use the [Captum](https://captum.ai/) library to compute explanations.

## To replicate our results:
For the WHO results, run `exps.sh`. CSV files will be saved as `ft_res_who.csv` for fine-tuning (Sec 4.1) and `rt_res_who` for retraining (Sec 4.2). 

For Adult/HELOC - TODO

## To run custom experiments:
The code is broken into a few pieces, depending on your goals. 

### Real-world distribution shifts
The code assumes that there will be two data files, named `filename_orig_train.csv` and `filename_shift_train.csv` (and likewise for the test sets). 

#### Retraining models
Use this section as a guide if you want to compare models that were retrained from scratch.

Start with `retrain_experiments.py` to generate the models, predictions, and test-set gradients. This file can be run as follows:

`python3 retrain_experiments.py <dataset> <file_base> <run_id> --dataset_shift=1`

Dataset is the name of the dataset (e.g., whobin, adult, or heloc). Filebase is the filename up through `_train.csv`. run_id is any unique string corresponding to this set of experiments.

These other parameters can be changed from the defaults, if desired:

*Setup*
* `--output_dir` default `.`. Where to save output files
* `--label_col` default `label`. Column name of the output variable in the dataset
* `--variations` default 10. How many trials to execute
* `--fixed_seed` default false. If true, use the same random model initialization for the original and shifted models

*Training parameters*
* `--lr` default 0.2. Learning rate
* `--lr_decay` default 0.8. Learning rate decay
* `--epochs` list of epochs (in ascending order) at which to calculate model explanations. the final (maximum) value is the total number of training epochs. Default [20]. 
* `--batch_size` default 128
* `--weight_decay` default 0
* `--dropout` default 0
* `--optimizer` default none, corresponding to SGD. Other options are `amsgrad` (Adam with amsgrad) or `adam`

*Model architecture*
* `--nodes_per_layer` default 50. Nodes per hidden layer
* `--num_layers` default 5. Number of hidden layers
* `--activation` default relu. Activation function, use `leak` for leaky relu and `soft` for softplus
* `--beta` default 5. If using softplus, beta parameter

*Extra* These parameters can also be used, but are not used for any of the experiments in the paper.
* `--adversarial` default false. If true, use adversarial training while training the models
* `--epsilon` (default 0.5, not used). Epsilon for constructing adversarial examples
* `--linear` default false. If true, train a linear model instead of a neural network

A number of files will be saved after running `retrain_experiments.py`: model parameters (`.pt` files), accuracy and loss (`.npy` files), and gradients (`.npy` files) for various attribution techniques. 
Several columns compare different data (original vs shifted), as follows:
* files with `shiftcompare` in their titles compare the original and updated models directly
* * files with `shiftcompare` and `full` in the title compare the original and shifted model as evaluated on the updated test dataset
* * files without `full` compare the models as evaluated on the original test dataset
* Files without `shiftcompare` are the standard comparison of behavior across multiple random seeds
* *  Files with `shift` in the title use the updated test data to evaluate the models trained on the updated training data 
* * The `orig` files use the original test data to evaluate the models trained using the original training data

To post-process these files into useful data, run `postprocess_retraining.py`, as follows:

`python3 postprocess_retraining.py <files_location> <output_file> --run_id <run_id1> <run_id2> <run_idn> --epochs <e1> <e2>`

files_location is where all of the `.npy` files live, i.e., the `output_dir` parameter from `retrain_experiments.py` (default `.`). output_file is the name of the csv file in which to store the results. `--run_id` takes a list of `run_id`'s from potentially multiple runs of `retrain_experiments.py` with different settings (however, all trials must have same `dataset_shift` value and `fixed_seed` value). Epochs is a list of epochs at which data was recorded (ascending order)

`postprocess_retraining.py` will save a CSV file containing aggregate information about explanation robustness. 

#### Fine-tuning models
Use `finetune_experiments.py` to run fine-tuning experiments on real-world data shifts.

`python3 finetune_experiments.py <dataset> <file_base> <run_id>`

The same command-line parameters as for `retrain_experiments.py` can be used, and have the same defaults, except for `--epochs` whose default is 1000. There is one additional command-line parameter, `--finetune_epochs` (default 250), which is the number of additional epochs for fine-tuning.

To post-process the raw output, run `postprocess_finetuning.py`, e.g.,

`python3 postprocess_finetuning.py <files_location> <output_file> --run_id <run_id1> <run_id2> <run_idn> --epochs <e1> --finetune_epochs <f1>`

Note that you must specify the epoch(s) at which the data was measured. 

### Synthetic dataset shift (Gaussian noise)
#### Retraining models
To compare models that are retrained from scratch, follow the same results as for real-world dataset shift retrained models (i.e., run `retrain_experiments.py` as described above, but omit the `dataset_shift` command-line parameter.)

These additional command-line parameters will be useful.
* `--threshold` default 0. Standard deviation of gaussian noise to add (for continuous features), or probability of modifying each feature (for binary features)
* `--base_repeats` default 10. How many "base models" to average over. The usage depends on whether `--fixed_seed` is true
* * For example, if fixed_seed is false, `--variations=5` and `--base_repeats=10`, we will train 10 base models and 5 modified models, for a total of 15 models. We compare each of the modified models with each of the base models. All 15 models that we train will use different random initializations.
* * For example, if `--fixed_seed=1`, `--variations=5`, and `--base_repeats=10`, we will train 10 base models and 5 modified models for each base model, for a total of 60 models. Each base model will be trained using a different model initialization, and all of the 5 modified models corresponding to it will use the same model initialization.


#### Fine-tuning models
For fine-tuning experiments on synthetic data shifts, use `finetune_synth_experiments.py` to run the experiments and `postprocess_finetuning_synth.py` to postprocess the results. E.g., 

`finetune_synth_experiments.py heloc <path_to_data> <run_id> --threshold 0.1`

`python postprocess_finetuning_synth.py . <output_file> --run_id <run_id> --epochs <e1> --finetune_epochs <f1>`




