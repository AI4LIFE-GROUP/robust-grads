# robust-grads

## Under construction!! 

## Datasets:
* WHO https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who?resource=download
* German credit 
* Adult income
* COMPAS
* MNIST
It is easy to add other tabular datasets, just search for wherever 'german' appears and update the list to include your dataset. Select the correct branches in datasets.load_data depending on if features are binary or continuous. 

The code assumes that there exists `<file_base>_train.csv` and `<file_base>_test.csv` files, where `<file_base>` is the second command-line argument and can be, for example, `data/german`. 

## To run:
The code is broken into two pieces. `baseline_experiments.py` trains 100 neural networks and saves their accuracy (train and test), test-set predictions (binary predictions and logits), and test-set gradients.

The second piece, 

`python3 baseline_experiments.py whobin path_to_data_files --lr=0.2 --lr_decay=0.8 --epochs=10`

### Parameters:

(see full list of parameters in baseline_experiments.py)

This will generate `.npy' files that contain the gradient, logits, and prediction for each point in the test set. Next, run 

`python3 postprocess_grads.py data_prefix` where `data_prefix` is the `.npy` file name up through the file `_`, e.g., `whobin_nn3_relu_t0.1`.

This will generate more numpy files, where each file stores information about the top-k or gradient norm (L2 norm is the default) for each test point. From there, you can load the .npy files in a script and compute the average across all test points for each metric.
