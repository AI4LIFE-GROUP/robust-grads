# robust-grads

I will add more documentation (specifically about the user-specified parameter options) when I add to AI4Life


To run:
`python3 baseline_experiments.py whobin path_to_data_files --lr=0.2 --lr_decay=0.8 --epochs=10`

(see full list of parameters in baseline_experiments.py)

This will generate `.npy' files that contain the gradient, logits, and prediction for each point in the test set. Next, run 

`python3 postprocess_grads.py data_prefix` where `data_prefix` is the `.npy` file name up through the file `_`, e.g., `whobin_nn3_relu_t0.1`.

This will generate more numpy files, where each file stores information about the top-k or gradient norm (L2 norm is the default) for each test point. From there, you can load the .npy files in a script and compute the average across all test points for each metric.
