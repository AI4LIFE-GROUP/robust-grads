#!/bin/bash
# Re-run the WHO experiments

# Finetune.
python dataset_shift_exp.py whobin75 data/who/whobin_75 whobin75_wd001_ft_soft_b5_ep4000 --activation=soft --beta=5 --variations=10 --lr=0.5 --epochs=4000 --finetune_epochs=1000 --finetune=1 --weight_decay=0.001
python dataset_shift_exp.py whobin75 data/who/whobin_75 whobin75_wd001_ft_soft_b10_ep4000 --activation=soft --beta=10 --variations=10 --lr=0.5 --epochs=4000 --finetune_epochs=1000 --finetune=1 --weight_decay=0.001
python dataset_shift_exp.py whobin75 data/who/whobin_75 whobin75_wd001_ft_relu_ep4000 --variations=10 --lr=0.5 --epochs=4000 --finetune_epochs=1000 --finetune=1 --weight_decay=0.001
python dataset_shift_exp.py whobin75 data/who/whobin_75 whobin75_wd01_ft_relu_ep4000 --variations=10 --lr=0.5 --epochs=4000 --finetune_epochs=1000 --finetune=1 --weight_decay=0.01
python dataset_shift_exp.py whobin75 data/who/whobin_75 whobin75_wd0_ft_relu_ep4000 --variations=10 --lr=0.5 --epochs=4000 --finetune_epochs=1000 --finetune=1 --weight_decay=0.0

python comp_fs_shift.py . ft_res_who --run_id whobin75_wd001_ft_soft_b5_ep4000 whobin75_wd001_ft_soft_b10_ep4000 whobin75_wd001_ft_relu_ep4000 whobin75_wd01_ft_relu_ep4000 whobin75_wd0_ft_relu_ep4000


# Retrain.
python baseline_experiments.py whobin75 data/who/whobin_75 whobin75_rt_relu_wd_001_lr8_ep80 --base_repeats=1 --variations=10 --dataset_shift=1 --lr=0.8 --lr_decay=1 --epochs=80 --weight_decay=0.001 --fixed_seed=1
python baseline_experiments.py whobin75 data/who/whobin_75 whobin75_rt_relu_wd_01_lr8_ep80 --base_repeats=1 --variations=10 --dataset_shift=1 --lr=0.8 --lr_decay=1 --epochs=80 --weight_decay=0.01 --fixed_seed=1
python baseline_experiments.py whobin75 data/who/whobin_75 whobin75_rt_relu_wd_0_lr8_ep80 --base_repeats=1 --variations=10 --dataset_shift=1 --lr=0.8 --lr_decay=1 --epochs=80 --weight_decay=0.0 --fixed_seed=1
python baseline_experiments.py whobin75 data/who/whobin_75 whobin75_rt_soft_b5_wd_001_lr8_ep80 --base_repeats=1 --variations=10 --dataset_shift=1 --lr=0.8 --lr_decay=1 --epochs=80 --weight_decay=0.001 --activation=soft --beta=5 --fixed_seed=1
python baseline_experiments.py whobin75 data/who/whobin_75 whobin75_rt_soft_b10_wd_001_lr8_ep80 --base_repeats=1 --variations=10 --dataset_shift=1 --lr=0.8 --lr_decay=1 --epochs=80 --weight_decay=0.001 --activation=soft --beta=10 --fixed_seed=1

python single_comp_metrics.py . rt_res_who --run_id whobin75_rt_relu_wd_001_lr8_ep80 whobin75_rt_relu_wd_01_lr8_ep80 whobin75_rt_relu_wd_0_lr8_ep80 whobin75_rt_soft_b5_wd_001_lr8_ep80 whobin75_rt_soft_b10_wd_001_lr8_ep80


