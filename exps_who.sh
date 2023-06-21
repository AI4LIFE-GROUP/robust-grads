#!/bin/bash
# Re-run the WHO experiments

# Finetune.
python src/finetune_experiments.py whobin75 data/who/whobin whobin_wd001_ft_soft_b5_ep4000 --base_repeats=10 --activation=soft --beta=5  --lr=0.5 --epochs=4000 --finetune_epochs=1000 --weight_decay=0.001 --dataset_shift=1
python src/finetune_experiments.py whobin75 data/who/whobin whobin_wd001_ft_soft_b10_ep4000 --base_repeats=10 --activation=soft --beta=10  --lr=0.5 --epochs=4000 --finetune_epochs=1000 --weight_decay=0.001 --dataset_shift=1
python src/finetune_experiments.py whobin75 data/who/whobin whobin_wd001_ft_relu_ep4000 --base_repeats=10 --lr=0.5 --epochs=4000 --finetune_epochs=1000 --weight_decay=0.001 --dataset_shift=1
python src/finetune_experiments.py whobin75 data/who/whobin whobin_wd01_ft_relu_ep4000  --base_repeats=10 --lr=0.5 --epochs=4000 --finetune_epochs=1000 --weight_decay=0.01 --dataset_shift=1 
python src/finetune_experiments.py whobin75 data/who/whobin whobin_wd0_ft_relu_ep4000  --base_repeats=10 --lr=0.5 --epochs=4000 --finetune_epochs=1000 --weight_decay=0.0 --dataset_shift=1

python src/postprocess_finetuning.py . ft_res_who --run_id whobin_wd001_ft_soft_b5_ep4000 whobin_wd001_ft_soft_b10_ep4000 whobin_wd001_ft_relu_ep4000 whobin_wd01_ft_relu_ep4000 whobin_wd0_ft_relu_ep4000 --epochs=4000 --finetune_epochs=1000

# Retrain.
python src/retrain_experiments.py whobin75 data/who/whobin whobin_rt_relu_wd_001_lr8_ep80 --base_repeats=10  --dataset_shift=1 --lr=0.8 --lr_decay=1 --epochs=80 --weight_decay=0.001 --fixed_seed=1 --dataset_shift=1
python src/retrain_experiments.py whobin75 data/who/whobin whobin_rt_relu_wd_01_lr8_ep80 --base_repeats=10  --dataset_shift=1 --lr=0.8 --lr_decay=1 --epochs=80 --weight_decay=0.01 --fixed_seed=1 --dataset_shift=1
python src/retrain_experiments.py whobin75 data/who/whobin whobin_rt_relu_wd_0_lr8_ep80 --base_repeats=10  --dataset_shift=1 --lr=0.8 --lr_decay=1 --epochs=80 --weight_decay=0.0 --fixed_seed=1 --dataset_shift=1
python src/retrain_experiments.py whobin75 data/who/whobin whobin_rt_soft_b5_wd_001_lr8_ep80 --base_repeats=10  --dataset_shift=1 --lr=0.8 --lr_decay=1 --epochs=80 --weight_decay=0.001 --activation=soft --beta=5 --fixed_seed=1 --dataset_shift=1 
python src/retrain_experiments.py whobin75 data/who/whobin whobin_rt_soft_b10_wd_001_lr8_ep80 --base_repeats=10  --dataset_shift=1 --lr=0.8 --lr_decay=1 --epochs=80 --weight_decay=0.001 --activation=soft --beta=10 --fixed_seed=1 --dataset_shift=1

python src/postprocess_retraining.py . rt_res_who --run_id whobin_rt_relu_wd_001_lr8_ep80 whobin_rt_relu_wd_01_lr8_ep80 whobin_rt_relu_wd_0_lr8_ep80 whobin_rt_soft_b5_wd_001_lr8_ep80 whobin_rt_soft_b10_wd_001_lr8_ep80 --epochs 80 --lime_epochs 80


