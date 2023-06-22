#!/bin/bash
# Re-run a subset of the Adult experiments

# vary threshold to run the complete set of experiments
threshold=0.001

# Finetune.
python src/finetune_synth_experiments.py adult data/adult/adult adult_wd001_ft_soft_b5_ep500 --base_repeats 10 --variations 1 --activation soft --beta 5  --lr 0.5 --lr_finetune 0.01 --epochs 500 --finetune_epochs 50 --weight_decay 0.001 --threshold $threshold
python src/finetune_synth_experiments.py adult data/adult/adult adult_wd001_ft_soft_b10_ep500 --base_repeats 10 --variations 1 --activation soft --beta 10  --lr 0.5 --lr_finetune 0.01 --epochs 500 --finetune_epochs 50 --weight_decay 0.001 --threshold $threshold
python src/finetune_synth_experiments.py adult data/adult/adult adult_wd001_ft_relu_ep500 --base_repeats 10 --variations 1 --lr 0.5 --lr_finetune 0.01 --epochs 500 --finetune_epochs 50 --weight_decay 0.001 --threshold $threshold --lr_decay 1
python src/finetune_synth_experiments.py adult data/adult/adult adult_wd01_ft_relu_ep500  --base_repeats 10 --variations 1 --lr 0.5 --lr-finetune 0.01 --epochs 500 --finetune_epochs 50 --weight_decay 0.01 --threshold $threshold --lr_decay 1
python src/finetune_synth_experiments.py adult data/adult/adult adult_wd0_ft_relu_ep500  --base_repeats 10 --variations 1 --lr 0.5 --lr_finetune 0.01 --epochs 500 --finetune_epochs 50 --weight_decay 0.0 --threshold $threshold --lr_decay 1

python src/postprocess_finetuning_synth.py . ft_res_adult --run_id adult_wd001_ft_soft_b5_ep500 adult_wd001_ft_soft_b10_ep500 adult_wd001_ft_relu_ep500 adult_wd01_ft_relu_ep500 adult_wd0_ft_relu_ep500 --epochs 500 --finetune_epochs 50

# Retrain.
python src/retrain_experiments.py adult data/adult/adult adult_rt_relu_wd_001_lr8_ep30 --base_repeats 10 --variations 1 --lr 0.2 --lr_decay 1 --epochs 30 --weight_decay 0.001 --fixed_seed 1 --threshold $threshold
python src/retrain_experiments.py adult data/adult/adult adult_rt_relu_wd_01_lr8_ep30 --base_repeats 10 --variations 1 --lr 0.2 --lr_decay 1 --epochs 30 --weight_decay 0.01 --fixed_seed 1 --threshold $threshold
python src/retrain_experiments.py adult data/adult/adult adult_rt_relu_wd_0_lr8_ep30 --base_repeats 10 --variations 1 --lr 0.2 --lr_decay 1 --epochs 30 --weight_decay 0.0 --fixed_seed 1 --threshold $threshold
python src/retrain_experiments.py adult data/adult/adult adult_rt_soft_b5_wd_001_lr8_ep30 --base_repeats 10 --variations 1 --lr 0.2 --lr_decay 1 --epochs 30 --weight_decay 0.001 --activation soft --beta 5 --fixed_seed 1 --threshold $threshold 
python src/retrain_experiments.py adult data/adult/adult adult_rt_soft_b10_wd_001_lr8_ep30 --base_repeats 10 --variations 1 --lr 0.2 --lr_decay 1 --epochs 30 --weight_decay 0.001 --activation soft --beta 10 --fixed_seed 1 --threshold $threshold

python src/postprocess_retraining.py . rt_res_adult --run_id adult_rt_relu_wd_001_lr8_ep30 adult_rt_relu_wd_01_lr8_ep30 adult_rt_relu_wd_0_lr8_ep30 adult_rt_soft_b5_wd_001_lr8_ep30 adult_rt_soft_b10_wd_001_lr8_ep30 --epochs 80 --lime_epochs 80

