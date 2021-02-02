## VAE
#python main.py --model VAE --dataset mnist --act_func gelu --binarize True --hidden_dim 64 --batch_size 100 --net_type conv --num_samples 1 --max_epochs 50
#python main.py --model VAE --dataset omniglot --act_func gelu --binarize True --hidden_dim 64 --batch_size 25 --net_type conv --num_samples 1 --max_epochs 50
#python main.py --model VAE --dataset celeba --act_func gelu --binarize False --hidden_dim 64 --batch_size 50 --net_type conv --num_samples 1 --max_epochs 50 --limit_train_batches 0.1 --limit_val_batches 0.1
#
### IWAE
#python main.py --model IWAE --dataset mnist --act_func gelu --binarize True --hidden_dim 64 --batch_size 100 --net_type conv --num_samples 10 --max_epochs 50 --track_grad_norm 2
#python main.py --model IWAE --dataset omniglot --act_func gelu --binarize True --hidden_dim 64 --batch_size 25 --net_type conv --num_samples 10 --max_epochs 50
#python main.py --model IWAE --dataset celeba --act_func gelu --binarize False --hidden_dim 64 --batch_size 50 --net_type conv --num_samples 10 --max_epochs 50 --limit_train_batches 0.1 --limit_val_batches 0.1

## AIWAE
#python main.py --model AIWAE --dataset mnist --act_func gelu --binarize True --use_barker False --hidden_dim 64 --batch_size 100 --net_type conv --num_samples 1 --max_epochs 50 --step_size 0.01 --K 5 --n_leapfrogs 3
#python main.py --model AIWAE --dataset omniglot --act_func gelu --binarize True --use_barker False --hidden_dim 64 --batch_size 25 --net_type conv --num_samples 1 --max_epochs 50 --step_size 0.01 --K 5 --n_leapfrogs 3
#python main.py --model AIWAE --dataset celeba --act_func gelu --binarize False --use_barker False --hidden_dim 64 --batch_size 10 --net_type conv --num_samples 1 --max_epochs 50 --limit_train_batches 0.1 --limit_val_batches 0.1 --step_size 0.01 --K 3 --n_leapfrogs 2

## AIS_VAE
#python main.py --model AIS_VAE --dataset mnist --act_func gelu --binarize True --use_barker False --hidden_dim 64 --batch_size 100 --net_type conv --num_samples 1 --max_epochs 50 --step_size 0.01 --K 2 --track_grad_norm 2 --learnable_transitions False
#python main.py --model AIS_VAE --dataset omniglot --act_func gelu --binarize True --use_barker False --hidden_dim 64 --batch_size 25 --net_type conv --num_samples 1 --max_epochs 50 --step_size 0.01 --K 5
#python main.py --model AIS_VAE --dataset celeba --act_func gelu --binarize False --use_barker False --hidden_dim 64 --batch_size 10 --net_type conv --num_samples 1 --max_epochs 50 --limit_train_batches 0.1 --limit_val_batches 0.1 --step_size 0.01 --K 5

## ULA_VAE
#python main.py --model ULA_VAE --dataset mnist --act_func gelu --binarize True --hidden_dim 64 --batch_size 100 --net_type conv --num_samples 1 --max_epochs 50 --step_size 0.01 --K 1 --use_score_matching False
#python main.py --model ULA_VAE --dataset mnist --act_func gelu --binarize True --hidden_dim 64 --batch_size 100 --net_type conv --num_samples 1 --max_epochs 50 --step_size 0.01 --K 5 --use_score_matching False --track_grad_norm 2 --learnable_transitions True
#python main.py --model ULA_VAE --dataset mnist --act_func gelu --binarize True --hidden_dim 64 --batch_size 100 --net_type conv --num_samples 1 --max_epochs 50 --step_size 0.01 --K 1 --use_score_matching True
#python main.py --model ULA_VAE --dataset mnist --act_func gelu --binarize True --hidden_dim 64 --batch_size 100 --net_type conv --num_samples 1 --max_epochs 50 --step_size 0.01 --K 3 --use_score_matching True

## Stacked_VAE
#python main.py --model Stacked_VAE --dataset mnist --act_func gelu --binarize True --use_barker False --hidden_dim 64 --batch_size 100 --net_type conv --num_samples 1 --max_epochs 50 --step_size 0.01 --K 7 --track_grad_norm 2
#python main.py --model Stacked_VAE --dataset omniglot --act_func gelu --binarize True --use_barker False --hidden_dim 64 --batch_size 25 --net_type conv --num_samples 1 --max_epochs 50 --step_size 0.01 --K 7 --track_grad_norm 2
#python main.py --model Stacked_VAE --dataset celeba --act_func gelu --binarize False --use_barker False --hidden_dim 64 --batch_size 50 --net_type conv --num_samples 1 --max_epochs 50 --step_size 0.01 --K 7 --track_grad_norm 2 --limit_train_batches 0.1 --limit_val_batches 0.1

#python main.py --model ULA_VAE --dataset mnist --binarize True --batch_size 100 --num_samples 1 --max_epochs 30 --step_size 0.01 --K 5 --variance_sensitive_step True --ula_skip_threshold 0.1 --acceptance_rate_target 0.5
#python main.py --model ULA_VAE --dataset mnist --binarize True --batch_size 100 --num_samples 1 --max_epochs 30 --step_size 0.01 --K 5 --variance_sensitive_step True --ula_skip_threshold 0.1 --acceptance_rate_target 0.75
#python main.py --model ULA_VAE --dataset mnist --binarize True --batch_size 100 --num_samples 1 --max_epochs 30 --step_size 0.01 --K 5 --variance_sensitive_step True --ula_skip_threshold 0.1 --acceptance_rate_target 0.9
#python main.py --model ULA_VAE --dataset mnist --binarize True --batch_size 100 --num_samples 1 --max_epochs 30 --step_size 0.01 --K 5 --variance_sensitive_step True --ula_skip_threshold 0.1 --acceptance_rate_target 0.99
#
#python main.py --model ULA_VAE --dataset mnist --binarize True --batch_size 100 --num_samples 1 --max_epochs 30 --step_size 0.01 --K 5 --variance_sensitive_step True --ula_skip_threshold 0.2 --acceptance_rate_target 0.5
#python main.py --model ULA_VAE --dataset mnist --binarize True --batch_size 100 --num_samples 1 --max_epochs 30 --step_size 0.01 --K 5 --variance_sensitive_step True --ula_skip_threshold 0.2 --acceptance_rate_target 0.75
#python main.py --model ULA_VAE --dataset mnist --binarize True --batch_size 100 --num_samples 1 --max_epochs 30 --step_size 0.01 --K 5 --variance_sensitive_step True --ula_skip_threshold 0.2 --acceptance_rate_target 0.9
#python main.py --model ULA_VAE --dataset mnist --binarize True --batch_size 100 --num_samples 1 --max_epochs 30 --step_size 0.01 --K 5 --variance_sensitive_step True --ula_skip_threshold 0.2 --acceptance_rate_target 0.99
#
#python main.py --model ULA_VAE --dataset mnist --binarize True --batch_size 100 --num_samples 1 --max_epochs 30 --step_size 0.01 --K 5 --variance_sensitive_step True --ula_skip_threshold 0.5 --acceptance_rate_target 0.5
#python main.py --model ULA_VAE --dataset mnist --binarize True --batch_size 100 --num_samples 1 --max_epochs 30 --step_size 0.01 --K 5 --variance_sensitive_step True --ula_skip_threshold 0.5 --acceptance_rate_target 0.75
#python main.py --model ULA_VAE --dataset mnist --binarize True --batch_size 100 --num_samples 1 --max_epochs 30 --step_size 0.01 --K 5 --variance_sensitive_step True --ula_skip_threshold 0.5 --acceptance_rate_target 0.9
#python main.py --model ULA_VAE --dataset mnist --binarize True --batch_size 100 --num_samples 1 --max_epochs 30 --step_size 0.01 --K 5 --variance_sensitive_step True --ula_skip_threshold 0.5 --acceptance_rate_target 0.99
#
#python main.py --model ULA_VAE --dataset mnist --binarize True --batch_size 100 --num_samples 1 --max_epochs 30 --step_size 0.01 --K 5 --variance_sensitive_step True --ula_skip_threshold 0.8 --acceptance_rate_target 0.5
#python main.py --model ULA_VAE --dataset mnist --binarize True --batch_size 100 --num_samples 1 --max_epochs 30 --step_size 0.01 --K 5 --variance_sensitive_step True --ula_skip_threshold 0.8 --acceptance_rate_target 0.75
#python main.py --model ULA_VAE --dataset mnist --binarize True --batch_size 100 --num_samples 1 --max_epochs 30 --step_size 0.01 --K 5 --variance_sensitive_step True --ula_skip_threshold 0.8 --acceptance_rate_target 0.9
#python main.py --model ULA_VAE --dataset mnist --binarize True --batch_size 100 --num_samples 1 --max_epochs 30 --step_size 0.01 --K 5 --variance_sensitive_step True --ula_skip_threshold 0.8 --acceptance_rate_target 0.99

#python main.py --model ULA_VAE --dataset mnist --binarize True --batch_size 100 --num_samples 1 --max_epochs 30 --step_size 0.01 --K 1 --variance_sensitive_step True --ula_skip_threshold 0.5 --acceptance_rate_target 0.95
#python main.py --model ULA_VAE --dataset mnist --binarize True --batch_size 100 --num_samples 1 --max_epochs 30 --step_size 0.01 --K 3 --variance_sensitive_step True --ula_skip_threshold 0.5 --acceptance_rate_target 0.95

#python main.py --model AIS_VAE --dataset mnist --binarize True --batch_size 100 --num_samples 1 --max_epochs 30 --step_size 0.01 --K 1 --use_barker False --variance_sensitive_step True --acceptance_rate_target 0.5
#python main.py --model AIS_VAE --dataset mnist --binarize True --batch_size 100 --num_samples 1 --max_epochs 30 --step_size 0.01 --K 1 --use_barker False --variance_sensitive_step True --acceptance_rate_target 0.75

#CUDA_VISIBLE_DEVICES=0 python main.py --model AIS_VAE --dataset mnist --binarize True --batch_size 75 --num_samples 10 --max_epochs 30 --step_size 0.01 --K 1 --use_barker False --variance_sensitive_step True --acceptance_rate_target 0.75 --use_alpha_annealing False --gpus 1
#python main.py --model AIS_VAE --dataset mnist --binarize True --batch_size 100 --num_samples 1 --max_epochs 30 --step_size 0.01 --K 2 --use_barker False --variance_sensitive_step True --acceptance_rate_target 0.75 --use_alpha_annealing False
#python main.py --model AIS_VAE --dataset mnist --binarize True --batch_size 100 --num_samples 1 --max_epochs 30 --step_size 0.01 --K 3 --use_barker False --variance_sensitive_step True --acceptance_rate_target 0.75 --use_alpha_annealing False
#python main.py --model AIS_VAE --dataset mnist --binarize True --batch_size 100 --num_samples 1 --max_epochs 30 --step_size 0.01 --K 5 --use_barker False --variance_sensitive_step True --acceptance_rate_target 0.75 --use_alpha_annealing False

#python main.py --model AIS_VAE --dataset mnist --binarize True --batch_size 100 --num_samples 1 --max_epochs 30 --step_size 0.01 --K 1 --use_barker False --variance_sensitive_step True --acceptance_rate_target 0.99

#python main.py --model ULA_VAE --dataset mnist --act_func gelu --binarize True --hidden_dim 64 --batch_size 100 --net_type conv --num_samples 1 --max_epochs 50 --step_size 0.01 --K 3 --use_score_matching False --variance_sensitive_step True
#
#python main.py --model ULA_VAE --dataset mnist --act_func gelu --binarize True --hidden_dim 64 --batch_size 100 --net_type conv --num_samples 1 --max_epochs 50 --step_size 0.01 --K 1 --use_score_matching False --variance_sensitive_step False
#python main.py --model ULA_VAE --dataset mnist --act_func gelu --binarize True --hidden_dim 64 --batch_size 100 --net_type conv --num_samples 1 --max_epochs 50 --step_size 0.01 --K 3 --use_score_matching False --variance_sensitive_step False


#python main.py --model ULA_VAE --dataset mnist --binarize True --batch_size 100 --num_samples 1 --max_epochs 100 --step_size 0.01 --K 1 --variance_sensitive_step True --ula_skip_threshold 0.1 --acceptance_rate_target 0.9
#python main.py --model ULA_VAE --dataset mnist --binarize True --batch_size 100 --num_samples 1 --max_epochs 100 --step_size 0.01 --K 3 --variance_sensitive_step True --ula_skip_threshold 0.1 --acceptance_rate_target 0.9
#python main.py --model ULA_VAE --dataset mnist --binarize True --batch_size 100 --num_samples 1 --max_epochs 100 --step_size 0.01 --K 5 --variance_sensitive_step True --ula_skip_threshold 0.1 --acceptance_rate_target 0.9
#python main.py --model ULA_VAE --dataset mnist --binarize True --batch_size 100 --num_samples 1 --max_epochs 100 --step_size 0.01 --K 7 --variance_sensitive_step True --ula_skip_threshold 0.1 --acceptance_rate_target 0.9
#python main.py --model ULA_VAE --dataset mnist --binarize True --batch_size 100 --num_samples 1 --max_epochs 100 --step_size 0.01 --K 10 --variance_sensitive_step True --ula_skip_threshold 0.1 --acceptance_rate_target 0.9


#CUDA_VISIBLE_DEVICES=0 python main.py --model ULA_VAE --dataset cifar --binarize False --batch_size 50 --num_samples 1 --max_epochs 100 --step_size 0.01 --K 1 --variance_sensitive_step True --ula_skip_threshold 0.1 --acceptance_rate_target 0.9 --gpus 1
#CUDA_VISIBLE_DEVICES=0 python main.py --model ULA_VAE --dataset cifar --binarize False --batch_size 50 --num_samples 1 --max_epochs 100 --step_size 0.01 --K 3 --variance_sensitive_step True --ula_skip_threshold 0.1 --acceptance_rate_target 0.9 --gpus 1
#CUDA_VISIBLE_DEVICES=0 python main.py --model ULA_VAE --dataset cifar --binarize False --batch_size 50 --num_samples 1 --max_epochs 100 --step_size 0.01 --K 5 --variance_sensitive_step True --ula_skip_threshold 0.1 --acceptance_rate_target 0.9 --gpus 1
#CUDA_VISIBLE_DEVICES=0 python main.py --model ULA_VAE --dataset cifar --binarize False --batch_size 50 --num_samples 1 --max_epochs 100 --step_size 0.01 --K 7 --variance_sensitive_step True --ula_skip_threshold 0.1 --acceptance_rate_target 0.9 --gpus 1
#CUDA_VISIBLE_DEVICES=0 python main.py --model ULA_VAE --dataset cifar --binarize False --batch_size 50 --num_samples 1 --max_epochs 100 --step_size 0.01 --K 10 --variance_sensitive_step True --ula_skip_threshold 0.1 --acceptance_rate_target 0.9 --gpus 1
#
#CUDA_VISIBLE_DEVICES=0 python main.py --model VAE --dataset cifar --act_func gelu --binarize False --hidden_dim 64 --batch_size 25 --net_type conv --num_samples 1 --max_epochs 100 --gpus 1
#CUDA_VISIBLE_DEVICES=0 python main.py --model VAE --dataset celeba --act_func gelu --binarize False --hidden_dim 64 --batch_size 50 --net_type conv --num_samples 1 --max_epochs 100 --limit_train_batches 0.1 --limit_val_batches 0.1 --gpus 1

#python main.py --model ULA_VAE --dataset mnist --binarize True --batch_size 100 --num_samples 1 --max_epochs 100 --step_size 0.01 --K 8 --variance_sensitive_step True --ula_skip_threshold 0.1 --acceptance_rate_target 0.9

#CUDA_VISIBLE_DEVICES=0 python main.py --model VAE --dataset mnist --act_func gelu --binarize True --hidden_dim 2 --batch_size 50 --net_type conv --num_samples 1 --max_epochs 20 --gpus 1
#CUDA_VISIBLE_DEVICES=0 python main.py --model AIS_VAE --dataset mnist --binarize True --batch_size 50 --num_samples 5 --max_epochs 50 --step_size 0.01 --K 1 --use_barker False --variance_sensitive_step True --acceptance_rate_target 0.8 --use_alpha_annealing False --gpus 1 --hidden_dim 2
#CUDA_VISIBLE_DEVICES=0 python main.py --model AIS_VAE --dataset mnist --binarize True --batch_size 50 --num_samples 5 --max_epochs 100 --step_size 0.01 --K 3 --use_barker False --variance_sensitive_step True --acceptance_rate_target 0.8 --use_alpha_annealing False --gpus 1
#CUDA_VISIBLE_DEVICES=0 python main.py --model AIS_VAE --dataset mnist --binarize True --batch_size 50 --num_samples 5 --max_epochs 100 --step_size 0.01 --K 5 --use_barker False --variance_sensitive_step True --acceptance_rate_target 0.8 --use_alpha_annealing False --gpus 1

#CUDA_VISIBLE_DEVICES=1 python main.py --model AIS_VAE --dataset mnist --binarize True --batch_size 50 --num_samples 5 --max_epochs 100 --step_size 0.01 --K 3 --use_barker False --variance_sensitive_step True --acceptance_rate_target 0.8 --use_alpha_annealing False --gpus 1
#CUDA_VISIBLE_DEVICES=1 python main.py --model AIS_VAE --dataset mnist --binarize True --batch_size 50 --num_samples 5 --max_epochs 100 --step_size 0.01 --K 5 --use_barker False --variance_sensitive_step True --acceptance_rate_target 0.8 --use_alpha_annealing False --gpus 1
#CUDA_VISIBLE_DEVICES=1 python main.py --model AIS_VAE --dataset mnist --binarize True --batch_size 50 --num_samples 5 --max_epochs 100 --step_size 0.01 --K 7 --use_barker False --variance_sensitive_step True --acceptance_rate_target 0.8 --use_alpha_annealing False --gpus 1
#CUDA_VISIBLE_DEVICES=1 python main.py --model AIS_VAE --dataset mnist --binarize True --batch_size 50 --num_samples 5 --max_epochs 100 --step_size 0.01 --K 10 --use_barker False --variance_sensitive_step True --acceptance_rate_target 0.8 --use_alpha_annealing False --gpus 1


#CUDA_VISIBLE_DEVICES=0 python main.py --model ULA_VAE --dataset mnist --binarize True --batch_size 50 --num_samples 1 --max_epochs 20 --step_size 0.01 --K 1 --variance_sensitive_step True --ula_skip_threshold 0.25 --acceptance_rate_target 0.75 --hidden_dim 2 --gpus 1


#CUDA_VISIBLE_DEVICES=0 python main.py --model AIS_VAE --dataset mnist --binarize True --batch_size 50 --num_samples 5 --max_epochs 100 --step_size 0.01 --K 10 --use_barker False --variance_sensitive_step True --acceptance_rate_target 0.8 --use_alpha_annealing True --gpus 1


#CUDA_VISIBLE_DEVICES=0 python main.py --model IWAE --dataset mnist --binarize True --batch_size 100 --num_samples 10 --max_epochs 100 --gpus 1
#CUDA_VISIBLE_DEVICES=0 python main.py --model IWAE --dataset mnist --binarize True --batch_size 100 --num_samples 50 --max_epochs 100 --gpus 1


#CUDA_VISIBLE_DEVICES=0 python main.py --model AIS_VAE --dataset cifar --binarize False --hidden_dim 128 --batch_size 50 --num_samples 5 --step_size 0.01 --K 3 --use_barker False --variance_sensitive_step True --acceptance_rate_target 0.8 --max_epochs 100 --gpus 1
#CUDA_VISIBLE_DEVICES=0 python main.py --model AIS_VAE --dataset cifar --binarize False --hidden_dim 128 --batch_size 50 --num_samples 5 --step_size 0.01 --K 5 --use_barker False --variance_sensitive_step True --acceptance_rate_target 0.8 --max_epochs 100 --gpus 1

#CUDA_VISIBLE_DEVICES=0 python main.py --model ULA_VAE --dataset mnist --binarize True --batch_size 50 --num_samples 1 --max_epochs 100 --step_size 0.01 --K 5 --variance_sensitive_step True --ula_skip_threshold 0.5 --acceptance_rate_target 0.99 --annealing_scheme sigmoidal --gpus 1

#CUDA_VISIBLE_DEVICES=1 python main.py --model ULA_VAE --dataset mnist --binarize True --batch_size 50 --num_samples 1 --max_epochs 50 --step_size 0.01 --K 1 --variance_sensitive_step True --ula_skip_threshold 0.1 --acceptance_rate_target 0.9 --annealing_scheme all_learnable --hidden_dim 2 --grad_clip_val 40 --gpus 1

#CUDA_VISIBLE_DEVICES=1 python main.py --model AIS_VAE --dataset mnist --binarize True --hidden_dim 2 --batch_size 50 --num_samples 1 --max_epochs 50 --step_size 0.01 --K 1 --variance_sensitive_step True --use_barker False --acceptance_rate_target 0.8 --annealing_scheme linear --grad_clip_val 20 --gpus 1

#CUDA_VISIBLE_DEVICES=0 python main.py --model VAE --dataset mnist --binarize True --hidden_dim 64 --batch_size 50 --max_epochs 50 --gpus 1
#CUDA_VISIBLE_DEVICES=0 python main.py --model VAE --dataset celeba --binarize False --batch_size 50 --max_epochs 100 --limit_train_batches 0.1 --limit_val_batches 0.1 --gpus 1


CUDA_VISIBLE_DEVICES=0 python main.py --model IWAE --dataset cifar --binarize False --hidden_dim 128 --batch_size 50 --max_epochs 100 --num_samples 10 --gpus 1
CUDA_VISIBLE_DEVICES=0 python main.py --model IWAE --dataset cifar --binarize False --hidden_dim 128 --batch_size 50 --max_epochs 100 --num_samples 50 --gpus 1
CUDA_VISIBLE_DEVICES=0 python main.py --model VAE --dataset cifar --binarize False --hidden_dim 128 --batch_size 50 --max_epochs 100 --gpus 1

CUDA_VISIBLE_DEVICES=0 python main.py --model ULA_VAE --dataset cifar --binarize False --hidden_dim 128 --batch_size 50 --num_samples 1 --max_epochs 100 --step_size 0.01 --K 5 --variance_sensitive_step True --ula_skip_threshold 0.3 --acceptance_rate_target 0.9 --annealing_scheme linear --grad_clip_val 50 --gpus 1
CUDA_VISIBLE_DEVICES=0 python main.py --model ULA_VAE --dataset cifar --binarize False --hidden_dim 128 --batch_size 50 --num_samples 1 --max_epochs 100 --step_size 0.01 --K 5 --variance_sensitive_step True --ula_skip_threshold 0.3 --acceptance_rate_target 0.9 --annealing_scheme sigmoidal --grad_clip_val 50 --gpus 1
CUDA_VISIBLE_DEVICES=0 python main.py --model ULA_VAE --dataset cifar --binarize False --hidden_dim 128 --batch_size 50 --num_samples 1 --max_epochs 100 --step_size 0.01 --K 5 --variance_sensitive_step True --ula_skip_threshold 0.3 --acceptance_rate_target 0.9 --annealing_scheme all_learnable --grad_clip_val 50 --gpus 1
CUDA_VISIBLE_DEVICES=0 python main.py --model ULA_VAE --dataset cifar --binarize False --hidden_dim 128 --batch_size 50 --num_samples 1 --max_epochs 100 --step_size 0.01 --K 10 --variance_sensitive_step True --ula_skip_threshold 0.3 --acceptance_rate_target 0.9 --annealing_scheme linear --grad_clip_val 50 --gpus 1
CUDA_VISIBLE_DEVICES=0 python main.py --model ULA_VAE --dataset cifar --binarize False --hidden_dim 128 --batch_size 50 --num_samples 1 --max_epochs 100 --step_size 0.01 --K 10 --variance_sensitive_step True --ula_skip_threshold 0.3 --acceptance_rate_target 0.9 --annealing_scheme sigmoidal --grad_clip_val 50 --gpus 1
CUDA_VISIBLE_DEVICES=0 python main.py --model ULA_VAE --dataset cifar --binarize False --hidden_dim 128 --batch_size 50 --num_samples 1 --max_epochs 100 --step_size 0.01 --K 10 --variance_sensitive_step True --ula_skip_threshold 0.3 --acceptance_rate_target 0.9 --annealing_scheme all_learnable --grad_clip_val 50 --gpus 1