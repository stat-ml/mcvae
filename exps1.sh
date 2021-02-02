#CUDA_VISIBLE_DEVICES=1 python main.py --model ULA_VAE --dataset celeba --binarize False --batch_size 50 --num_samples 1 --max_epochs 100 --step_size 0.01 --K 1 --variance_sensitive_step True --ula_skip_threshold 0.1 --acceptance_rate_target 0.9 --limit_train_batches 0.1 --limit_val_batches 0.1 --gpus 1
#CUDA_VISIBLE_DEVICES=1 python main.py --model ULA_VAE --dataset celeba --binarize False --batch_size 50 --num_samples 1 --max_epochs 100 --step_size 0.01 --K 3 --variance_sensitive_step True --ula_skip_threshold 0.1 --acceptance_rate_target 0.9 --limit_train_batches 0.1 --limit_val_batches 0.1 --gpus 1
#CUDA_VISIBLE_DEVICES=1 python main.py --model ULA_VAE --dataset celeba --binarize False --batch_size 50 --num_samples 1 --max_epochs 100 --step_size 0.01 --K 5 --variance_sensitive_step True --ula_skip_threshold 0.1 --acceptance_rate_target 0.9 --limit_train_batches 0.1 --limit_val_batches 0.1 --gpus 1
#CUDA_VISIBLE_DEVICES=1 python main.py --model ULA_VAE --dataset celeba --binarize False --batch_size 50 --num_samples 1 --max_epochs 100 --step_size 0.01 --K 7 --variance_sensitive_step True --ula_skip_threshold 0.1 --acceptance_rate_target 0.9 --limit_train_batches 0.1 --limit_val_batches 0.1 --gpus 1
#CUDA_VISIBLE_DEVICES=1 python main.py --model ULA_VAE --dataset celeba --binarize False --batch_size 50 --num_samples 1 --max_epochs 100 --step_size 0.01 --K 10 --variance_sensitive_step True --ula_skip_threshold 0.1 --acceptance_rate_target 0.9 --limit_train_batches 0.1 --limit_val_batches 0.1 --gpus 1

#CUDA_VISIBLE_DEVICES=1 python main.py --model IWAE --dataset cifar --act_func gelu --binarize False --hidden_dim 64 --batch_size 50 --net_type conv --num_samples 10 --max_epochs 100 --gpus 1
#CUDA_VISIBLE_DEVICES=1 python main.py --model VAE --dataset celeba --act_func gelu --binarize False --hidden_dim 64 --batch_size 50 --net_type conv --num_samples 1 --max_epochs 100 --limit_train_batches 0.1 --limit_val_batches 0.1 --gpus 1
#CUDA_VISIBLE_DEVICES=1 python main.py --model IWAE --dataset celeba --act_func gelu --binarize False --hidden_dim 64 --batch_size 50 --net_type conv --num_samples 10 --max_epochs 100 --limit_train_batches 0.1 --limit_val_batches 0.1 --gpus 1

#CUDA_VISIBLE_DEVICES=1 python main.py --model ULA_VAE --dataset celeba --binarize False --batch_size 50 --num_samples 1 --max_epochs 30 --step_size 0.01 --K 1 --variance_sensitive_step True --ula_skip_threshold 0.1 --acceptance_rate_target 0.5 --limit_train_batches 0.1 --limit_val_batches 0.1 --gpus 1
#CUDA_VISIBLE_DEVICES=1 python main.py --model ULA_VAE --dataset celeba --binarize False --batch_size 50 --num_samples 1 --max_epochs 30 --step_size 0.01 --K 1 --variance_sensitive_step True --ula_skip_threshold 0.1 --acceptance_rate_target 0.75 --limit_train_batches 0.1 --limit_val_batches 0.1 --gpus 1
#CUDA_VISIBLE_DEVICES=1 python main.py --model ULA_VAE --dataset celeba --binarize False --batch_size 50 --num_samples 1 --max_epochs 30 --step_size 0.01 --K 1 --variance_sensitive_step True --ula_skip_threshold 0.1 --acceptance_rate_target 0.9 --limit_train_batches 0.1 --limit_val_batches 0.1 --gpus 1
#CUDA_VISIBLE_DEVICES=1 python main.py --model ULA_VAE --dataset celeba --binarize False --batch_size 50 --num_samples 1 --max_epochs 30 --step_size 0.01 --K 1 --variance_sensitive_step True --ula_skip_threshold 0.1 --acceptance_rate_target 0.99 --limit_train_batches 0.1 --limit_val_batches 0.1 --gpus 1
#
#CUDA_VISIBLE_DEVICES=1 python main.py --model ULA_VAE --dataset celeba --binarize False --batch_size 50 --num_samples 1 --max_epochs 30 --step_size 0.01 --K 1 --variance_sensitive_step True --ula_skip_threshold 0.25 --acceptance_rate_target 0.5 --limit_train_batches 0.1 --limit_val_batches 0.1 --gpus 1
#CUDA_VISIBLE_DEVICES=1 python main.py --model ULA_VAE --dataset celeba --binarize False --batch_size 50 --num_samples 1 --max_epochs 30 --step_size 0.01 --K 1 --variance_sensitive_step True --ula_skip_threshold 0.25 --acceptance_rate_target 0.75 --limit_train_batches 0.1 --limit_val_batches 0.1 --gpus 1
#CUDA_VISIBLE_DEVICES=1 python main.py --model ULA_VAE --dataset celeba --binarize False --batch_size 50 --num_samples 1 --max_epochs 30 --step_size 0.01 --K 1 --variance_sensitive_step True --ula_skip_threshold 0.25 --acceptance_rate_target 0.9 --limit_train_batches 0.1 --limit_val_batches 0.1 --gpus 1
#CUDA_VISIBLE_DEVICES=1 python main.py --model ULA_VAE --dataset celeba --binarize False --batch_size 50 --num_samples 1 --max_epochs 30 --step_size 0.01 --K 1 --variance_sensitive_step True --ula_skip_threshold 0.25 --acceptance_rate_target 0.99 --limit_train_batches 0.1 --limit_val_batches 0.1 --gpus 1
#
#CUDA_VISIBLE_DEVICES=1 python main.py --model ULA_VAE --dataset celeba --binarize False --batch_size 50 --num_samples 1 --max_epochs 30 --step_size 0.01 --K 1 --variance_sensitive_step True --ula_skip_threshold 0.5 --acceptance_rate_target 0.5 --limit_train_batches 0.1 --limit_val_batches 0.1 --gpus 1
#CUDA_VISIBLE_DEVICES=1 python main.py --model ULA_VAE --dataset celeba --binarize False --batch_size 50 --num_samples 1 --max_epochs 30 --step_size 0.01 --K 1 --variance_sensitive_step True --ula_skip_threshold 0.5 --acceptance_rate_target 0.75 --limit_train_batches 0.1 --limit_val_batches 0.1 --gpus 1
#CUDA_VISIBLE_DEVICES=1 python main.py --model ULA_VAE --dataset celeba --binarize False --batch_size 50 --num_samples 1 --max_epochs 30 --step_size 0.01 --K 1 --variance_sensitive_step True --ula_skip_threshold 0.5 --acceptance_rate_target 0.9 --limit_train_batches 0.1 --limit_val_batches 0.1 --gpus 1
#CUDA_VISIBLE_DEVICES=1 python main.py --model ULA_VAE --dataset celeba --binarize False --batch_size 50 --num_samples 1 --max_epochs 30 --step_size 0.01 --K 1 --variance_sensitive_step True --ula_skip_threshold 0.5 --acceptance_rate_target 0.99 --limit_train_batches 0.1 --limit_val_batches 0.1 --gpus 1

#CUDA_VISIBLE_DEVICES=1 python main.py --model AIS_VAE --dataset mnist --binarize True --batch_size 50 --num_samples 5 --max_epochs 100 --step_size 0.01 --K 5 --use_barker False --variance_sensitive_step True --acceptance_rate_target 0.8 --use_alpha_annealing False --gpus 1

#CUDA_VISIBLE_DEVICES=0 python main.py --model AIS_VAE --dataset mnist --binarize True --batch_size 50 --num_samples 5 --max_epochs 100 --step_size 0.01 --K 5 --use_barker False --variance_sensitive_step True --acceptance_rate_target 0.8 --use_alpha_annealing False --hidden_dim 2 --gpus 1
#CUDA_VISIBLE_DEVICES=0 python main.py --model AIS_VAE --dataset mnist --binarize True --batch_size 50 --num_samples 5 --max_epochs 100 --step_size 0.01 --K 5 --use_barker False --variance_sensitive_step True --acceptance_rate_target 0.8 --use_alpha_annealing False --hidden_dim 2 --gpus 1

#CUDA_VISIBLE_DEVICES=1 python main.py --model AIS_VAE --dataset mnist --binarize True --batch_size 50 --num_samples 5 --max_epochs 50 --step_size 0.01 --K 1 --use_barker False --variance_sensitive_step True --acceptance_rate_target 0.8 --use_alpha_annealing True --gpus 1 --hidden_dim 2
#CUDA_VISIBLE_DEVICES=1 python main.py --model AIS_VAE --dataset mnist --binarize True --batch_size 50 --num_samples 5 --max_epochs 100 --step_size 0.01 --K 3 --use_barker False --variance_sensitive_step True --acceptance_rate_target 0.8 --use_alpha_annealing True --gpus 1
#CUDA_VISIBLE_DEVICES=1 python main.py --model AIS_VAE --dataset mnist --binarize True --batch_size 50 --num_samples 5 --max_epochs 100 --step_size 0.01 --K 5 --use_barker False --variance_sensitive_step True --acceptance_rate_target 0.8 --use_alpha_annealing True --gpus 1


#CUDA_VISIBLE_DEVICES=1 python main.py --model IWAE --dataset mnist --act_func gelu --binarize True --hidden_dim 2 --batch_size 50 --net_type conv --num_samples 10 --max_epochs 20 --gpus 1
#CUDA_VISIBLE_DEVICES=1 python main.py --model ULA_VAE --dataset mnist --binarize True --batch_size 50 --num_samples 1 --max_epochs 20 --step_size 0.01 --K 2 --variance_sensitive_step True --ula_skip_threshold 0.25 --acceptance_rate_target 0.9 --hidden_dim 2 --gpus 1
#CUDA_VISIBLE_DEVICES=0 python main.py --model ULA_VAE --dataset cifar --binarize False --batch_size 50 --num_samples 1 --max_epochs 100 --step_size 0.01 --K 5 --variance_sensitive_step True --ula_skip_threshold 0.1 --acceptance_rate_target 0.9 --gpus 1



#CUDA_VISIBLE_DEVICES=1 python main.py --model IWAE --dataset mnist --binarize True --batch_size 100 --num_samples 50 --max_epochs 100 --gpus 1
#CUDA_VISIBLE_DEVICES=1 python main.py --model ULA_VAE --dataset mnist --binarize True --batch_size 100 --num_samples 5 --max_epochs 100 --step_size 0.01 --K 1 --variance_sensitive_step True --ula_skip_threshold 0.1 --acceptance_rate_target 0.9 --gpus 1




#CUDA_VISIBLE_DEVICES=1 python main.py --model VAE --dataset cifar --binarize False --hidden_dim 128 --batch_size 50 --max_epochs 100 --gpus 1
#
#CUDA_VISIBLE_DEVICES=1 python main.py --model IWAE --dataset cifar --binarize False --hidden_dim 128 --batch_size 50 --num_samples 10 --max_epochs 100 --gpus 1
#CUDA_VISIBLE_DEVICES=1 python main.py --model IWAE --dataset cifar --binarize False --hidden_dim 128 --batch_size 50 --num_samples 50 --max_epochs 100 --gpus 1

#CUDA_VISIBLE_DEVICES=1 python main.py --model ULA_VAE --dataset cifar --binarize False --hidden_dim 128 --batch_size 50 --step_size 0.01 --K 5 --variance_sensitive_step True --ula_skip_threshold 0.3 --acceptance_rate_target 0.9 --max_epochs 100 --grad_clip_val 20 --gpus 1
#CUDA_VISIBLE_DEVICES=1 python main.py --model ULA_VAE --dataset cifar --binarize False --hidden_dim 128 --batch_size 50 --step_size 0.01 --K 10 --variance_sensitive_step True --ula_skip_threshold 0.3 --acceptance_rate_target 0.9 --max_epochs 100 --grad_clip_val 40 --gpus 1

#CUDA_VISIBLE_DEVICES=1 python main.py --model AIS_VAE --dataset cifar --binarize False --hidden_dim 128 --batch_size 50 --num_samples 5 --step_size 0.01 --K 3 --use_barker False --variance_sensitive_step True --acceptance_rate_target 0.8 --max_epochs 100 --gpus 1
#CUDA_VISIBLE_DEVICES=1 python main.py --model AIS_VAE --dataset cifar --binarize False --hidden_dim 128 --batch_size 50 --num_samples 5 --step_size 0.01 --K 5 --use_barker False --variance_sensitive_step True --acceptance_rate_target 0.8 --max_epochs 100 --gpus 1

#CUDA_VISIBLE_DEVICES=1 python main.py --model IWAE --dataset celeba --act_func gelu --binarize False --hidden_dim 64 --batch_size 50 --net_type conv --num_samples 50 --max_epochs 100 --limit_train_batches 0.1 --limit_val_batches 0.1 --gpus 1

#CUDA_VISIBLE_DEVICES=1 python main.py --model ULA_VAE --dataset mnist --binarize True --batch_size 50 --num_samples 1 --max_epochs 100 --step_size 0.01 --K 5 --variance_sensitive_step True --ula_skip_threshold 0.5 --acceptance_rate_target 0.99 --annealing_scheme all_learnable --gpus 1

#CUDA_VISIBLE_DEVICES=1 python main.py --model AIS_VAE --dataset mnist --binarize True --batch_size 50 --num_samples 5 --max_epochs 50 --step_size 0.01 --K 3 --use_barker False --variance_sensitive_step True --acceptance_rate_target 0.8 --annealing_scheme all_learnable --hidden_dim 2 --gpus 1

#CUDA_VISIBLE_DEVICES=1 python main.py --model ULA_VAE --dataset celeba --binarize False --batch_size 50 --num_samples 1 --max_epochs 100 --step_size 0.01 --K 10 --variance_sensitive_step True --ula_skip_threshold 0.1 --acceptance_rate_target 0.95 --limit_train_batches 0.1 --limit_val_batches 0.1 --grad_clip_val 40 --gpus 1

#CUDA_VISIBLE_DEVICES=1 python main.py --model VAE --dataset cifar --binarize False --hidden_dim 128 --batch_size 50 --max_epochs 100 --gpus 1


CUDA_VISIBLE_DEVICES=1 python main.py --model AIS_VAE --dataset cifar --binarize False --hidden_dim 128 --batch_size 50 --num_samples 5 --max_epochs 100 --step_size 0.01 --K 3 --variance_sensitive_step True --use_barker False --acceptance_rate_target 0.8 --annealing_scheme linear --gpus 1
CUDA_VISIBLE_DEVICES=1 python main.py --model AIS_VAE --dataset cifar --binarize False --hidden_dim 128 --batch_size 50 --num_samples 5 --max_epochs 100 --step_size 0.01 --K 5 --variance_sensitive_step True --use_barker False --acceptance_rate_target 0.8 --annealing_scheme linear --gpus 1