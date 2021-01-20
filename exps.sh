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


python main.py --model AIS_VAE --dataset mnist --binarize True --batch_size 100 --num_samples 1 --max_epochs 30 --step_size 0.01 --K 1 --use_barker False --variance_sensitive_step True --acceptance_rate_target 0.5
python main.py --model AIS_VAE --dataset mnist --binarize True --batch_size 100 --num_samples 1 --max_epochs 30 --step_size 0.01 --K 1 --use_barker False --variance_sensitive_step True --acceptance_rate_target 0.75
python main.py --model AIS_VAE --dataset mnist --binarize True --batch_size 100 --num_samples 1 --max_epochs 30 --step_size 0.01 --K 1 --use_barker False --variance_sensitive_step True --acceptance_rate_target 0.9
python main.py --model AIS_VAE --dataset mnist --binarize True --batch_size 100 --num_samples 1 --max_epochs 30 --step_size 0.01 --K 1 --use_barker False --variance_sensitive_step True --acceptance_rate_target 0.99

python main.py --model ULA_VAE --dataset mnist --binarize True --batch_size 100 --num_samples 1 --max_epochs 30 --step_size 0.01 --K 1 --variance_sensitive_step True --ula_skip_threshold 0.1 --acceptance_rate_target 0.9
python main.py --model ULA_VAE --dataset mnist --binarize True --batch_size 100 --num_samples 1 --max_epochs 30 --step_size 0.01 --K 1 --variance_sensitive_step True --ula_skip_threshold 0.1 --acceptance_rate_target 0.99
python main.py --model ULA_VAE --dataset mnist --binarize True --batch_size 100 --num_samples 1 --max_epochs 30 --step_size 0.01 --K 1 --variance_sensitive_step True --ula_skip_threshold 0.01 --acceptance_rate_target 0.9
python main.py --model ULA_VAE --dataset mnist --binarize True --batch_size 100 --num_samples 1 --max_epochs 30 --step_size 0.01 --K 1 --variance_sensitive_step True --ula_skip_threshold 0.01 --acceptance_rate_target 0.99

#python main.py --model ULA_VAE --dataset mnist --act_func gelu --binarize True --hidden_dim 64 --batch_size 100 --net_type conv --num_samples 1 --max_epochs 50 --step_size 0.01 --K 3 --use_score_matching False --variance_sensitive_step True
#
#python main.py --model ULA_VAE --dataset mnist --act_func gelu --binarize True --hidden_dim 64 --batch_size 100 --net_type conv --num_samples 1 --max_epochs 50 --step_size 0.01 --K 1 --use_score_matching False --variance_sensitive_step False
#python main.py --model ULA_VAE --dataset mnist --act_func gelu --binarize True --hidden_dim 64 --batch_size 100 --net_type conv --num_samples 1 --max_epochs 50 --step_size 0.01 --K 3 --use_score_matching False --variance_sensitive_step False



