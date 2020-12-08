## VAE
#python main.py --model VAE --dataset mnist --act_func leakyrelu --binarize True --use_barker False --hidden_dim 64 --batch_size 100 --net_type conv --num_samples 1 --max_epochs 30 --track_grad_norm 2
#python main.py --model VAE --dataset omniglot --act_func leakyrelu --binarize True --use_barker False --hidden_dim 64 --batch_size 25 --net_type conv --num_samples 1 --max_epochs 30
#python main.py --model VAE --dataset celeba --act_func leakyrelu --binarize False --use_barker False --hidden_dim 64 --batch_size 50 --net_type conv --num_samples 1 --max_epochs 30 --limit_train_batches 0.1 --limit_val_batches 0.1
#
### IWAE
#python main.py --model IWAE --dataset mnist --act_func leakyrelu --binarize True --use_barker False --hidden_dim 64 --batch_size 100 --net_type conv --num_samples 10 --max_epochs 30 --track_grad_norm 2
#python main.py --model IWAE --dataset omniglot --act_func leakyrelu --binarize True --use_barker False --hidden_dim 64 --batch_size 25 --net_type conv --num_samples 10 --max_epochs 30
#python main.py --model IWAE --dataset celeba --act_func leakyrelu --binarize False --use_barker False --hidden_dim 64 --batch_size 50 --net_type conv --num_samples 10 --max_epochs 30 --limit_train_batches 0.1 --limit_val_batches 0.1

## AIWAE
#python main.py --model AIWAE --dataset mnist --act_func leakyrelu --binarize True --use_barker False --hidden_dim 64 --batch_size 100 --net_type conv --num_samples 1 --max_epochs 30 --step_size 0.01 --K 5 --n_leapfrogs 3
#python main.py --model AIWAE --dataset omniglot --act_func leakyrelu --binarize True --use_barker False --hidden_dim 64 --batch_size 25 --net_type conv --num_samples 1 --max_epochs 30 --step_size 0.01 --K 5 --n_leapfrogs 3
#python main.py --model AIWAE --dataset celeba --act_func leakyrelu --binarize False --use_barker False --hidden_dim 64 --batch_size 10 --net_type conv --num_samples 1 --max_epochs 30 --limit_train_batches 0.1 --limit_val_batches 0.1 --step_size 0.01 --K 3 --n_leapfrogs 2

## AIS_VAE
#python main.py --model AIS_VAE --dataset mnist --act_func leakyrelu --binarize True --use_barker False --hidden_dim 64 --batch_size 100 --net_type conv --num_samples 1 --max_epochs 30 --step_size 0.01 --K 5 --track_grad_norm 2
#python main.py --model AIS_VAE --dataset omniglot --act_func leakyrelu --binarize True --use_barker False --hidden_dim 64 --batch_size 25 --net_type conv --num_samples 1 --max_epochs 30 --step_size 0.01 --K 5
#python main.py --model AIS_VAE --dataset celeba --act_func leakyrelu --binarize False --use_barker False --hidden_dim 64 --batch_size 10 --net_type conv --num_samples 1 --max_epochs 30 --limit_train_batches 0.1 --limit_val_batches 0.1 --step_size 0.01 --K 5

## ULA_VAE
#python main.py --model ULA_VAE --dataset mnist --act_func leakyrelu --binarize True --use_barker False --hidden_dim 64 --batch_size 100 --net_type conv --num_samples 1 --max_epochs 30 --step_size 0.01 --K 5 --track_grad_norm 2
#python main.py --model ULA_VAE --dataset omniglot --act_func leakyrelu --binarize True --use_barker False --hidden_dim 64 --batch_size 25 --net_type conv --num_samples 1 --max_epochs 30 --step_size 0.01 --K 5
python main.py --model ULA_VAE --dataset celeba --act_func leakyrelu --binarize False --use_barker False --hidden_dim 64 --batch_size 50 --net_type conv --num_samples 1 --max_epochs 30 --limit_train_batches 0.1 --limit_val_batches 0.1 --step_size 0.01 --step_size 0.01 --K 5