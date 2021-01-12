## VAE
#python main.py --model VAE --dataset mnist --act_func gelu --binarize True --hidden_dim 64 --batch_size 100 --net_type conv --num_samples 1 --max_epochs 30
#python main.py --model VAE --dataset omniglot --act_func gelu --binarize True --hidden_dim 64 --batch_size 25 --net_type conv --num_samples 1 --max_epochs 30
#python main.py --model VAE --dataset celeba --act_func gelu --binarize False --hidden_dim 64 --batch_size 50 --net_type conv --num_samples 1 --max_epochs 30 --limit_train_batches 0.1 --limit_val_batches 0.1
#
### IWAE
#python main.py --model IWAE --dataset mnist --act_func gelu --binarize True --hidden_dim 64 --batch_size 100 --net_type conv --num_samples 10 --max_epochs 30 --track_grad_norm 2
#python main.py --model IWAE --dataset omniglot --act_func gelu --binarize True --hidden_dim 64 --batch_size 25 --net_type conv --num_samples 10 --max_epochs 30
#python main.py --model IWAE --dataset celeba --act_func gelu --binarize False --hidden_dim 64 --batch_size 50 --net_type conv --num_samples 10 --max_epochs 30 --limit_train_batches 0.1 --limit_val_batches 0.1

## AIWAE
#python main.py --model AIWAE --dataset mnist --act_func gelu --binarize True --use_barker False --hidden_dim 64 --batch_size 100 --net_type conv --num_samples 1 --max_epochs 30 --step_size 0.01 --K 5 --n_leapfrogs 3
#python main.py --model AIWAE --dataset omniglot --act_func gelu --binarize True --use_barker False --hidden_dim 64 --batch_size 25 --net_type conv --num_samples 1 --max_epochs 30 --step_size 0.01 --K 5 --n_leapfrogs 3
#python main.py --model AIWAE --dataset celeba --act_func gelu --binarize False --use_barker False --hidden_dim 64 --batch_size 10 --net_type conv --num_samples 1 --max_epochs 30 --limit_train_batches 0.1 --limit_val_batches 0.1 --step_size 0.01 --K 3 --n_leapfrogs 2

## AIS_VAE
#python main.py --model AIS_VAE --dataset mnist --act_func gelu --binarize True --use_barker False --hidden_dim 64 --batch_size 100 --net_type conv --num_samples 1 --max_epochs 30 --step_size 0.01 --K 2 --track_grad_norm 2 --learnable_transitions False
#python main.py --model AIS_VAE --dataset omniglot --act_func gelu --binarize True --use_barker False --hidden_dim 64 --batch_size 25 --net_type conv --num_samples 1 --max_epochs 30 --step_size 0.01 --K 5
#python main.py --model AIS_VAE --dataset celeba --act_func gelu --binarize False --use_barker False --hidden_dim 64 --batch_size 10 --net_type conv --num_samples 1 --max_epochs 30 --limit_train_batches 0.1 --limit_val_batches 0.1 --step_size 0.01 --K 5

## ULA_VAE
#python main.py --model ULA_VAE --dataset mnist --act_func gelu --binarize True --hidden_dim 64 --batch_size 100 --net_type conv --num_samples 1 --max_epochs 30 --step_size 0.01 --K 1 --use_transforms False
#python main.py --model ULA_VAE --dataset mnist --act_func gelu --binarize True --hidden_dim 64 --batch_size 100 --net_type conv --num_samples 1 --max_epochs 30 --step_size 0.01 --K 5 --use_transforms False --track_grad_norm 2 --learnable_transitions True
#python main.py --model ULA_VAE --dataset mnist --act_func gelu --binarize True --hidden_dim 64 --batch_size 100 --net_type conv --num_samples 1 --max_epochs 30 --step_size 0.01 --K 1 --use_transforms True
#python main.py --model ULA_VAE --dataset mnist --act_func gelu --binarize True --hidden_dim 64 --batch_size 100 --net_type conv --num_samples 1 --max_epochs 30 --step_size 0.01 --K 3 --use_transforms True

## Stacked_VAE
#python main.py --model Stacked_VAE --dataset mnist --act_func gelu --binarize True --use_barker False --hidden_dim 64 --batch_size 100 --net_type conv --num_samples 1 --max_epochs 30 --step_size 0.01 --K 7 --track_grad_norm 2
#python main.py --model Stacked_VAE --dataset omniglot --act_func gelu --binarize True --use_barker False --hidden_dim 64 --batch_size 25 --net_type conv --num_samples 1 --max_epochs 30 --step_size 0.01 --K 7 --track_grad_norm 2
#python main.py --model Stacked_VAE --dataset celeba --act_func gelu --binarize False --use_barker False --hidden_dim 64 --batch_size 50 --net_type conv --num_samples 1 --max_epochs 30 --step_size 0.01 --K 7 --track_grad_norm 2 --limit_train_batches 0.1 --limit_val_batches 0.1

#python main.py --model AIS_VAE --dataset mnist --act_func gelu --binarize True --use_barker False --hidden_dim 64 --batch_size 100 --net_type conv --num_samples 1 --max_epochs 30 --step_size 0.01 --K 1 --track_grad_norm 2 --grad_clip_val 10 --grad_skip_val 50 --use_cloned_decoder False
#python main.py --model AIS_VAE --dataset mnist --act_func gelu --binarize True --use_barker False --hidden_dim 64 --batch_size 100 --net_type conv --num_samples 1 --max_epochs 30 --step_size 0.01 --K 2 --track_grad_norm 2 --grad_clip_val 10 --grad_skip_val 50 --use_cloned_decoder False
#python main.py --model AIS_VAE --dataset mnist --act_func gelu --binarize True --use_barker False --hidden_dim 64 --batch_size 100 --net_type conv --num_samples 1 --max_epochs 30 --step_size 0.01 --K 3 --track_grad_norm 2 --grad_clip_val 10 --grad_skip_val 50 --use_cloned_decoder False
#python main.py --model AIS_VAE --dataset mnist --act_func gelu --binarize True --use_barker False --hidden_dim 64 --batch_size 100 --net_type conv --num_samples 1 --max_epochs 30 --step_size 0.01 --K 5 --track_grad_norm 2 --grad_clip_val 10 --grad_skip_val 50 --use_cloned_decoder False
#python main.py --model AIS_VAE --dataset mnist --act_func gelu --binarize True --use_barker False --hidden_dim 64 --batch_size 100 --net_type conv --num_samples 1 --max_epochs 30 --step_size 0.01 --K 10 --track_grad_norm 2 --grad_clip_val 10 --grad_skip_val 50 --use_cloned_decoder False

#python main.py --model AIS_VAE --dataset mnist --act_func gelu --binarize True --use_barker False --hidden_dim 64 --batch_size 100 --net_type conv --num_samples 1 --max_epochs 30 --step_size 0.01 --K 1 --track_grad_norm 2 --grad_clip_val 10 --grad_skip_val 50 --use_cloned_decoder True
#python main.py --model AIS_VAE --dataset mnist --act_func gelu --binarize True --use_barker False --hidden_dim 64 --batch_size 100 --net_type conv --num_samples 1 --max_epochs 30 --step_size 0.01 --K 3 --track_grad_norm 2 --grad_clip_val 10 --grad_skip_val 50 --use_cloned_decoder True
#python main.py --model AIS_VAE --dataset mnist --act_func gelu --binarize True --use_barker False --hidden_dim 64 --batch_size 100 --net_type conv --num_samples 1 --max_epochs 30 --step_size 0.01 --K 5 --track_grad_norm 2 --grad_clip_val 10 --grad_skip_val 50 --use_cloned_decoder True
#
#python main.py --model AIS_VAE --dataset mnist --act_func gelu --binarize True --use_barker False --hidden_dim 64 --batch_size 100 --net_type conv --num_samples 1 --max_epochs 30 --step_size 0.01 --K 1 --track_grad_norm 2
#python main.py --model AIS_VAE --dataset mnist --act_func gelu --binarize True --use_barker False --hidden_dim 64 --batch_size 100 --net_type conv --num_samples 1 --max_epochs 30 --step_size 0.01 --K 2 --track_grad_norm 2 --use_cloned_decoder True # --grad_clip_val 10 --grad_skip_val 50
#python main.py --model AIS_VAE --dataset mnist --act_func gelu --binarize True --use_barker False --hidden_dim 64 --batch_size 100 --net_type conv --num_samples 1 --max_epochs 30 --step_size 0.01 --K 3 --track_grad_norm 2
#python main.py --model AIS_VAE --dataset mnist --act_func gelu --binarize True --use_barker False --hidden_dim 64 --batch_size 100 --net_type conv --num_samples 1 --max_epochs 30 --step_size 0.01 --K 5 --track_grad_norm 2
#python main.py --model AIS_VAE --dataset mnist --act_func gelu --binarize True --use_barker False --hidden_dim 64 --batch_size 100 --net_type conv --num_samples 1 --max_epochs 30 --step_size 0.01 --K 10 --track_grad_norm 2

#python main.py --model AIS_VAE_S --dataset mnist --act_func gelu --binarize True --use_barker False --hidden_dim 64 --batch_size 100 --net_type conv --num_samples 1 --max_epochs 30 --step_size 0.01 --K 1 --track_grad_norm 2 --grad_clip_val 10 --grad_skip_val 50 --use_cloned_decoder False
#python main.py --model AIS_VAE_S --dataset mnist --act_func gelu --binarize True --use_barker False --hidden_dim 64 --batch_size 100 --net_type conv --num_samples 1 --max_epochs 30 --step_size 0.01 --K 3 --track_grad_norm 2 --grad_clip_val 10 --grad_skip_val 50 --use_cloned_decoder False
#
#python main.py --model AIS_VAE_S --dataset mnist --act_func gelu --binarize True --use_barker False --hidden_dim 64 --batch_size 100 --net_type conv --num_samples 1 --max_epochs 30 --step_size 0.01 --K 1 --track_grad_norm 2 --grad_clip_val 10 --grad_skip_val 50 --use_cloned_decoder True
#python main.py --model AIS_VAE_S --dataset mnist --act_func gelu --binarize True --use_barker False --hidden_dim 64 --batch_size 100 --net_type conv --num_samples 1 --max_epochs 30 --step_size 0.01 --K 3 --track_grad_norm 2 --grad_clip_val 10 --grad_skip_val 50 --use_cloned_decoder True

python main.py --model ULA_VAE --dataset mnist --act_func gelu --binarize True --hidden_dim 64 --batch_size 100 --net_type conv --num_samples 1 --max_epochs 30 --step_size 0.01 --K 2 --use_transforms False --track_grad_norm 2 --learnable_transitions False --use_cloned_decoder True