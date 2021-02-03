#CUDA_VISIBLE_DEVICES=0 python main.py --model VAE --dataset mnist --binarize True --hidden_dim 100 --net_type fc --batch_size 100 --max_epochs 10 --gpus 1

CUDA_VISIBLE_DEVICES=0 python main.py --model IWAE --dataset mnist --binarize True --hidden_dim 100 --net_type fc --batch_size 100 --max_epochs 100 --num_samples 50 --specific_likelihood gaussian --sigma 0.1 --gpus 1

#CUDA_VISIBLE_DEVICES=0 python main.py --model ULA_VAE --dataset mnist --binarize True --hidden_dim 100 --net_type fc --batch_size 100 --max_epochs 10 --num_samples 1 --step_size 0.01 --K 5 --variance_sensitive_step True --ula_skip_threshold 0.3 --acceptance_rate_target 0.9 --annealing_scheme linear --gpus 1

#CUDA_VISIBLE_DEVICES=0 python main.py --model AIS_VAE --dataset mnist --binarize True --hidden_dim 100 --net_type fc --batch_size 100 --max_epochs 10 --num_samples 5 --step_size 0.01 --K 3 --variance_sensitive_step True --use_barker False --acceptance_rate_target 0.8 --annealing_scheme linear --gpus 1