The repository compliments code of 'Monte Carlo VAE' paper.



All the experiments which appear in the paper can be run via the `exps.sh` script. For example:

```bash
python main.py --model L-MCVAE --dataset mnist --act_func gelu --binarize True --hidden_dim 64 --batch_size 100 --net_type conv --num_samples 1 --max_epochs 50 --step_size 0.01 --K 1 --use_transforms True --learnable_transitions False --use_cloned_decoder True
```

For each experiment we can set 

- **model** -- which model we want to train: 'VAE', 'IWAE', 'L-MCVAE' or 'A-MCVAE'

- **dataset** -- which dataset to use (now available MNIST, CIFAR-10, OMNIGLOT, Fashion-MNIST, CelebA[should be downloaded separately]) 

- **act_func** -- activation function

    More arguments with the description are available in `main.py` 



Visualization of toy example's posterior is located in 'toy_example.ipynb' notebook.

Parameter estimates for the toy example can be obtained via in 'run_toy.py'.

