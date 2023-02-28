# Monte Carlo Variational Auto-Encoders

The repository compliments code of 'Monte Carlo Variational Auto-Encoders' paper.

In this paper we introduce new objectives for training a VAE. These objectives are inspired by the exactness of MCMC methods and its specific application in annealing importance sampling.

The posteriors we received using our approach is very flexible, allowing us to learn complicated shapes, not feasible to parametric approaches. The results on the toy example presented below:

<p align="center">
  <img width="400" alt="Resulting posteriors" src="https://github.com/stat-ml/mcvae/blob/master/pics/different_approximations.jpg?raw=true">
</p>


In image datasets, in particular MNIST, our new objectives and induced posterior approximations outperform other approaches by a large margin:

<p align="center">
  <img width="400" alt="Likelihood Comparison" src="https://github.com/stat-ml/mcvae/blob/master/pics/different_likelihoods.jpg?raw=true">
</p>


Results on other image datasets are presented below:

<p align="center">
  <img width="400" alt="Results on different datasets" src="https://github.com/stat-ml/mcvae/blob/master/pics/results.jpg?raw=true">
</p>



# How to run the code?

All the experiments which appear in the paper can be run via the `exps.sh` script. For example:

```bash
python main.py --model L-MCVAE --dataset mnist --act_func gelu --binarize True --hidden_dim 64 --batch_size 100 --net_type conv --num_samples 1 --max_epochs 50 --step_size 0.01 --K 1 --use_transforms True --learnable_transitions False --use_cloned_decoder True
```

For each experiment we can set 

- **model** -- which model we want to train: 'VAE', 'IWAE', 'L-MCVAE' or 'A-MCVAE'

- **dataset** -- which dataset to use (now available MNIST, CIFAR-10, OMNIGLOT, Fashion-MNIST, CelebA[should be downloaded separately]) 

- **act_func** -- activation function

    More arguments with the description are available in `main.py` 



# Citation

The original paper can be found [here](http://proceedings.mlr.press/v139/thin21a/thin21a.pdf). If you use **MCVAE**, we kindly ask you to cite:

```
@inproceedings{thin2021monte,
  title={Monte Carlo variational auto-encoders},
  author={Thin, Achille and Kotelevskii, Nikita and Doucet, Arnaud and Durmus, Alain and Moulines, Eric and Panov, Maxim},
  booktitle={International Conference on Machine Learning},
  pages={10247--10257},
  year={2021},
  organization={PMLR}
}
```


