from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from models import VAE, IWAE, AIS_VAE, ULA_VAE, Stacked_VAE
from utils import make_dataloaders, get_activations, str2bool

if __name__ == '__main__':
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    tb_logger = pl_loggers.TensorBoardLogger('lightning_logs/')

    parser.add_argument("--model", default="Stacked_VAE",
                        choices=["VAE", "IWAE", "AIS_VAE", "ULA_VAE", "Stacked_VAE"])

    ## Dataset params
    parser.add_argument("--dataset", default='mnist', choices=['mnist', 'fashionmnist', 'cifar', 'omniglot', 'celeba'])
    parser.add_argument("--binarize", type=str2bool, default=False)
    ## Training parameters
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--val_batch_size", default=50, type=int)
    parser.add_argument("--grad_skip_val", type=float, default=0.)
    parser.add_argument("--grad_clip_val", type=float, default=0.)

    ## Architecture
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--num_samples", default=1, type=int)
    parser.add_argument("--act_func", default="gelu",
                        choices=["relu", "leakyrelu", "tanh", "logsigmoid", "logsoftmax", "softplus", "gelu"])
    parser.add_argument("--net_type", choices=["fc", "conv"], type=str, default="conv")

    ## Specific parameters
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--n_leapfrogs", type=int, default=3)
    parser.add_argument("--step_size", type=float, default=0.01)

    parser.add_argument("--use_barker", type=str2bool, default=False)
    parser.add_argument("--use_score_matching", type=str2bool, default=False)  # for ULA
    parser.add_argument("--use_cloned_decoder", type=str2bool,
                        default=False)  # for AIS VAE (to make grad throught alphas easier)
    parser.add_argument("--learnable_transitions", type=str2bool,
                        default=False)  # for AIS VAE and ULA (if learn stepsize or not)
    parser.add_argument("--variance_sensitive_step", type=str2bool,
                        default=False)  # for AIS VAE and ULA (adapt stepsize based on dim's variance)
    parser.add_argument("--use_alpha_annealing", type=str2bool,
                        default=False)  # for AIS VAE, True if we want anneal sum_log_alphas during training
    parser.add_argument("--annealing_scheme", type=str,
                        default='linear')  # for AIS VAE and ULA VAE, strategy to do annealing
    parser.add_argument("--specific_likelihood", type=str,
                        default=None)  # specific likelihood

    parser.add_argument("--ula_skip_threshold", type=float,
                        default=0.0)  # Probability threshold, if below -- skip transition
    parser.add_argument("--acceptance_rate_target", type=float,
                        default=0.95)  # Target acceptance rate
    parser.add_argument("--sigma", type=float, default=1.)

    act_func = get_activations()

    args = parser.parse_args()
    print(args)

    kwargs = {'num_workers': 20, 'pin_memory': True}
    train_loader, val_loader = make_dataloaders(dataset=args.dataset,
                                                batch_size=args.batch_size,
                                                val_batch_size=args.val_batch_size,
                                                binarize=args.binarize,
                                                **kwargs)
    image_shape = train_loader.dataset.shape_size
    if args.model == "VAE":
        model = VAE(shape=image_shape, act_func=act_func[args.act_func],
                    num_samples=args.num_samples, hidden_dim=args.hidden_dim,
                    net_type=args.net_type, dataset=args.dataset, specific_likelihood=args.specific_likelihood,
                    sigma=args.sigma)
    elif args.model == "IWAE":
        model = IWAE(shape=image_shape, act_func=act_func[args.act_func], num_samples=args.num_samples,
                     hidden_dim=args.hidden_dim,
                     name=args.model, net_type=args.net_type, dataset=args.dataset,
                     specific_likelihood=args.specific_likelihood, sigma=args.sigma)
    elif args.model == 'AIS_VAE':
        model = AIS_VAE(shape=image_shape, step_size=args.step_size, K=args.K, use_barker=args.use_barker,
                        num_samples=args.num_samples, acceptance_rate_target=args.acceptance_rate_target,
                        dataset=args.dataset, net_type=args.net_type, act_func=act_func[args.act_func],
                        hidden_dim=args.hidden_dim, name=args.model, grad_skip_val=args.grad_skip_val,
                        grad_clip_val=args.grad_clip_val,
                        use_cloned_decoder=args.use_cloned_decoder, learnable_transitions=args.learnable_transitions,
                        variance_sensitive_step=args.variance_sensitive_step,
                        use_alpha_annealing=args.use_alpha_annealing, annealing_scheme=args.annealing_scheme,
                        specific_likelihood=args.specific_likelihood, sigma=args.sigma)
    elif args.model == 'ULA_VAE':
        model = ULA_VAE(shape=image_shape, step_size=args.step_size, K=args.K,
                        num_samples=args.num_samples, acceptance_rate_target=args.acceptance_rate_target,
                        dataset=args.dataset, net_type=args.net_type, act_func=act_func[args.act_func],
                        hidden_dim=args.hidden_dim, name=args.model, grad_skip_val=args.grad_skip_val,
                        grad_clip_val=args.grad_clip_val, use_score_matching=args.use_score_matching,
                        use_cloned_decoder=args.use_cloned_decoder, learnable_transitions=args.learnable_transitions,
                        variance_sensitive_step=args.variance_sensitive_step,
                        ula_skip_threshold=args.ula_skip_threshold, annealing_scheme=args.annealing_scheme,
                        specific_likelihood=args.specific_likelihood, sigma=args.sigma)
    elif args.model == 'Stacked_VAE':
        model = Stacked_VAE(shape=image_shape, act_func=act_func[args.act_func], num_samples=args.num_samples,
                            hidden_dim=args.hidden_dim,
                            name=args.model, net_type=args.net_type, dataset=args.dataset, step_size=args.step_size,
                            K=args.K, use_barker=args.use_barker, n_first_iterations=5, first_model='VAE',
                            second_model='AIS_VAE')
    else:
        raise ValueError

    args.gradient_clip_val = args.grad_clip_val
    automatic_optimization = (args.grad_skip_val == 0.) and (args.gradient_clip_val == 0.)

    trainer = pl.Trainer.from_argparse_args(args, logger=tb_logger, fast_dev_run=False,
                                            terminate_on_nan=automatic_optimization,
                                            automatic_optimization=automatic_optimization)
    pl.Trainer()
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)
