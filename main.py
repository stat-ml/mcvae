from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from models import VAE, IWAE
from utils import make_dataloaders, get_activations

if __name__ == '__main__':
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    tb_logger = pl_loggers.TensorBoardLogger('lightning_logs/')
    parser.add_argument("--model", default="IWAE", choices=["VAE", "IWAE"])
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--val_batch_size", default=50, type=int)
    parser.add_argument("--hidden_dim", default=2, type=int)
    parser.add_argument("--dataset", default='fashionmnist', choices=['mnist', 'fashionmnist', 'cifar'])
    parser.add_argument("--num_samples", default=1, type=int)
    parser.add_argument("--act_func", default="tanh", choices=["relu", "leakyrelu", "tanh", "logsigmoid", "logsoftmax"])
    act_func = get_activations()

    args = parser.parse_args()
    args.gpus = 1

    kwargs = {'num_workers': 20, 'pin_memory': True} if args.gpus else {}
    train_loader, val_loader = make_dataloaders(dataset=args.dataset,
                                                batch_size=args.batch_size,
                                                val_batch_size=args.val_batch_size,
                                                **kwargs)
    if args.model == "VAE":
        model = VAE(act_func=act_func[args.act_func], num_samples=args.num_samples, hidden_dim=args.hidden_dim)
    elif args.model == "IWAE":
        model = IWAE(act_func=act_func[args.act_func], num_samples=args.num_samples, hidden_dim=args.hidden_dim,
                     name=args.model)
    else:
        raise ValueError

    trainer = pl.Trainer.from_argparse_args(args, logger=tb_logger, deterministic=True, max_epochs=150,
                                            fast_dev_run=True)
    pl.Trainer()
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)
    trainer.save_checkpoint(f"./checkpoints/{args.model}_{args.hidden_dim}.ckpt")
