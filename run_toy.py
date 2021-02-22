import json

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import Dataset, DataLoader

from models.vaes import Base, VAE, IWAE, AMCVAE, LMCVAE, VAE_with_flows

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def repeat_data(x, n_samples):
    '''
    Repeats data n_samples times, taking dimensionality into account
    '''
    if len(x.shape) == 4:
        x = x.repeat(n_samples, 1, 1, 1)
    else:
        x = x.repeat(n_samples, 1)
    return x


use_true = False


def generate_dataset(N, d=2, sigma=1.):
    z = np.random.randn(N, d)
    x = 2 * np.pi * (np.linalg.norm(z, axis=1, keepdims=True) + 2.) + np.random.randn(N, 1) * sigma
    return x


def replace_enc_dec(model):
    model.encoder_net = ToyEncoder()
    model.decoder_net = ToyDecoder()
    model = model.to(device)
    return model


class Toy(Base):
    def joint_logdensity(self, use_true_decoder=None):
        def density(z, x):
            log_Pr = torch.distributions.Normal(loc=torch.tensor(0., device=x.device, dtype=torch.float32),
                                                scale=torch.tensor(1., device=x.device, dtype=torch.float32)).log_prob(
                z).sum(-1)

            return torch.distributions.Normal(loc=self.decoder_net.alpha * (
                    torch.sqrt(torch.sum(torch.pow(z, 2), dim=1, keepdim=True)) + self.decoder_net.beta),
                                              scale=self.decoder_net.sigma).log_prob(x).sum(
                -1) + log_Pr

        return density


class VAE_Toy(VAE, Toy):
    def step(self, batch):
        x, _ = batch
        z, mu, logvar = self.enc_rep(x, self.num_samples)
        x = repeat_data(x, self.num_samples)
        loss = self.loss_function(z, x, mu, logvar)
        return loss, None, z

    def loss_function(self, z, x, mu, logvar):
        batch_size = mu.shape[0] // self.num_samples
        loglikelihood = torch.distributions.Normal(loc=self.decoder_net.alpha * (
                torch.sqrt(torch.sum(torch.pow(z, 2), dim=1, keepdim=True)) + self.decoder_net.beta),
                                                   scale=self.decoder_net.sigma).log_prob(x).view(
            (self.num_samples, batch_size, -1)).mean(0).sum(-1).mean()
        KLD = -0.5 * torch.mean((1 + logvar - mu.pow(2) - logvar.exp()).view(
            (self.num_samples, -1, self.hidden_dim)).mean(0).sum(-1))
        loss = -loglikelihood + KLD
        return loss


class IWAE_Toy(IWAE, Toy):
    def loss_function(self, recon_x, x, mu, logvar, z):
        batch_size = mu.shape[0] // self.num_samples
        self.hidden_dim = mu.shape[1]
        log_Q = torch.distributions.Normal(loc=mu,
                                           scale=torch.exp(0.5 * logvar)).log_prob(z).view(
            (self.num_samples, -1, self.hidden_dim)).sum(-1)

        log_Pr = torch.sum((-0.5 * torch.abs(z).pow(2.)).view((self.num_samples, -1, self.hidden_dim)), -1)
        loglikelihood = torch.distributions.Normal(loc=self.decoder_net.alpha * (
                torch.sqrt(torch.sum(torch.pow(z, 2), dim=1, keepdim=True)) + self.decoder_net.beta),
                                                   scale=self.decoder_net.sigma).log_prob(x).view(
            (self.num_samples, batch_size, -1)).sum(-1)

        log_weight = log_Pr + loglikelihood - log_Q
        log_weight = log_weight - torch.max(log_weight, 0)[0]  # for stability
        weight = torch.exp(log_weight)
        weight = weight / torch.sum(weight, 0)
        weight = weight.detach()
        loss = torch.mean(torch.sum(weight * (-log_Pr - loglikelihood + log_Q), 0))
        return loss


class VAE_with_flows_Toy(VAE_with_flows, Toy):
    def loss_function(self, recon_x, x, mu, logvar, z, z_transformed, log_jac):
        batch_size = mu.shape[0] // self.num_samples
        loglikelihood = torch.distributions.Normal(loc=self.decoder_net.alpha * (
                torch.sqrt(torch.sum(torch.pow(z_transformed, 2), dim=1, keepdim=True)) + self.decoder_net.beta),
                                                   scale=self.decoder_net.sigma).log_prob(x).view(
            (self.num_samples, batch_size, -1)).mean(0).sum(-1).mean()
        log_Q = torch.mean(torch.distributions.Normal(loc=mu, scale=torch.exp(0.5 * logvar)).log_prob(z).view(
            (self.num_samples, batch_size, -1)).sum(-1) - log_jac.view((self.num_samples, -1)), dim=0).mean()
        log_Pr = (-0.5 * z_transformed ** 2).view(
            (self.num_samples, batch_size, -1)).mean(0).sum(-1).mean()
        KLD = log_Q - log_Pr
        loss = -loglikelihood + KLD
        return loss


class LMCVAE_Toy(LMCVAE, Toy):
    def loss_function(self, sum_log_weights):
        loss = super(LMCVAE_Toy, self).loss_function(sum_log_weights)
        return loss


class AMCVAE_Toy(AMCVAE, Toy):
    def loss_function(self, sum_log_alphas, sum_log_weights):
        loss = super(AMCVAE_Toy, self).loss_function(sum_log_alphas, sum_log_weights)
        return loss


class ToyDataset(Dataset):
    def __init__(self, data):
        super(ToyDataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sample = torch.tensor(self.data[item], dtype=torch.float32, device=device)
        return sample, -1.


class ToyEncoder(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.aux = nn.Parameter(torch.tensor(0., dtype=torch.float32))

    def forward(self, x):
        return torch.zeros(x.shape[0], 2 * d, device=x.device, dtype=torch.float32) + self.aux * 0.


class ToyEncoder_VB(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.var_mean = nn.Parameter(torch.zeros(d, dtype=torch.float32, device=device))
        self.log_var_z = nn.Parameter(torch.zeros(d, dtype=torch.float32, device=device))

    def forward(self, x):
        ol = torch.ones(x.shape[0], self.log_var_z.shape[0], device=device)
        return torch.cat([self.var_mean * ol, self.log_var_z * ol], dim=1)


class ToyDecoder(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.tensor(2., device=device, dtype=torch.float32))
        self.log_beta = nn.Parameter(torch.tensor(2., device=device, dtype=torch.float32))
        self.sigma = sigma

    @property
    def alpha(self, ):
        return torch.exp(self.log_alpha)

    @property
    def beta(self, ):
        return torch.exp(self.log_beta)

    def forward(self, x):
        return self.alpha * (
                torch.sqrt(torch.sum(torch.pow(x, 2), dim=1, keepdim=True)) + self.beta)


def run_trainer(model, num_epoches=21):
    tb_logger = pl_loggers.TensorBoardLogger('lightning_logs/')
    trainer = pl.Trainer(logger=tb_logger, fast_dev_run=False, max_epochs=num_epoches, automatic_optimization=True, )
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)


def compute_discrepancy(model):
    with torch.no_grad():
        return (model.decoder_net.alpha - 2 * np.pi) ** 2 + (model.decoder_net.beta - 2.) ** 2


def get_alpha(model):
    with torch.no_grad():
        return model.decoder_net.alpha


def get_beta(model):
    with torch.no_grad():
        return model.decoder_net.beta


if __name__ == '__main__':
    N = 10000
    sigma = 1.
    full_results = {'VAE': [], 'IWAE': [], 'L-MCVAE': [], 'A-MCVAE': [], 'RealNVP': []}
    full_alphas = {'VAE': [], 'IWAE': [], 'L-MCVAE': [], 'A-MCVAE': [], 'RealNVP': []}
    full_betas = {'VAE': [], 'IWAE': [], 'L-MCVAE': [], 'A-MCVAE': [], 'RealNVP': []}
    for d in [2, 5, 10, 20, 50, 100, 150, 200, 300]:
        print(f'Current dimension is {d}')
        # ----- VAE ------ #
        vae = VAE_Toy(shape=28, act_func=nn.LeakyReLU,
                      num_samples=1, hidden_dim=d,
                      net_type='conv', dataset='toy')
        vae = replace_enc_dec(vae)
        vae.encoder_net = ToyEncoder_VB(d=d).to(device)

        # ----- IWAE ------ #
        iwae = IWAE_Toy(shape=28, act_func=nn.LeakyReLU,
                        num_samples=5, hidden_dim=d,
                        net_type='conv', dataset='toy')
        iwae = replace_enc_dec(iwae)
        iwae.name = 'IWAE'
        iwae.encoder_net = ToyEncoder_VB(d=d).to(device)

        # ----- LMCVAE ----- #
        ula_vae = LMCVAE_Toy(shape=28, act_func=nn.LeakyReLU,
                              num_samples=1, hidden_dim=d,
                              net_type='conv', dataset='toy',
                              step_size=0.01, K=10, use_transforms=False, learnable_transitions=False,
                              return_pre_alphas=True, use_score_matching=False,
                              ula_skip_threshold=0.1, grad_skip_val=0., grad_clip_val=0., use_cloned_decoder=False,
                              variance_sensitive_step=True,
                              acceptance_rate_target=0.9, annealing_scheme='linear')
        ula_vae = replace_enc_dec(ula_vae)
        ula_vae.name = 'LMCVAE'

        # ----- AMCVAE ----- #
        ais_vae = AMCVAE_Toy(shape=28, act_func=nn.LeakyReLU,
                              num_samples=5, hidden_dim=d,
                              net_type='conv', dataset='toy',
                              step_size=0.01, K=10, use_barker=False, learnable_transitions=False,
                              use_alpha_annealing=True, grad_skip_val=0.,
                              grad_clip_val=0., use_cloned_decoder=False, variance_sensitive_step=True,
                              acceptance_rate_target=0.9, annealing_scheme='linear')
        ais_vae = replace_enc_dec(ais_vae)
        ais_vae.name = 'AMCVAE'

        # ----- VAE_with_Flows ----- #
        flows_vae = VAE_with_flows_Toy(shape=28, act_func=nn.LeakyReLU,
                                       num_samples=1, hidden_dim=d,
                                       net_type='conv', dataset='toy',
                                       flow_type='RNVP', num_flows=2, need_permute=True)
        flows_vae = replace_enc_dec(flows_vae)
        flows_vae.name = 'VAE_with_Flows'
        flows_vae.encoder_net = ToyEncoder_VB(d=d).to(device)

        X_train = generate_dataset(N=N, d=d, sigma=sigma)
        X_val = generate_dataset(N=N // 100, d=d, sigma=sigma)

        train_dataset = ToyDataset(data=X_train)
        val_dataset = ToyDataset(data=X_val)
        train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)

        print('IWAE')
        run_trainer(iwae)
        full_results['IWAE'].append(compute_discrepancy(iwae).cpu().item())
        full_alphas['IWAE'].append(get_alpha(iwae).cpu().item())
        full_betas['IWAE'].append(get_beta(iwae).cpu().item())

        print('VAE')
        run_trainer(vae)
        full_results['VAE'].append(compute_discrepancy(vae).cpu().item())
        full_alphas['VAE'].append(get_alpha(vae).cpu().item())
        full_betas['VAE'].append(get_beta(vae).cpu().item())

        print('Flows')
        run_trainer(flows_vae, num_epoches=31)
        full_results['RealNVP'].append(compute_discrepancy(flows_vae).cpu().item())
        full_alphas['RealNVP'].append(get_alpha(flows_vae).cpu().item())
        full_betas['RealNVP'].append(get_beta(flows_vae).cpu().item())

        print('AIS')
        run_trainer(ais_vae)
        full_results['A-MCVAE'].append(compute_discrepancy(ais_vae).cpu().item())
        full_alphas['A-MCVAE'].append(get_alpha(ais_vae).cpu().item())
        full_betas['A-MCVAE'].append(get_beta(ais_vae).cpu().item())

        print('ULA')
        run_trainer(ula_vae)
        full_results['L-MCVAE'].append(compute_discrepancy(ula_vae).cpu().item())
        full_alphas['L-MCVAE'].append(get_alpha(ula_vae).cpu().item())
        full_betas['L-MCVAE'].append(get_beta(ula_vae).cpu().item())

    # as requested in comment
    with open('./toy_results.txt', 'w') as file:
        file.write(json.dumps(full_results))  # use `json.loads` to do the reverse

    with open('./toy_results_alpha.txt', 'w') as file:
        file.write(json.dumps(full_alphas))  # use `json.loads` to do the reverse

    with open('./toy_results_beta.txt', 'w') as file:
        file.write(json.dumps(full_betas))  # use `json.loads` to do the reverse
