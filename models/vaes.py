import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from models.samplers import HMC, MALA
from models.encoders import get_encoder
from models.decoders import get_decoder


class Base(pl.LightningModule):
    def __init__(self, num_samples, act_func, hidden_dim=64, name="VAE", net_type="fc",
                 dataset='mnist'):
        super(Base, self).__init__()
        self.hidden_dim = hidden_dim
        # Define Encoder part
        self.encoder_net = get_encoder(net_type, act_func, hidden_dim, dataset)
        # # Define Decoder part
        self.decoder_net = get_decoder(net_type, act_func, hidden_dim, dataset)
        # Number of latent samples per object
        self.num_samples = num_samples
        # Fixed random vector, which we recover each epoch
        self.random_z = torch.randn((64, hidden_dim), dtype=torch.float32)
        # Name, which is used for logging
        self.name = name
        self.dataset = dataset

    def encode(self, x):
        # We treat the first half of output as mu, and the rest as logvar
        h = self.encoder_net(x)
        return h[:, :h.shape[1] // 2], h[:, h.shape[1] // 2:]

    def reparameterize(self, mu, logvar):
        # Reparametrization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def enc_rep(self, x):
        mu, logvar = self.encode(x)
        mu = mu.repeat(self.num_samples, 1)
        logvar = logvar.repeat(self.num_samples, 1)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z):
        return self.decoder_net(z)

    def forward(self, z):
        return self.decode(z)

    def joint_density(self, ):
        def density(z, x):
            x_reconst = self(z)
            log_Pr = torch.distributions.Normal(loc=torch.tensor(0., device=z.device, dtype=torch.float32),
                                                scale=torch.tensor(1., device=z.device, dtype=torch.float32)).log_prob(
                z).sum(-1)
            return -F.binary_cross_entropy_with_logits(x_reconst, x.view(x_reconst.shape[0], -1), reduction='none').sum(
                -1) + log_Pr

        return density

    def validation_epoch_end(self, outputs):
        # Some stuff, which is needed for logging and tensorboard
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_BCE = torch.stack([x['BCE'] for x in outputs]).mean()

        if self.dataset.lower().find('cifar') > -1:
            x_hat = torch.sigmoid(self(self.random_z.to(val_loss.device))).view((-1, 3, 32, 32))
        else:
            x_hat = torch.sigmoid(self(self.random_z.to(val_loss.device))).view((-1, 1, 28, 28))
        grid = torchvision.utils.make_grid(x_hat)

        self.logger.experiment.add_image(f'image/{self.name}', grid, self.current_epoch)
        self.logger.experiment.add_scalar(f'avg_val_loss/{self.name}', val_loss, self.current_epoch)
        self.logger.experiment.add_scalar(f'avg_val_BCE/{self.name}', val_BCE, self.current_epoch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        output = self.step(batch)
        return {"loss": output[0], "BCE": output[-1]}

    def validation_step(self, batch, batch_idx):
        output = self.step(batch)
        return {"val_loss": output[0], "BCE": output[-1]}


class VAE(Base):
    def loss_function(self, recon_x, x, mu, logvar):
        batch_size = mu.shape[0] // self.num_samples
        BCE = F.binary_cross_entropy_with_logits(recon_x.view(mu.shape[0], -1), x.view(mu.shape[0], -1),
                                                 reduction='none').view(
            (self.num_samples, batch_size, -1)).mean(0).sum(-1).mean()
        KLD = -0.5 * torch.mean((1 + logvar - mu.pow(2) - logvar.exp()).view(
            (self.num_samples, -1, self.hidden_dim)).mean(0).sum(-1))
        loss = BCE + KLD
        return loss, BCE

    def step(self, batch):
        x, _ = batch
        z, mu, logvar = self.enc_rep(x)
        x_hat = self(z)
        loss, BCE = self.loss_function(x_hat, x.repeat(self.num_samples, 1, 1, 1), mu, logvar)
        return loss, x_hat, z, BCE


class IWAE(Base):
    def loss_function(self, recon_x, x, mu, logvar, z):
        batch_size = mu.shape[0] // self.num_samples
        log_Q = torch.distributions.Normal(loc=mu,
                                           scale=torch.exp(0.5 * logvar)).log_prob(z).view(
            (self.num_samples, -1, self.hidden_dim)).sum(-1)

        log_Pr = torch.sum((-0.5 * z ** 2).view((self.num_samples, -1, self.hidden_dim)), -1)
        BCE = F.binary_cross_entropy_with_logits(recon_x.view(mu.shape[0], -1), x.view(mu.shape[0], -1),
                                                 reduction='none').view(
            (self.num_samples, batch_size, -1)).sum(-1)
        log_weight = log_Pr - BCE - log_Q
        log_weight = log_weight - torch.max(log_weight, 0)[0]  # for stability
        weight = torch.exp(log_weight)
        weight = weight / torch.sum(weight, 0)
        weight = weight.detach()
        loss = torch.mean(torch.sum(weight * (-log_Pr + BCE + log_Q), 0))

        return loss, torch.sum(BCE * weight, dim=0).mean()

    def step(self, batch):
        x, _ = batch
        z, mu, logvar = self.enc_rep(x)
        x_hat = self(z)
        loss, BCE = self.loss_function(x_hat, x.repeat(self.num_samples, 1, 1, 1), mu, logvar, z)
        return loss, x_hat, z, BCE


class BaseMet(Base):
    def step(self, batch):
        x, _ = batch
        z, mu, logvar = self.enc_rep(x)
        z_transformed, p_transformed, sum_log_jacobian, sum_log_alpha, all_acceptance, p_old = self.run_transitions(z=z,
                                                                                                                    x=x.repeat(
                                                                                                                        self.num_samples,
                                                                                                                        1,
                                                                                                                        1,
                                                                                                                        1))
        x_hat = self(z_transformed)

        loss, BCE = self.loss_function(recon_x=x_hat, x=x.repeat(self.num_samples, 1, 1, 1), mu=mu, logvar=logvar, z=z,
                                       z_transformed=z_transformed,
                                       sum_log_alpha=sum_log_alpha, sum_log_jacobian=sum_log_jacobian, p=p_old,
                                       p_transformed=p_transformed)

        return loss, x_hat, z, all_acceptance, BCE

    def validation_step(self, batch, batch_idx):
        output = self.step(batch)
        return {"val_loss": output[0], "BCE": output[-1], "Acceptance": output[-2]}

    def validation_epoch_end(self, outputs):
        # Some stuff, which is needed for logging and tensorboard
        super(BaseMet, self).validation_epoch_end(outputs)

        val_acceptance = torch.stack([x['Acceptance'] for x in outputs]).mean(0).mean(-1)
        for i in range(len(val_acceptance)):
            self.logger.experiment.add_scalar(f'avg_val_acceptance_{i}/{self.name}', val_acceptance[i],
                                              self.current_epoch)


class MetHMC_VAE(BaseMet):
    def __init__(self, n_leapfrogs, step_size, K, use_barker=True, **kwargs):
        super().__init__(**kwargs)
        self.K = K
        self.transitions = nn.ModuleList(
            [HMC(n_leapfrogs=n_leapfrogs, step_size=step_size, use_barker=use_barker, partial_ref=True, learnable=True)
             for _ in range(self.K)])

    def run_transitions(self, z, x):
        sum_log_alpha = torch.zeros_like(z[:, 0])
        sum_log_jacobian = torch.zeros_like(z[:, 0])

        p = torch.randn_like(z)
        z_transformed = z
        p_transformed = p
        all_acceptance = torch.tensor([], dtype=torch.float32, device=x.device)
        for i in range(self.K):
            z_transformed, log_jac, current_log_alphas, directions, p_transformed = self.one_transition(current_num=i,
                                                                                                        z=z_transformed,
                                                                                                        p=p_transformed,
                                                                                                        x=x)
            sum_log_alpha = sum_log_alpha + current_log_alphas
            sum_log_jacobian = sum_log_jacobian + log_jac
            all_acceptance = torch.cat([all_acceptance, directions[None]])
        return z_transformed, p_transformed, sum_log_jacobian, sum_log_alpha, all_acceptance, p

    def one_transition(self, current_num, z, x, p):
        if p is None:
            p = torch.randn_like(z)
        log_jac = p.shape[1] * torch.log(self.transitions[current_num].alpha) * torch.ones_like(
            z[:, 0])
        z_new, p_new, directions, current_log_alphas = self.transitions[current_num].make_transition(z=z, x=x, p=p,
                                                                                                     target=self.joint_density())
        return z_new, log_jac, current_log_alphas, directions, p_new

    def loss_function(self, recon_x, x, mu, logvar, z, z_transformed, sum_log_alpha, sum_log_jacobian, p=None,
                      p_transformed=None):
        ## logdensity of Variational family
        log_q = torch.mean(torch.distributions.Normal(loc=mu, scale=torch.exp(0.5 * logvar)).log_prob(
            z).sum(1) + sum_log_alpha - sum_log_jacobian)
        ## logdensity of prior
        log_priors = torch.mean(-1. / 2 * torch.sum(z_transformed * z_transformed, 1))
        if p is not None:
            log_priors += torch.mean(- 1. / 2 * torch.sum(p_transformed * p_transformed, 1))
            log_q += torch.mean(- 1. / 2 * torch.sum(p * p, 1))

        log_r = -self.K * torch.log(torch.tensor(2., device=z.device, dtype=torch.float32))

        batch_size = mu.shape[0] // self.num_samples
        BCE = F.binary_cross_entropy_with_logits(recon_x.view(mu.shape[0], -1), x.view(mu.shape[0], -1),
                                                 reduction='none').view(
            (self.num_samples, batch_size, -1)).mean(0).sum(-1).mean()

        ELBO = -BCE + (log_priors + log_r - log_q)
        loss = -torch.mean(ELBO + ELBO.detach() * sum_log_alpha)
        return loss, BCE


class AIS_VAE(BaseMet):
    def __init__(self, step_size, K, use_barker=True, annealing_learnable=False, beta=None, **kwargs):
        super().__init__(**kwargs)
        self.K = K
        self.transitions = nn.ModuleList(
            [MALA(step_size=step_size, use_barker=use_barker, learnable=True)
             for _ in range(self.K)])

        self.annealing_learnable = annealing_learnable
        if beta is None:
            beta = np.linspace(0., 1., 10)
        self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float32), requires_grad=self.annealing_learnable)

    def one_transition(self, current_num, z, x, annealing_logdens):
        z_new, directions, current_log_alphas = self.transitions[current_num].make_transition(z=z, x=x,
                                                                                              target=annealing_logdens)
        return z_new, current_log_alphas, directions

    def run_transitions(self, z, x, mu, logvar):
        sum_log_alpha = torch.zeros_like(z[:, 0])
        sum_log_weights = torch.zeros_like(z[:, 0])

        init_logdens = lambda z: torch.distributions.Normal(loc=mu, scale=torch.exp(0.5 * logvar)).log_prob(
            z).sum(-1)
        annealing_logdens = lambda beta: lambda x, z: (1. - beta) * init_logdens(z=z) + beta * self.joint_density()(z=z,
                                                                                                                    x=x)
        z_transformed = z
        sum_log_weights += (self.beta[0] - 0) * (self.joint_density()(z=z, x=x) - init_logdens(z))

        all_acceptance = torch.tensor([], dtype=torch.float32, device=x.device)
        for i in range(self.K):
            z_transformed, current_log_alphas, directions = self.one_transition(current_num=i,
                                                                                z=z_transformed,
                                                                                x=x,
                                                                                annealing_logdens=annealing_logdens(
                                                                                    self.beta[i]))
            sum_log_alpha += current_log_alphas
            sum_log_weights += (self.beta[i + 1] - self.beta[i]) * (
                    self.joint_density()(z=z_transformed, x=x) - init_logdens(z=z_transformed))
            all_acceptance = torch.cat([all_acceptance, directions[None]])
        return z_transformed, sum_log_alpha, sum_log_weights, all_acceptance

    def step(self, batch):
        x, _ = batch
        z, mu, logvar = self.enc_rep(x)
        z_transformed, sum_log_alpha, sum_log_weights, all_acceptance = self.run_transitions(z=z, x=x.repeat(
            self.num_samples, 1, 1, 1),
                                                                                             mu=mu,
                                                                                             logvar=logvar)
        x_hat = self(z_transformed)

        loss, BCE = self.loss_function(sum_log_alpha=sum_log_alpha, sum_log_weights=sum_log_weights)

        return loss, x_hat, z, all_acceptance, BCE

    def loss_function(self, sum_log_alpha, sum_log_weights):
        elbo_est = sum_log_weights.mean()
        grad_elbo = -(
                elbo_est + elbo_est.detach() * sum_log_alpha.mean())  ###To think about : detach target in alpha terms to keep good objective for target

        return grad_elbo, elbo_est
