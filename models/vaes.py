import copy

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision

from models.aux import ULA_nn_sm
from models.decoders import get_decoder
from models.encoders import get_encoder
from models.normflows import NormFlow
from models.samplers import HMC, MALA, ULA


def binary_crossentropy_logits_stable(x, y):
    '''
    Special binary crossentropy where y can be a single data batch, while
    x has several repeats
    '''
    # return (1. - y)* x + torch.log(1.+torch.exp(-x))
    return torch.clamp(x, 0) - x * y + torch.log(1 + torch.exp(-torch.abs(x)))


def repeat_data(x, n_samples):
    if len(x.shape) == 4:
        x = x.repeat(n_samples, 1, 1, 1)
    else:
        x = x.repeat(n_samples, 1)
    return x


class Pass_grads:
    def __init__(self):
        pass

    def __enter__(self):
        return

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val:
            raise


class Stop_grads:
    def __init__(self, stop_grad):
        if stop_grad:
            self.action = torch.no_grad()
        else:
            self.action = Pass_grads()

    def __enter__(self):
        return self.action.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.action.__exit__(exc_type, exc_val, exc_tb)


class Base(pl.LightningModule):
    def __init__(self, num_samples, act_func, shape, hidden_dim=64, name="VAE", net_type="fc",
                 dataset='mnist', **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.dataset = dataset
        self.hidden_dim = hidden_dim
        # Define Encoder part
        self.encoder_net = get_encoder(net_type, act_func, hidden_dim, dataset, shape=shape)
        # # Define Decoder part
        self.decoder_net = get_decoder(net_type, act_func, hidden_dim, dataset, shape=shape)
        # Number of latent samples per object
        self.num_samples = num_samples
        # Fixed random vector, which we recover each epoch
        self.random_z = torch.randn((64, hidden_dim), dtype=torch.float32)
        # Name, which is used for logging
        self.name = name
        self.transitions_nll = nn.ModuleList(
            [HMC(n_leapfrogs=3, step_size=0.05, use_barker=False)
             for _ in range(5 - 1)])
        for p in self.transitions_nll.parameters():
            p.requires_grad_(False)

    def encode(self, x):
        # We treat the first half of output as mu, and the rest as logvar
        h = self.encoder_net(x)
        return h[:, :h.shape[1] // 2], h[:, h.shape[1] // 2:]

    def reparameterize(self, mu, logvar):
        # Reparametrization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def enc_rep(self, x, n_samples=1):
        mu, logvar = self.encode(x)
        mu = mu.repeat(n_samples, 1)
        logvar = logvar.repeat(n_samples, 1)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z):
        return self.decoder_net(z)

    def forward(self, z):
        return self.decode(z)

    def one_transition(self, current_num, z, x, annealing_logdens, nll=False):
        # nll flag is to use those transitions, which are specifically defined for nll.
        z_new = self.transitions_nll[current_num].make_transition(z=z, x=x,
                                                                  target=annealing_logdens)
        return z_new

    def joint_logdensity(self, use_true_decoder=None):
        def density(z, x):
            if (use_true_decoder is not None) and use_true_decoder:
                x_reconst = self(z)
            elif hasattr(self, 'use_cloned_decoder') and self.use_cloned_decoder:
                x_reconst = self.cloned_decoder(z)
            else:
                x_reconst = self(z)

            log_Pr = torch.distributions.Normal(loc=torch.tensor(0., device=x.device, dtype=torch.float32),
                                                scale=torch.tensor(1., device=x.device, dtype=torch.float32)).log_prob(
                z).sum(-1)
            # return -F.binary_cross_entropy_with_logits(x_reconst.view(x_reconst.shape[0], -1),
            #                                            x.view(x_reconst.shape[0], -1), reduction='none').sum(
            #     -1) + log_Pr
            return -binary_crossentropy_logits_stable(x_reconst.view(x_reconst.shape[0], -1),
                                                      x.view(x_reconst.shape[0], -1)).sum(-1) + log_Pr

        return density

    def validation_epoch_end(self, outputs):
        # Some stuff, which is needed for logging and tensorboard
        if "val_loss" in outputs[0].keys():
            val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
            self.logger.experiment.add_scalar(f'{self.dataset}/{self.name}/avg_val_loss', val_loss, self.current_epoch)
        else:
            val_loss = torch.stack([x['val_loss_enc'] for x in outputs]).mean()
            val_loss_dec = torch.stack([x['val_loss_dec'] for x in outputs]).mean()
            self.logger.experiment.add_scalar(f'{self.dataset}/{self.name}/avg_val_loss_enc', val_loss,
                                              self.current_epoch)
            self.logger.experiment.add_scalar(f'{self.dataset}/{self.name}/avg_val_loss_dec', val_loss_dec,
                                              self.current_epoch)
            if "val_loss_score_match" in outputs[0].keys():
                val_loss_score_match = torch.stack([x['val_loss_score_match'] for x in outputs]).mean()
                self.logger.experiment.add_scalar(f'{self.dataset}/{self.name}/avg_val_loss_score_match',
                                                  val_loss_score_match,
                                                  self.current_epoch)

        if "acceptance_rate" in outputs[0].keys():
            acceptance = torch.stack([x['acceptance_rate'] for x in outputs]).mean(0)
            for i in range(len(acceptance)):
                self.logger.experiment.add_scalar(f'{self.dataset}/{self.name}/acceptance_rate_{i}',
                                                  acceptance[i].item(),
                                                  self.current_epoch)

        if "nll" in outputs[0].keys():
            nll = torch.stack([x['nll'] for x in outputs]).mean(0)
            self.logger.experiment.add_scalar(f'{self.dataset}/{self.name}/nll', nll,
                                              self.current_epoch)

        if self.dataset.lower() in ['cifar', 'mnist', 'omniglot', 'celeba']:
            if self.dataset.lower().find('cifar') > -1:
                x_hat = torch.sigmoid(self(self.random_z.to(val_loss.device))).view((-1, 3, 32, 32))
            elif self.dataset.lower().find('mnist') > -1:
                x_hat = torch.sigmoid(self(self.random_z.to(val_loss.device))).view((-1, 1, 28, 28))
            elif self.dataset.lower().find('omniglot') > -1:
                x_hat = torch.sigmoid(self(self.random_z.to(val_loss.device))).view((-1, 1, 105, 105))
            elif self.dataset.lower().find('celeba') > -1:
                x_hat = torch.sigmoid(self(self.random_z.to(val_loss.device))).view((-1, 3, 64, 64))
            grid = torchvision.utils.make_grid(x_hat)
            self.logger.experiment.add_image(f'{self.dataset}/{self.name}/image', grid, self.current_epoch)
        else:
            pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        output = self.step(batch)
        return {"loss": output[0]}

    def validation_step(self, batch, batch_idx):
        output = self.step(batch)
        d = {"val_loss": output[0]}
        # TODO: Bypass self.current_epoch here or 'dataset'
        if (self.current_epoch == 29):
            nll = self.evaluate_nll(batch=batch,
                                    beta=torch.linspace(0., 1., 5, device=batch[0].device, dtype=torch.float32))
            d.update({"nll": nll})
        return d

    def evaluate_nll(self, batch, beta):
        x, _ = batch
        with torch.no_grad():
            n_samples = 10  # num samples for NLL estimation
            z, mu, logvar = self.enc_rep(x, n_samples=n_samples)
            x = repeat_data(x, n_samples)
            init_logdens = lambda z: torch.distributions.Normal(loc=mu, scale=torch.exp(0.5 * logvar)).log_prob(
                z).sum(-1)
            annealing_logdens = lambda beta: lambda z, x: (1. - beta) * init_logdens(
                z=z) + beta * self.joint_logdensity()(
                z=z,
                x=x)
            sum_log_weights = (beta[1] - beta[0]) * (self.joint_logdensity()(z=z, x=x) - init_logdens(z))

            for i in range(1, len(beta) - 1):
                z = self.one_transition(current_num=i - 1, z=z, x=x,
                                        annealing_logdens=annealing_logdens(beta=beta[i]), nll=True)[0]

                sum_log_weights += (beta[i + 1] - beta[i]) * (self.joint_logdensity()(z=z, x=x) - init_logdens(z=z))
            sum_log_weights = sum_log_weights.view(n_samples, batch[0].shape[0])
            batch_nll_estimator = -torch.logsumexp(sum_log_weights,
                                                   dim=0) + torch.log(torch.tensor(n_samples, dtype=torch.float32,
                                                                                   device=x.device))  ###Should be a vector of batchsize containing nll estimator for each term of the batch

            return torch.mean(batch_nll_estimator)


class VAE(Base):
    def loss_function(self, recon_x, x, mu, logvar):
        batch_size = mu.shape[0] // self.num_samples
        # BCE = F.binary_cross_entropy_with_logits(recon_x.view(mu.shape[0], -1), x.view(mu.shape[0], -1),
        #                                          reduction='none').view(
        #     (self.num_samples, batch_size, -1)).mean(0).sum(-1).mean()
        BCE = binary_crossentropy_logits_stable(recon_x.view(mu.shape[0], -1), x.view(mu.shape[0], -1)).view(
            (self.num_samples, batch_size, -1)).mean(0).sum(-1).mean()
        KLD = -0.5 * torch.mean((1 + logvar - mu.pow(2) - logvar.exp()).view(
            (self.num_samples, -1, self.hidden_dim)).mean(0).sum(-1))
        loss = BCE + KLD
        return loss

    def step(self, batch):
        x, _ = batch
        z, mu, logvar = self.enc_rep(x, self.num_samples)
        x_hat = self(z)
        x = repeat_data(x, self.num_samples)
        loss = self.loss_function(x_hat, x, mu, logvar)
        return loss, x_hat, z


class IWAE(Base):
    def loss_function(self, recon_x, x, mu, logvar, z):
        batch_size = mu.shape[0] // self.num_samples
        log_Q = torch.distributions.Normal(loc=mu, scale=torch.exp(0.5 * logvar)).log_prob(z).view(
            (self.num_samples, -1, self.hidden_dim)).sum(-1)

        log_Pr = torch.sum((-0.5 * z ** 2).view((self.num_samples, -1, self.hidden_dim)), -1)
        # BCE = F.binary_cross_entropy_with_logits(recon_x.view(mu.shape[0], -1), x.view(mu.shape[0], -1),
        #                                          reduction='none').view(
        #     (self.num_samples, batch_size, -1)).sum(-1)
        BCE = binary_crossentropy_logits_stable(recon_x.view(mu.shape[0], -1), x.view(mu.shape[0], -1)).view(
            (self.num_samples, batch_size, -1)).sum(-1)
        log_weight = log_Pr - BCE - log_Q
        log_weight = log_weight - torch.max(log_weight, 0)[0]  # for stability
        weight = torch.exp(log_weight)
        weight = weight / torch.sum(weight, 0)
        weight = weight.detach()
        loss = torch.mean(torch.sum(weight * (-log_Pr + BCE + log_Q), 0))

        return loss

    def step(self, batch):
        x, _ = batch
        z, mu, logvar = self.enc_rep(x, self.num_samples)
        x_hat = self(z)
        x = repeat_data(x, self.num_samples)
        loss = self.loss_function(x_hat, x, mu, logvar, z)
        return loss, x_hat, z


class VAE_with_flows(Base):
    def __init__(self, flow_type, num_flows, need_permute, **kwargs):
        super().__init__(**kwargs)
        self.Flow = NormFlow(flow_type, num_flows, self.hidden_dim, need_permute)
        self.save_hyperparameters()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def loss_function(self, recon_x, x, mu, logvar, z, z_transformed, log_jac):
        batch_size = mu.shape[0] // self.num_samples
        BCE = binary_crossentropy_logits_stable(recon_x, x.view(mu.shape[0], -1)).view(
            (self.num_samples, batch_size, -1)).mean(0).sum(-1).mean()
        log_Q = torch.mean(torch.distributions.Normal(loc=mu, scale=torch.exp(0.5 * logvar)).log_prob(z).view(
            (self.num_samples, batch_size, -1)).sum(-1) - log_jac.view((self.num_samples, -1)), dim=0).mean()
        log_Pr = (-0.5 * z_transformed ** 2).view(
            (self.num_samples, batch_size, -1)).mean(0).sum(-1).mean()
        KLD = torch.mean(log_Q - log_Pr).mean(0)
        loss = BCE + KLD
        return loss

    def step(self, batch):
        x, _ = batch
        z, mu, logvar = self.enc_rep(x, self.num_samples)
        x = repeat_data(x, self.num_samples)
        z_transformed, log_jac = self.Flow(z)
        x_hat = self(z_transformed)
        loss = self.loss_function(x_hat, x, mu, logvar, z, z_transformed, log_jac)
        return loss, x_hat, z_transformed


class BaseAIS(Base):
    def __init__(self, step_size, K, beta=None, grad_skip_val=0., grad_clip_val=0., use_cloned_decoder=False, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.K = K
        self.epsilons = [step_size for _ in range(self.K)]
        self.epsilon_target = 0.95
        self.epsilon_decrease_alpha = 0.998
        self.epsilon_increase_alpha = 1.002
        self.epsilon_min = 0.001
        self.epsilon_max = 0.3
        if beta is None:
            beta = torch.tensor(np.linspace(0., 1., self.K + 2), dtype=torch.float32)
        self.register_buffer('beta', beta)
        self.grad_skip_val = grad_skip_val
        self.grad_clip_val = grad_clip_val
        self.use_cloned_decoder = use_cloned_decoder
        if self.use_cloned_decoder:
            self.cloned_decoder = copy.deepcopy(self.decoder_net)

    def configure_optimizers(self):
        decoder_optimizer = torch.optim.Adam(self.decoder_net.parameters(), lr=1e-3, eps=1e-4)
        inference_params = list(self.encoder_net.parameters()) + list(self.transitions.parameters())
        ## If we are using cloned decoder to approximate the true one, we add its params to inference optimizer
        if self.use_cloned_decoder:
            inference_params += list(self.cloned_decoder.parameters())
        encoder_optimizer = torch.optim.Adam(inference_params, lr=1e-3, eps=1e-4)
        return [decoder_optimizer, encoder_optimizer]

    def run_transitions(self, z, x, mu, logvar, inference_part=False):
        init_logdens = lambda z: torch.distributions.Normal(loc=mu, scale=torch.exp(0.5 * logvar)).log_prob(
            z).sum(-1)
        init_logdens_detached = lambda z: torch.distributions.Normal(loc=mu.detach(),
                                                                     scale=torch.exp(0.5 * logvar.detach())).log_prob(
            z).sum(-1)

        ## If we need to prop grads thru alphas, use init_logdens(w/o detaching), otherwise init_logdens_detached
        init_logdens_aux = init_logdens if inference_part else init_logdens_detached

        annealing_logdens = lambda beta: lambda z, x: (1. - beta) * init_logdens_aux(
            z=z) + beta * self.joint_logdensity()(
            z=z,
            x=x)

        output = self.specific_transitions(
            z=z,
            x=x,
            init_logdens=init_logdens,
            annealing_logdens=annealing_logdens,
            inference_part=inference_part
        )

        return output

    def update_stepsize(self, accept_rate):
        accept_rate = list(accept_rate.cpu().data.numpy())
        for l in range(0, self.K - 1):
            if accept_rate[l] < self.epsilon_target:
                self.epsilons[l] *= self.epsilon_decrease_alpha
            else:
                self.epsilons[l] *= self.epsilon_increase_alpha

            if self.epsilons[l] < self.epsilon_min:
                self.epsilons[l] = self.epsilon_min
            if self.epsilons[l] > self.epsilon_max:
                self.epsilons[l] = self.epsilon_max
            self.transitions[l].log_stepsize.data = torch.tensor(np.log(self.epsilons[l]), dtype=torch.float32,
                                                                 device=self.transitions[l].log_stepsize.device)

    def step(self, batch):
        x, _ = batch
        z, mu, logvar = self.enc_rep(x, self.num_samples)
        x = repeat_data(x, self.num_samples)
        z_transformed, sum_log_weights, _, all_acceptance = self.run_transitions(z=z, x=x, mu=mu, logvar=logvar)
        x_hat = self(z_transformed)
        loss_enc = self.loss_function(z, x_hat, x, mu, logvar, sum_log_weights, inference_part=False)
        loss_dec = self.loss_function(z_transformed, x_hat, x, mu, logvar, sum_log_weights, inference_part=True)
        return loss_enc, loss_dec, all_acceptance, z_transformed

    def validation_step(self, batch, batch_idx):
        output = self.step(batch)
        d = {"val_loss_enc": output[0], "val_loss_dec": output[1], "acceptance_rate": output[2].mean(1)}
        if self.current_epoch == 49:
            nll = self.evaluate_nll(batch=batch,
                                    beta=torch.linspace(0., 1., 5, device=batch[0].device, dtype=torch.float32))
            d.update({"nll": nll})
        return d


class AIWAE(BaseAIS):
    def __init__(self, n_leapfrogs, use_barker, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.transitions = nn.ModuleList(
            [HMC(n_leapfrogs=n_leapfrogs, step_size=self.epsilons[0], use_barker=use_barker)
             for _ in range(self.K - 1)])

    def one_transition(self, current_num, z, x, annealing_logdens):
        z_new, _, directions, current_log_alphas = self.transitions[current_num].make_transition(z=z, x=x,
                                                                                                 target=annealing_logdens)
        return z_new, current_log_alphas, directions

    def specific_transitions(self, z, x, init_logdens, annealing_logdens, inference_part=False):
        all_acceptance = torch.tensor([], dtype=torch.float32, device=x.device)
        sum_log_weights = (self.beta[1] - self.beta[0]) * (self.joint_logdensity()(z=z, x=x) - init_logdens(z))
        z_transformed = z
        for i in range(1, self.K - 1):
            z_transformed, current_log_alphas, directions = self.one_transition(current_num=i - 1,
                                                                                z=z_transformed,
                                                                                x=x,
                                                                                annealing_logdens=annealing_logdens(
                                                                                    beta=self.beta[i]))
            sum_log_weights += (self.beta[i + 1] - self.beta[i]) * (
                    self.joint_logdensity()(z=z_transformed, x=x) - init_logdens(z=z_transformed))
            all_acceptance = torch.cat([all_acceptance, directions[None]])

        z_transformed, current_log_alphas, directions = self.one_transition(current_num=self.K - 2,
                                                                            z=z_transformed,
                                                                            x=x,
                                                                            annealing_logdens=annealing_logdens(
                                                                                self.beta[self.K - 1]))
        all_acceptance = torch.cat([all_acceptance, directions[None]])
        self.update_stepsize(all_acceptance.mean(1))
        return z_transformed, sum_log_weights, None, all_acceptance

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, _ = batch
        if optimizer_idx == 0:  # decoder
            with torch.no_grad():
                z, mu, logvar = self.enc_rep(x, self.num_samples)
                x = repeat_data(x, self.num_samples)
                z_transformed, sum_log_weights, _, all_acceptance = self.run_transitions(z=z, x=x, mu=mu, logvar=logvar)
                x_hat = self(z_transformed)
            loss = self.loss_function(z_transformed, x_hat, x, mu, logvar, sum_log_weights, inference_part=False)
        else:  # encoder
            z, mu, logvar = self.enc_rep(x)
            x_hat = self(z)
            loss = self.loss_function(z, x_hat, x, mu, logvar, None, inference_part=True)
        return {"loss": loss}

    def loss_function(self, z_transformed, x_hat, x, mu, logvar, sum_log_weights, inference_part):
        batch_size = mu.shape[0] // self.num_samples
        if inference_part:
            # BCE = F.binary_cross_entropy_with_logits(x_hat.view(mu.shape[0], -1), x.view(mu.shape[0], -1),
            #                                          reduction='none').sum(-1).mean()
            BCE = binary_crossentropy_logits_stable(x_hat.view(mu.shape[0], -1), x.view(mu.shape[0], -1)).sum(-1).mean()
            KLD = -0.5 * torch.mean((1 + logvar - mu.pow(2) - logvar.exp()).sum(-1))
            loss = BCE + KLD
        else:
            sum_log_weights = sum_log_weights.view((self.num_samples, batch_size))
            log_weight = sum_log_weights - torch.max(sum_log_weights, 0)[0]  # for stability
            weight = torch.exp(log_weight)
            weight = weight / torch.sum(weight, 0)
            loss = -torch.mean(
                torch.sum(weight * self.joint_logdensity()(z=z_transformed, x=x).view((self.num_samples, batch_size)),
                          0))
        return loss


class AIS_VAE(BaseAIS):
    def __init__(self, use_barker, learnable_transitions=False, **kwargs):
        super().__init__(**kwargs)
        self.learnable_transitions = learnable_transitions
        self.transitions = nn.ModuleList(
            [MALA(step_size=self.epsilons[0], use_barker=use_barker, learnable=learnable_transitions)
             for _ in range(self.K)])
        self.save_hyperparameters()
        self.moving_averages = []

    def one_transition(self, current_num, z, x, annealing_logdens, nll=False):
        if nll:
            with torch.no_grad():
                z_new = self.transitions_nll[current_num].make_transition(z=z, x=x, target=annealing_logdens)
            return z_new
        else:
            z_new, directions, current_log_alphas = self.transitions[current_num].make_transition(z=z, x=x,
                                                                                                  target=annealing_logdens)
            return z_new, current_log_alphas, directions

    def specific_transitions(self, z, x, init_logdens, annealing_logdens, inference_part=False):
        all_acceptance = torch.tensor([], dtype=torch.float32, device=x.device)
        sum_log_weights = (self.beta[1] - self.beta[0]) * (self.joint_logdensity()(z=z, x=x) - init_logdens(z))
        sum_log_alpha = torch.zeros_like(z[:, 0])
        z_transformed = z
        for i in range(1, self.K + 1):
            with Stop_grads(not inference_part):  # and self.use_cloned_decoder): #######
                z_transformed, current_log_alphas, directions = self.one_transition(current_num=i - 1,
                                                                                    z=z_transformed,
                                                                                    x=x,
                                                                                    annealing_logdens=annealing_logdens(
                                                                                        beta=self.beta[i]))
                sum_log_alpha += current_log_alphas

            sum_log_weights += (self.beta[i + 1] - self.beta[i]) * (
                    self.joint_logdensity(use_true_decoder=(i == self.K))(z=z_transformed, x=x) - init_logdens(
                z=z_transformed))
            all_acceptance = torch.cat([all_acceptance, directions[None]])

        self.update_stepsize(all_acceptance.mean(1))
        return z_transformed, sum_log_weights, sum_log_alpha, all_acceptance

    def step(self, batch):
        x, _ = batch
        z, mu, logvar = self.enc_rep(x, self.num_samples)
        x = repeat_data(x, self.num_samples)
        z_transformed, sum_log_weights, sum_log_alpha, all_acceptance = self.run_transitions(z=z,
                                                                                             x=x,
                                                                                             mu=mu,
                                                                                             logvar=logvar,
                                                                                             inference_part=False)

        loss_enc = self.loss_function(sum_log_alpha=sum_log_alpha, sum_log_weights=sum_log_weights, inference_part=True)
        loss_dec = self.loss_function(sum_log_alpha=sum_log_alpha, sum_log_weights=sum_log_weights,
                                      inference_part=False)
        return loss_enc, loss_dec, all_acceptance, z_transformed

    def loss_function(self, sum_log_alpha, sum_log_weights, inference_part):
        batch_size = sum_log_weights.shape[0] // self.num_samples
        elbo_est = sum_log_weights.view((self.num_samples, batch_size)).sum(0)
        if len(self.moving_averages) == 0:
            self.moving_averages.append(torch.mean(sum_log_alpha.view((self.num_samples, batch_size)).sum(0).detach()))
            self.moving_averages.append(torch.mean(elbo_est.detach()))
        else:
            self.moving_averages[0] = 0.9 * self.moving_averages[0] + 0.1 * torch.mean(
                sum_log_alpha.view((self.num_samples, batch_size)).sum(0).detach())
            self.moving_averages[1] = 0.9 * self.moving_averages[1] + 0.1 * torch.mean(elbo_est.detach())

        loss = -(torch.mean(elbo_est) + torch.mean(
            (elbo_est.detach() - self.moving_averages[1]) * (
                    sum_log_alpha.view((self.num_samples, batch_size)).sum(0) - self.moving_averages[0])))
        return loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, _ = batch
        if self.grad_skip_val == 0.:
            inference_part = {0: False, 1: True}[optimizer_idx]
            with Stop_grads(not inference_part):
                z, mu, logvar = self.enc_rep(x, self.num_samples)

            x = repeat_data(x, self.num_samples)
            z_transformed, sum_log_weights, sum_log_alpha, all_acceptance = self.run_transitions(z=z,
                                                                                                 x=x,
                                                                                                 mu=mu,
                                                                                                 logvar=logvar,
                                                                                                 inference_part=inference_part)
            loss = self.loss_function(sum_log_alpha=sum_log_alpha, sum_log_weights=sum_log_weights,
                                      inference_part=inference_part)
        else:
            (opt_decoder, opt_encoder_flows) = self.optimizers()
            ## decoder
            with Stop_grads(True):
                z, mu, logvar = self.enc_rep(x, self.num_samples)
            x_ = repeat_data(x, self.num_samples)
            z_transformed, sum_log_weights, sum_log_alpha, all_acceptance = self.run_transitions(z=z,
                                                                                                 x=x_,
                                                                                                 mu=mu,
                                                                                                 logvar=logvar,
                                                                                                 inference_part=False)
            loss = self.loss_function(sum_log_alpha=sum_log_alpha, sum_log_weights=sum_log_weights,
                                      inference_part=False)
            self.manual_backward(loss, opt_decoder)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.decoder_net.parameters(), self.grad_clip_val).item()
            if grad_norm < self.grad_skip_val:
                opt_decoder.step()

            ## encoder and flows
            with Stop_grads(False):
                z, mu, logvar = self.enc_rep(x, self.num_samples)
            z_transformed, sum_log_weights, sum_log_alpha, all_acceptance = self.run_transitions(z=z,
                                                                                                 x=x_,
                                                                                                 mu=mu,
                                                                                                 logvar=logvar,
                                                                                                 inference_part=True)
            loss = self.loss_function(sum_log_alpha=sum_log_alpha, sum_log_weights=sum_log_weights,
                                      inference_part=True)
            self.manual_backward(loss, opt_encoder_flows)
            inf_params = list(self.encoder_net.parameters()) + list(self.transitions.parameters())
            if self.use_cloned_decoder:
                inf_params += list(self.cloned_decoder.parameters())
            grad_norm = torch.nn.utils.clip_grad_norm_(inf_params, self.grad_clip_val).item()
            if grad_norm < self.grad_skip_val:
                opt_encoder_flows.step()
        return {"loss": loss}


class AIS_VAE_S(AIS_VAE):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()

    def configure_optimizers(self):
        params = list(self.decoder_net.parameters()) + list(self.encoder_net.parameters()) + list(
            self.transitions.parameters())
        ## If we are using cloned decoder to approximate the true one, we add its params to inference optimizer
        if self.use_cloned_decoder:
            params += list(self.cloned_decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-3, eps=1e-4)
        return [optimizer]

    def specific_transitions(self, z, x, init_logdens, annealing_logdens, inference_part=True):
        all_acceptance = torch.tensor([], dtype=torch.float32, device=x.device)
        sum_log_weights = (self.beta[1] - self.beta[0]) * (self.joint_logdensity()(z=z, x=x) - init_logdens(z))
        sum_log_alpha = torch.zeros_like(z[:, 0])
        z_transformed = z
        for i in range(1, self.K + 1):
            z_transformed, current_log_alphas, directions = self.one_transition(current_num=i - 1,
                                                                                z=z_transformed,
                                                                                x=x,
                                                                                annealing_logdens=annealing_logdens(
                                                                                    beta=self.beta[i]))
            sum_log_alpha += current_log_alphas
            sum_log_weights += (self.beta[i + 1] - self.beta[i]) * (
                    self.joint_logdensity(use_true_decoder=(i == self.K))(z=z_transformed, x=x) - init_logdens(
                z=z_transformed))
            all_acceptance = torch.cat([all_acceptance, directions[None]])

        self.update_stepsize(all_acceptance.mean(1))
        return z_transformed, sum_log_weights, sum_log_alpha, all_acceptance

    def step(self, batch):
        x, _ = batch
        z, mu, logvar = self.enc_rep(x, self.num_samples)
        x = repeat_data(x, self.num_samples)
        z_transformed, sum_log_weights, sum_log_alpha, all_acceptance = self.run_transitions(z=z,
                                                                                             x=x,
                                                                                             mu=mu,
                                                                                             logvar=logvar,
                                                                                             inference_part=True)

        loss = self.loss_function(sum_log_alpha=sum_log_alpha, sum_log_weights=sum_log_weights)
        return loss, all_acceptance, z_transformed

    def validation_step(self, batch, batch_idx):
        output = self.step(batch)
        d = {"val_loss": output[0], "acceptance_rate": output[1].mean(1)}
        if self.current_epoch == 49:
            nll = self.evaluate_nll(batch=batch,
                                    beta=torch.linspace(0., 1., 5, device=batch[0].device, dtype=torch.float32))
            d.update({"nll": nll})
        return d

    def loss_function(self, sum_log_alpha, sum_log_weights):
        batch_size = sum_log_weights.shape[0] // self.num_samples
        elbo_est = sum_log_weights.view((self.num_samples, batch_size)).sum(0)
        if len(self.moving_averages) == 0:
            self.moving_averages.append(torch.mean(sum_log_alpha.view((self.num_samples, batch_size)).sum(0).detach()))
            self.moving_averages.append(torch.mean(elbo_est.detach()))
        else:
            self.moving_averages[0] = 0.9 * self.moving_averages[0] + 0.1 * torch.mean(
                sum_log_alpha.view((self.num_samples, batch_size)).sum(0).detach())
            self.moving_averages[1] = 0.9 * self.moving_averages[1] + 0.1 * torch.mean(elbo_est.detach())
        loss = -(torch.mean(elbo_est) + torch.mean(
            (elbo_est.detach() - self.moving_averages[1]) * (
                    sum_log_alpha.view((self.num_samples, batch_size)).sum(0) - self.moving_averages[0])))
        return loss

    def training_step(self, batch, batch_idx):
        x, _ = batch
        if self.grad_skip_val == 0.:
            inference_part = True
            z, mu, logvar = self.enc_rep(x, self.num_samples)

            x = repeat_data(x, self.num_samples)
            z_transformed, sum_log_weights, sum_log_alpha, all_acceptance = self.run_transitions(z=z,
                                                                                                 x=x,
                                                                                                 mu=mu,
                                                                                                 logvar=logvar,
                                                                                                 inference_part=inference_part)
            loss = self.loss_function(sum_log_alpha=sum_log_alpha, sum_log_weights=sum_log_weights)
            return {"loss": loss}
        else:
            optimizer = self.optimizers()

            ## decoder
            z, mu, logvar = self.enc_rep(x, self.num_samples)
            x_ = repeat_data(x, self.num_samples)
            z_transformed, sum_log_weights, sum_log_alpha, all_acceptance = self.run_transitions(z=z,
                                                                                                 x=x_,
                                                                                                 mu=mu,
                                                                                                 logvar=logvar,
                                                                                                 inference_part=True)
            loss = self.loss_function(sum_log_alpha=sum_log_alpha, sum_log_weights=sum_log_weights)
            self.manual_backward(loss, optimizer)
            params = list(self.decoder_net.parameters()) + list(self.encoder_net.parameters()) + list(
                self.transitions.parameters())
            ## If we are using cloned decoder to approximate the true one, we add its params to inference optimizer
            if self.use_cloned_decoder:
                params += list(self.cloned_decoder.parameters())
            grad_norm = torch.nn.utils.clip_grad_norm_(params, self.grad_clip_val).item()
            if grad_norm < self.grad_skip_val:
                optimizer.step()
            return


class ULA_VAE(BaseAIS):
    def __init__(self, use_transforms=False, learnable_transitions=False, **kwargs):
        super().__init__(**kwargs)
        self.learnable_transitions = learnable_transitions
        if use_transforms:
            # if kwargs['score_matching']: to write it better
            additional_dims = 2 if kwargs['dataset'] == 'toy' else kwargs['shape'] ** 2
            transforms = lambda: ULA_nn_sm(input=kwargs['hidden_dim'] + additional_dims,
                                           output=kwargs['hidden_dim'],
                                           hidden=(kwargs['hidden_dim'] * 10, kwargs['hidden_dim'] * 10),
                                           h_dim=None)
        else:
            transforms = None
        self.transitions = nn.ModuleList(
            [ULA(step_size=self.epsilons[0], learnable=self.learnable_transitions, transforms=transforms,
                 return_pre_alphas=True)
             for _ in range(self.K)])
        self.save_hyperparameters()
        self.score_matching = use_transforms

    def configure_optimizers(self):
        inference_params = list(self.encoder_net.parameters())
        score_matching_optimizer = None
        if self.use_cloned_decoder:
            inference_params += list(self.cloned_decoder.parameters())

        if self.score_matching:
            score_matching_optimizer = torch.optim.Adam(self.transitions.parameters(), lr=1e-4, eps=1e-4)
        else:
            inference_params += list(self.transitions.parameters())

        decoder_optimizer = torch.optim.Adam(self.decoder_net.parameters(), lr=1e-3, eps=1e-5)
        encoder_optimizer = torch.optim.Adam(inference_params, lr=1e-3, eps=1e-4)
        optimizers_list = [decoder_optimizer, encoder_optimizer]
        if score_matching_optimizer is not None:
            optimizers_list.append(score_matching_optimizer)

        return optimizers_list

    def one_transition(self, current_num, z, x, annealing_logdens, nll=False):
        if nll:
            z_new = self.transitions_nll[current_num].make_transition(z=z, x=x,
                                                                      target=annealing_logdens)
            return z_new
        else:
            z_new, current_log_weights, directions, score_match_cur = self.transitions[current_num].make_transition(z=z,
                                                                                                                    x=x,
                                                                                                                    target=annealing_logdens)
            return z_new, current_log_weights, directions, score_match_cur

    def specific_transitions(self, z, x, init_logdens, annealing_logdens, inference_part=False):
        all_acceptance = torch.tensor([], dtype=torch.float32, device=x.device)
        sum_log_weights = -init_logdens(z=z)
        loss_sm = torch.zeros_like(z)
        z_transformed = z
        for i in range(1, self.K + 1):
            with Stop_grads(not inference_part):
                z_transformed, current_log_weights, directions, score_match_cur = self.one_transition(current_num=i - 1,
                                                                                                      z=z_transformed,
                                                                                                      x=x,
                                                                                                      annealing_logdens=annealing_logdens(
                                                                                                          beta=
                                                                                                          self.beta[i]))
            loss_sm += score_match_cur
            sum_log_weights += current_log_weights
            all_acceptance = torch.cat([all_acceptance, directions[None]])
        sum_log_weights += self.joint_logdensity()(z=z_transformed, x=x)
        self.update_stepsize(all_acceptance.mean(1))
        return z_transformed, sum_log_weights, loss_sm, all_acceptance

    def step(self, batch):
        x, _ = batch
        z, mu, logvar = self.enc_rep(x, self.num_samples)
        x = repeat_data(x, self.num_samples)
        z_transformed, sum_log_weights, loss_sm, all_acceptance = self.run_transitions(z=z,
                                                                                       x=x,
                                                                                       mu=mu,
                                                                                       logvar=logvar,
                                                                                       inference_part=False)

        loss_enc = self.loss_function(sum_log_weights=sum_log_weights)
        loss_dec = self.loss_function(sum_log_weights=sum_log_weights)
        loss_sm = loss_sm.sum(1).mean()
        return loss_enc, loss_dec, all_acceptance, loss_sm, z_transformed

    def loss_function(self, sum_log_weights):
        # import pdb
        # pdb.set_trace()
        batch_size = sum_log_weights.shape[0] // self.num_samples
        elbo_est = sum_log_weights.view((self.num_samples, batch_size)).sum(0)
        loss = -torch.mean(elbo_est)
        return loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, _ = batch
        inference_part = {0: False, 1: True, 2: True}[optimizer_idx]
        with Stop_grads(not inference_part):
            z, mu, logvar = self.enc_rep(x, self.num_samples)

        x = repeat_data(x, self.num_samples)
        z_transformed, sum_log_weights, loss_sm, all_acceptance = self.run_transitions(z=z,
                                                                                       x=x,
                                                                                       mu=mu,
                                                                                       logvar=logvar,
                                                                                       inference_part=inference_part)
        if optimizer_idx == 2:
            loss = loss_sm.sum(1).mean()
        else:
            loss = self.loss_function(sum_log_weights)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        output = self.step(batch)
        d = {"val_loss_enc": output[0], "val_loss_dec": output[1], "acceptance_rate": output[2].mean(1),
             "val_loss_score_match": output[3]}
        if (self.current_epoch == 49):
            nll = self.evaluate_nll(batch=batch,
                                    beta=torch.linspace(0., 1., 5, device=batch[0].device, dtype=torch.float32))
            d.update({"nll": nll})
        return d


class Stacked_VAE(pl.LightningModule):
    def __init__(self, n_first_iterations=10, first_model='IWAE', second_model='AIS_VAE', **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.n_first_iterations = n_first_iterations
        self.dataset = kwargs['dataset']
        self.hidden_dim = kwargs['hidden_dim']
        self.random_z = torch.randn((64, kwargs['hidden_dim']), dtype=torch.float32)
        self.is_model_changed = False
        if first_model == 'IWAE':
            self.proxy_model = IWAE(**kwargs)
        else:
            self.proxy_model = VAE(**kwargs)

        if second_model == 'AIS_VAE':
            self.main_model = AIS_VAE(**kwargs)
        else:
            self.main_model = ULA_VAE(**kwargs)
        # Name, which is used for logging
        self.name = kwargs['name']
        self.transitions_nll = nn.ModuleList(
            [HMC(n_leapfrogs=3, step_size=0.05, use_barker=False)
             for _ in range(5 - 1)])
        for p in self.transitions_nll.parameters():
            p.requires_grad_(False)

    @property
    def current_model(self):
        if self.current_epoch < self.n_first_iterations:
            return self.proxy_model
        else:
            if not self.is_model_changed:
                self.is_model_changed = True
                self.main_model.encoder_net = copy.deepcopy(self.proxy_model.encoder_net)
                self.main_model.decoder_net = copy.deepcopy(self.proxy_model.decoder_net)
                self.main_model.num_samples = 1
                self.configure_optimizers()
            return self.main_model

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        if self.current_epoch < self.n_first_iterations:
            d = self.current_model.training_step(batch, batch_idx)
        else:
            d = self.current_model.training_step(batch, batch_idx, optimizer_idx)
        return d

    def validation_step(self, batch, batch_idx):
        d = self.current_model.validation_step(batch, batch_idx)
        if (self.current_epoch == 49):
            nll = self.current_model.evaluate_nll(batch=batch,
                                                  beta=torch.linspace(0., 1., 5, device=batch[0].device,
                                                                      dtype=torch.float32))
            d.update({"nll": nll})
        return d

    def configure_optimizers(self):
        if self.current_epoch < self.n_first_iterations:
            return torch.optim.Adam(self.current_model.parameters(), lr=1e-3)
        else:
            lambda_lr = lambda epoch: 10 ** (-epoch / 7.0)
            decoder_optimizer = torch.optim.Adam(self.current_model.decoder_net.parameters(), lr=1e-3, eps=1e-4)
            decoder_scheduler_lr = torch.optim.lr_scheduler.LambdaLR(decoder_optimizer, lambda_lr)

            encoder_optimizer = torch.optim.Adam(
                list(self.current_model.encoder_net.parameters()) + list(self.current_model.transitions.parameters()),
                lr=1e-3, eps=1e-4)
            encoder_scheduler_lr = torch.optim.lr_scheduler.LambdaLR(encoder_optimizer, lambda_lr)
            return [decoder_optimizer, encoder_optimizer], [decoder_scheduler_lr, encoder_scheduler_lr]

    def validation_epoch_end(self, outputs):
        # Some stuff, which is needed for logging and tensorboard
        if "val_loss" in outputs[0].keys():
            val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
            self.logger.experiment.add_scalar(f'{self.dataset}/{self.name}/avg_val_loss', val_loss, self.current_epoch)
        else:
            val_loss = torch.stack([x['val_loss_enc'] for x in outputs]).mean()
            val_loss_dec = torch.stack([x['val_loss_dec'] for x in outputs]).mean()
            self.logger.experiment.add_scalar(f'{self.dataset}/{self.name}/avg_val_loss_enc', val_loss,
                                              self.current_epoch)
            self.logger.experiment.add_scalar(f'{self.dataset}/{self.name}/avg_val_loss_dec', val_loss_dec,
                                              self.current_epoch)
            if "val_loss_score_match" in outputs[0].keys():
                val_loss_score_match = torch.stack([x['val_loss_score_match'] for x in outputs]).mean()
                self.logger.experiment.add_scalar(f'{self.dataset}/{self.name}/avg_val_loss_score_match',
                                                  val_loss_score_match,
                                                  self.current_epoch)

        if "acceptance_rate" in outputs[0].keys():
            acceptance = torch.stack([x['acceptance_rate'] for x in outputs]).mean(0)
            for i in range(len(acceptance)):
                self.logger.experiment.add_scalar(f'{self.dataset}/{self.name}/acceptance_rate_{i}',
                                                  acceptance[i].item(),
                                                  self.current_epoch)

        if "nll" in outputs[0].keys():
            nll = torch.stack([x['nll'] for x in outputs]).mean(0)
            self.logger.experiment.add_scalar(f'{self.dataset}/{self.name}/nll', nll,
                                              self.current_epoch)

        if self.dataset.lower() in ['cifar', 'mnist', 'omniglot', 'celeba']:
            if self.dataset.lower().find('cifar') > -1:
                x_hat = torch.sigmoid(self.current_model(self.random_z.to(val_loss.device))).view((-1, 3, 32, 32))
            elif self.dataset.lower().find('mnist') > -1:
                x_hat = torch.sigmoid(self.current_model(self.random_z.to(val_loss.device))).view((-1, 1, 28, 28))
            elif self.dataset.lower().find('omniglot') > -1:
                x_hat = torch.sigmoid(self.current_model(self.random_z.to(val_loss.device))).view((-1, 1, 105, 105))
            elif self.dataset.lower().find('celeba') > -1:
                x_hat = torch.sigmoid(self.current_model(self.random_z.to(val_loss.device))).view((-1, 3, 64, 64))
            grid = torchvision.utils.make_grid(x_hat)
            self.logger.experiment.add_image(f'{self.dataset}/{self.name}/image', grid, self.current_epoch)
        else:
            pass
