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
    return torch.clamp(x, 0) - x * y + torch.log(1 + torch.exp(-torch.abs(x)))


def repeat_data(x, n_samples):
    '''
    Repeats data n_samples times, taking dimensionality into account
    '''
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
    '''
    Context manager, which stops gradients if the flag is True, otherwise computes graph
    '''

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
    '''
    Base class for all VAEs
    '''

    def __init__(self, num_samples, act_func, shape, hidden_dim, net_type,
                 dataset, name="VAE", **kwargs):
        '''

        :param num_samples: how many latent samples per object to use
        :param act_func: activation function
        :param shape: image shape
        :param hidden_dim: hidden dim
        :param name: model name
        :param net_type: network type (conv or fc)
        :param dataset: dataset name
        :param kwargs: specific args
        '''
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
        # Transitions, which are going to be used for NLL estimation
        self.transitions_nll = nn.ModuleList(
            [HMC(n_leapfrogs=3, step_size=0.05, use_barker=False)
             for _ in range(5 - 1)])
        # We dont need these transitions to have trainable parameters
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
        # Encode and reparametrize
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
        """
        Defines joint p(x, z). Needed for transitions-based models (AIS, ULA).
        :param use_true_decoder: When true, we use true decoder. Otherwise, could be auxilary model, or true one (if no auxilary).
        :return: joint logdensity of batchsize shape.
        """

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
            return -binary_crossentropy_logits_stable(x_reconst.view(x_reconst.shape[0], -1),
                                                      x.view(x_reconst.shape[0], -1)).sum(-1) + log_Pr

        return density

    def validation_epoch_end(self, outputs):
        # Tensorboard logging
        if "val_loss" in outputs[0].keys():  # if we have single loss
            val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
            self.logger.experiment.add_scalar(f'{self.dataset}/{self.name}/avg_val_loss', val_loss, self.current_epoch)
        else:  # if we have separate losses for encoder and decoder
            val_loss = torch.stack([x['val_loss_enc'] for x in outputs]).mean()
            val_loss_dec = torch.stack([x['val_loss_dec'] for x in outputs]).mean()
            self.logger.experiment.add_scalar(f'{self.dataset}/{self.name}/avg_val_loss_enc', val_loss,
                                              self.current_epoch)
            self.logger.experiment.add_scalar(f'{self.dataset}/{self.name}/avg_val_loss_dec', val_loss_dec,
                                              self.current_epoch)
            if "val_loss_score_match" in outputs[0].keys():  # if we have score matching loss
                val_loss_score_match = torch.stack([x['val_loss_score_match'] for x in outputs]).mean()
                self.logger.experiment.add_scalar(f'{self.dataset}/{self.name}/avg_val_loss_score_match',
                                                  val_loss_score_match,
                                                  self.current_epoch)

        if "acceptance_rate" in outputs[0].keys():  # if we have acceptance, we plot the mean after each transition
            acceptance = torch.stack([x['acceptance_rate'] for x in outputs]).mean(0)
            for i in range(len(acceptance)):
                self.logger.experiment.add_scalar(f'{self.dataset}/{self.name}/acceptance_rate_{i}',
                                                  acceptance[i].item(),
                                                  self.current_epoch)

        if "nll" in outputs[0].keys():  # if we have nll (typically at epoch 49), we plot it
            nll = torch.stack([x['nll'] for x in outputs]).mean(0)
            self.logger.experiment.add_scalar(f'{self.dataset}/{self.name}/nll', nll,
                                              self.current_epoch)

        if hasattr(self, 'transitions'):  # if we have transitions, we show median stepsize
            for i in range(len(self.transitions)):
                self.logger.experiment.add_scalar(f'{self.dataset}/{self.name}/median_step_size_{i}',
                                                  np.median(self.transitions[i].step_size.cpu().detach().numpy()),
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
        if self.current_epoch % 10 == 9:
            nll = self.evaluate_nll(batch=batch,
                                    beta=torch.linspace(0., 1., 5, device=batch[0].device, dtype=torch.float32))
            d.update({"nll": nll})
        return d

    def evaluate_nll(self, batch, beta):
        # the function evaluates NLL using AIS
        x, _ = batch
        with torch.no_grad():
            n_samples = 10  # num samples for NLL estimation per data object
            z, mu, logvar = self.enc_rep(x, n_samples=n_samples)
            x = repeat_data(x, n_samples)
            init_logdensity = lambda z: torch.distributions.Normal(loc=mu, scale=torch.exp(0.5 * logvar)).log_prob(
                z).sum(-1)
            annealing_logdens = lambda beta: lambda z, x: (1. - beta) * init_logdensity(
                z=z) + beta * self.joint_logdensity()(
                z=z,
                x=x)
            sum_log_weights = (beta[1] - beta[0]) * (self.joint_logdensity()(z=z, x=x) - init_logdensity(z))

            for i in range(1, len(beta) - 1):
                z = self.one_transition(current_num=i - 1, z=z, x=x,
                                        annealing_logdens=annealing_logdens(beta=beta[i]), nll=True)[0]

                sum_log_weights += (beta[i + 1] - beta[i]) * (self.joint_logdensity()(z=z, x=x) - init_logdensity(z=z))
            sum_log_weights = sum_log_weights.view(n_samples, batch[0].shape[0])
            batch_nll_estimator = -torch.logsumexp(sum_log_weights,
                                                   dim=0) + torch.log(torch.tensor(n_samples, dtype=torch.float32,
                                                                                   device=x.device))  # Should be a vector of batchsize containing nll estimator for each term of the batch

            return torch.mean(batch_nll_estimator)


class VAE(Base):
    def loss_function(self, recon_x, x, mu, logvar):
        batch_size = mu.shape[0] // self.num_samples

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

        log_Pr = torch.distributions.Normal(loc=torch.tensor(0., device=x.device, dtype=torch.float32),
                                            scale=torch.tensor(1., device=x.device, dtype=torch.float32)).log_prob(
            z).view((self.num_samples, -1, self.hidden_dim)).sum(-1)

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
            (self.num_samples, batch_size, -1)).sum(-1) - log_jac.view((self.num_samples, -1)), dim=0)
        log_Pr = (-0.5 * z_transformed ** 2).view(
            (self.num_samples, batch_size, -1)).mean(0).sum(-1)
        KLD = torch.mean(log_Q - log_Pr)
        loss = BCE + KLD
        return loss

    def step(self, batch):
        x, _ = batch
        z, mu, logvar = self.enc_rep(x, self.num_samples)
        x = repeat_data(x, self.num_samples)
        z_transformed, log_jac = self.Flow(z)
        x_hat = self(z_transformed)
        loss = self.loss_function(recon_x=x_hat, x=x, mu=mu, logvar=logvar, z=z, z_transformed=z_transformed,
                                  log_jac=log_jac)
        return loss, x_hat, z_transformed


class BaseMCMC(Base):
    '''
    Base class for MCMC transition-based VAEs
    '''

    def __init__(self, step_size, K, grad_skip_val, grad_clip_val, use_cloned_decoder,
                 learnable_transitions, variance_sensitive_step, acceptance_rate_target, annealing_scheme='linear',
                 **kwargs):
        '''

        :param step_size: stepsize for transitions
        :param K: number of transitions
        :param grad_skip_val: threshold for gradient norm to skip gradient update. Set to 0 if dont need it
        :param grad_clip_val: clip gradients to be no more than this value. Set to 0 if dont need it
        :param use_cloned_decoder: Flag, if true we use cloned version of decoder to simplify grads for decoder through alphas
        :param learnable_transitions: Flag, if true we train stepsize of transitions
        :param variance_sensitive_step: Flag, if true we adapt stepsize in a variance awared manner. If false, adapt it to match target acceptance rate
        :param acceptance_rate_target: Target acceptance rate. Stepsize will be adjusted to fit it.
        :param annealing_scheme: 'linear', 'sigmoidal', 'all_learnable'
        :param kwargs: specific arguments
        '''
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.learnable_transitions = learnable_transitions
        self.K = K
        self.variance_sensitive_step = variance_sensitive_step
        if self.variance_sensitive_step:
            self.gamma_0 = [0.1 for _ in range(self.K)]  # an auxilary multipliers to adjust stepsize
        self.epsilons = [step_size for _ in range(self.K)]
        self.acceptance_rate_target = acceptance_rate_target  # target acceptance rate
        self.epsilon_decrease_alpha = 0.998
        self.epsilon_increase_alpha = 1.002
        self.epsilon_min = 0.001  # minimal possible stepsize
        self.epsilon_max = 0.5  # maximal possible stepsize
        self.annealing_scheme = annealing_scheme
        linear_beta = torch.tensor(np.linspace(0., 1., self.K + 2), dtype=torch.float32)
        self.register_buffer('linear_beta', linear_beta)
        if self.annealing_scheme == 'sigmoidal':
            self.tempering_alpha = nn.Parameter(torch.tensor(-3., dtype=torch.float32))
        elif self.annealing_scheme == 'all_learnable':
            self.tempering_beta_logits = nn.Parameter(torch.zeros(self.K, dtype=torch.float32))
        else:
            raise ValueError('Please, select temrering scheme, which is one of [linear, sigmoidal, all_learnable].')
        self.grad_skip_val = grad_skip_val
        self.grad_clip_val = grad_clip_val
        self.use_cloned_decoder = use_cloned_decoder
        if self.use_cloned_decoder:
            # In case we use cloned decoder, we create an exact copy of a true one in the beginning. After that, it will be trained in its own manner
            self.cloned_decoder = copy.deepcopy(self.decoder_net)

    def configure_optimizers(self):
        # all_params = list(self.decoder_net.parameters()) + list(self.encoder_net.parameters()) + list(
        #     self.transitions.parameters())
        # ## If we are using cloned decoder to approximate the true one, we add its params to inference optimizer
        # if self.use_cloned_decoder:
        #     all_params += list(self.cloned_decoder.parameters())

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.25)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=10, epochs=5)
        # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.01, step_size_up=20,
        #                                               mode="exp_range", gamma=0.85)
        return [optimizer], [scheduler]

    def run_transitions(self, z, x, mu, logvar):
        '''
        The function maintains the logic of transitions. It defines target, run specific transition, returns output
        :param z: current state
        :param x: data batch
        :param mu: parameter mu, returned by encoder
        :param logvar: parameter logvar, returned by encoder
        :return: List with specific content
        '''
        init_logdensity = lambda z: torch.distributions.Normal(loc=mu, scale=torch.exp(0.5 * logvar)).log_prob(
            z).sum(-1)

        annealing_logdens = lambda beta: lambda z, x: (1. - beta) * init_logdensity(
            z=z) + beta * self.joint_logdensity()(
            z=z,
            x=x)

        output = self.specific_transitions(
            z=z,
            x=x,
            init_logdensity=init_logdensity,
            annealing_logdens=annealing_logdens
        )

        return output

    def update_stepsize(self, accept_rate=None, current_tran_id=None, current_gradient_batch=None):
        '''
        Stepsize update machinery.
        :param accept_rate: List of mean acceptance rates after each transitions.
        :param current_tran_id: Current transition id
        :param current_gradient_batch: Current batch of gradients of target logdensity wrt inputs
        :return:
        '''
        if not self.variance_sensitive_step:
            accept_rate = list(accept_rate.cpu().data.numpy())
            for l in range(0, self.K):
                if accept_rate[l] < self.acceptance_rate_target:
                    self.epsilons[l] *= self.epsilon_decrease_alpha
                else:
                    self.epsilons[l] *= self.epsilon_increase_alpha

                if self.epsilons[l] < self.epsilon_min:
                    self.epsilons[l] = self.epsilon_min
                if self.epsilons[l] > self.epsilon_max:
                    self.epsilons[l] = self.epsilon_max
                self.transitions[l].log_stepsize.data = torch.tensor(np.log(self.epsilons[l]), dtype=torch.float32,
                                                                     device=self.transitions[l].log_stepsize.device)
        elif self.variance_sensitive_step:
            with torch.no_grad():
                gradient_std = torch.std(current_gradient_batch, dim=0)
                self.epsilons[current_tran_id] = 0.9 * self.epsilons[current_tran_id] + 0.1 * self.gamma_0[
                    current_tran_id] / (gradient_std + 1.)
                self.transitions[current_tran_id].log_stepsize.data = torch.log(self.epsilons[current_tran_id])
                if accept_rate.mean() < self.acceptance_rate_target:
                    self.gamma_0[current_tran_id] *= 0.99
                else:
                    self.gamma_0[current_tran_id] *= 1.02
        else:
            raise ValueError

    def get_betas(self, ):
        if self.annealing_scheme == 'linear':
            betas = self.linear_beta
        elif self.annealing_scheme == 'sigmoidal':
            beta_0 = torch.sigmoid(-self.tempering_alpha)
            beta_K_1 = torch.sigmoid(self.tempering_alpha)
            betas_unnormed = torch.sigmoid(self.tempering_alpha * (2 * self.linear_beta - 1))
            betas = (betas_unnormed - beta_0) / (beta_K_1 - beta_0)

        elif self.annealing_scheme == 'all_learnable':
            betas = torch.cat([torch.zeros(1, dtype=torch.float32, device=self.tempering_beta_logits.device),
                               torch.sigmoid(self.tempering_beta_logits),
                               torch.ones(1, dtype=torch.float32, device=self.tempering_beta_logits.device)])
        else:
            raise ValueError('Please, select temrering scheme, which is one of [linear, sigmoidal, all_learnable].')
        return betas

    def step(self, batch):
        x, _ = batch
        z, mu, logvar = self.enc_rep(x, self.num_samples)
        x = repeat_data(x, self.num_samples)
        z_transformed, sum_log_weights, _, all_acceptance = self.run_transitions(z=z, x=x, mu=mu, logvar=logvar)
        x_hat = self(z_transformed)
        loss = self.loss_function(z_transformed, x_hat, x, mu, logvar, sum_log_weights)
        return loss, all_acceptance, z_transformed

    def validation_step(self, batch, batch_idx):
        output = self.step(batch)
        d = {"val_loss": output[0], "acceptance_rate": output[1].mean(1)}
        if self.current_epoch % 10 == 9:
            nll = self.evaluate_nll(batch=batch,
                                    beta=torch.linspace(0., 1., 5, device=batch[0].device, dtype=torch.float32))
            d.update({"nll": nll})
        return d


class AIWAE(BaseMCMC):
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

    def specific_transitions(self, z, x, init_logdensity, annealing_logdens, inference_part=False):
        all_acceptance = torch.tensor([], dtype=torch.float32, device=x.device)
        beta = self.get_betas()
        sum_log_weights = (beta[1] - beta[0]) * (self.joint_logdensity()(z=z, x=x) - init_logdensity(z))
        z_transformed = z
        for i in range(1, self.K - 1):
            z_transformed, current_log_alphas, directions = self.one_transition(current_num=i - 1,
                                                                                z=z_transformed,
                                                                                x=x,
                                                                                annealing_logdens=annealing_logdens(
                                                                                    beta=beta[i]))
            sum_log_weights += (beta[i + 1] - beta[i]) * (
                    self.joint_logdensity()(z=z_transformed, x=x) - init_logdensity(z=z_transformed))
            all_acceptance = torch.cat([all_acceptance, directions[None]])

        z_transformed, current_log_alphas, directions = self.one_transition(current_num=self.K - 2,
                                                                            z=z_transformed,
                                                                            x=x,
                                                                            annealing_logdens=annealing_logdens(
                                                                                beta[self.K - 1]))
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


class AIS_VAE(BaseMCMC):
    '''
    Annealed Importance Sampling VAE.
    '''

    def __init__(self, use_barker, use_alpha_annealing, **kwargs):
        '''

        :param use_barker: Flag, whether to use Barker ratio or not
        :param kwargs: Args for BaseMCMC class
        '''
        super().__init__(**kwargs)
        self.transitions = nn.ModuleList(
            [MALA(step_size=self.epsilons[0], use_barker=use_barker, learnable=self.learnable_transitions)
             for _ in range(self.K)])

        self.use_alpha_annealing = use_alpha_annealing
        self.save_hyperparameters()

    def one_transition(self, current_num, z, x, annealing_logdens, nll=False):
        """

        :param current_num: current transition number
        :param z: current state
        :param x: current batch
        :param annealing_logdens: current target (annealed combination of variational family and induced by decoder)
        :param nll: Whether to compute nll (we use another set of transitions) or not
        :return:
        """
        if nll:
            with torch.no_grad():
                z_new = self.transitions_nll[current_num].make_transition(z=z, x=x, target=annealing_logdens)
            return z_new
        else:
            z_new, directions, current_log_alphas, forward_grad = self.transitions[current_num].make_transition(z=z,
                                                                                                                x=x,
                                                                                                                target=annealing_logdens)
            return z_new, current_log_alphas, directions, forward_grad

    def specific_transitions(self, z, x, init_logdensity, annealing_logdens):
        """
        Specific transition for AIS VAE
        :param z: current state
        :param x: current batch
        :param init_logdensity: Initial log density (induced by encoder)
        :param annealing_logdens: current target (annealed combination of variational family and induced by decoder)
        :return: Specific output: final states, summed low weights, summed log alphas and acceptances
        """
        all_acceptance = torch.tensor([], dtype=torch.float32, device=x.device)
        beta = self.get_betas()
        sum_log_weights = (beta[1] - beta[0]) * (self.joint_logdensity()(z=z, x=x) - init_logdensity(z))
        sum_log_alphas = torch.zeros_like(z[:, 0])
        z_transformed = z
        for i in range(1, self.K + 1):
            z_transformed, current_log_alphas, directions, forward_grad = self.one_transition(current_num=i - 1,
                                                                                              z=z_transformed,
                                                                                              x=x,
                                                                                              annealing_logdens=annealing_logdens(
                                                                                                  beta=beta[i]))
            sum_log_alphas += current_log_alphas

            sum_log_weights += (beta[i + 1] - beta[i]) * (
                    self.joint_logdensity(use_true_decoder=(i == self.K))(z=z_transformed, x=x) - init_logdensity(
                z=z_transformed))

            all_acceptance = torch.cat([all_acceptance, directions[None]])
            if self.variance_sensitive_step:
                self.update_stepsize(accept_rate=directions, current_tran_id=i - 1, current_gradient_batch=forward_grad)

        if not self.variance_sensitive_step:
            self.update_stepsize(accept_rate=all_acceptance.mean(1))

        return z_transformed, sum_log_weights, sum_log_alphas, all_acceptance

    def step(self, batch):
        x, _ = batch
        z, mu, logvar = self.enc_rep(x, self.num_samples)
        x = repeat_data(x, self.num_samples)
        z_transformed, sum_log_weights, sum_log_alphas, all_acceptance = self.run_transitions(z=z,
                                                                                              x=x,
                                                                                              mu=mu,
                                                                                              logvar=logvar)

        loss = self.loss_function(sum_log_alphas=sum_log_alphas, sum_log_weights=sum_log_weights)
        return loss, all_acceptance, z_transformed

    def loss_function(self, sum_log_alphas, sum_log_weights):
        batch_size = sum_log_weights.shape[0] // self.num_samples
        elbo_est = sum_log_weights.view((self.num_samples, batch_size))
        ######
        elbo_est_normalized = 1.

        if self.use_alpha_annealing:
            annealing_coeff = np.minimum(1., 0.1 * self.current_epoch)
        else:
            annealing_coeff = 1.

        if self.num_samples > 1:
            multiplier = (self.num_samples * elbo_est.detach() - elbo_est.detach().sum(0)) / (
                    self.num_samples - 1.)
        else:
            multiplier = elbo_est.detach()

        loss = -torch.mean(elbo_est * elbo_est_normalized) - annealing_coeff * torch.mean(
            multiplier * sum_log_alphas.view((self.num_samples, batch_size)))
        return loss

    def training_step(self, batch, batch_idx):
        x, _ = batch
        z, mu, logvar = self.enc_rep(x, self.num_samples)
        x = repeat_data(x, self.num_samples)
        z_transformed, sum_log_weights, sum_log_alphas, all_acceptance = self.run_transitions(z=z,
                                                                                              x=x,
                                                                                              mu=mu,
                                                                                              logvar=logvar)
        loss = self.loss_function(sum_log_alphas=sum_log_alphas, sum_log_weights=sum_log_weights)

        if (self.grad_skip_val != 0.) or (self.grad_clip_val != 0.):
            optimizer = self.optimizers()
            self.manual_backward(loss, optimizer)
            all_params = list(self.decoder_net.parameters()) + list(self.encoder_net.parameters()) + list(
                self.transitions.parameters())
            ## If we are using cloned decoder to approximate the true one, we add its params to inference optimizer
            if self.use_cloned_decoder:
                all_params += list(self.cloned_decoder.parameters())

            grad_norm = torch.nn.utils.clip_grad_norm_(all_params, self.grad_clip_val).item()
            if self.grad_skip_val != 0.:
                if grad_norm < self.grad_skip_val:
                    optimizer.step()
            else:
                optimizer.step()
        else:
            return {"loss": loss}


class ULA_VAE(BaseMCMC):
    '''
    Class for ULA VAE
    '''

    def __init__(self, use_score_matching, ula_skip_threshold, **kwargs):
        '''

        :param use_score_matching: If true, we are using score matching (we do not estimate the correct gradient)
        :param kwargs: Parameters for base function
        '''
        super().__init__(**kwargs)
        self.score_matching = use_score_matching
        if self.score_matching:
            additional_dims = 2 if kwargs['dataset'] == 'toy' else kwargs['shape'] ** 2
            transforms = lambda: ULA_nn_sm(input=kwargs['hidden_dim'] + additional_dims,
                                           output=kwargs['hidden_dim'],
                                           hidden=(kwargs['hidden_dim'] * 10, kwargs['hidden_dim'] * 10),
                                           h_dim=None)
        else:
            transforms = None
        self.transitions = nn.ModuleList(
            [ULA(step_size=self.epsilons[0], learnable=self.learnable_transitions, transforms=transforms,
                 ula_skip_threshold=ula_skip_threshold) for _ in range(self.K)])
        self.save_hyperparameters()

    def one_transition(self, current_num, z, x, annealing_logdens, nll=False):
        if nll:
            z_new = self.transitions_nll[current_num].make_transition(z=z, x=x,
                                                                      target=annealing_logdens)
            return z_new
        else:
            z_new, current_log_weights, directions, score_match_cur, forward_grad = self.transitions[
                current_num].make_transition(z=z,
                                             x=x,
                                             target=annealing_logdens)
            return z_new, current_log_weights, directions, score_match_cur, forward_grad

    def specific_transitions(self, z, x, init_logdensity, annealing_logdens):
        all_acceptance = torch.tensor([], dtype=torch.float32, device=x.device)
        beta = self.get_betas()
        sum_log_weights = -init_logdensity(z=z)
        loss_sm = torch.zeros_like(z)
        z_transformed = z
        for i in range(1, self.K + 1):
            z_transformed, current_log_weights, directions, score_match_cur, forward_grad = self.one_transition(
                current_num=i - 1,
                z=z_transformed,
                x=x,
                annealing_logdens=annealing_logdens(beta=beta[i]))
            loss_sm += score_match_cur
            sum_log_weights += current_log_weights
            all_acceptance = torch.cat([all_acceptance, directions[None]])
            if self.variance_sensitive_step:
                self.update_stepsize(accept_rate=directions, current_tran_id=i - 1, current_gradient_batch=forward_grad)
        sum_log_weights += self.joint_logdensity(use_true_decoder=True)(z=z_transformed, x=x)
        if not self.variance_sensitive_step:
            self.update_stepsize(accept_rate=all_acceptance.mean(1))
        return z_transformed, sum_log_weights, loss_sm, all_acceptance

    def step(self, batch):
        x, _ = batch
        z, mu, logvar = self.enc_rep(x, self.num_samples)
        x = repeat_data(x, self.num_samples)
        z_transformed, sum_log_weights, loss_sm, all_acceptance = self.run_transitions(z=z,
                                                                                       x=x,
                                                                                       mu=mu,
                                                                                       logvar=logvar)

        loss = self.loss_function(sum_log_weights=sum_log_weights)
        loss_sm = loss_sm.sum(1).mean()
        return loss, all_acceptance, loss_sm, z_transformed

    def loss_function(self, sum_log_weights):
        batch_size = sum_log_weights.shape[0] // self.num_samples
        elbo_est = sum_log_weights.view((self.num_samples, batch_size))
        if self.num_samples > 1:
            with torch.no_grad():
                log_weight = elbo_est - torch.max(elbo_est, 0)[0]  # for stability
                weight = torch.exp(log_weight)
                weight = weight / torch.sum(weight, 0)
                weight = weight.detach()
            loss = -torch.mean(torch.sum(weight * elbo_est, dim=0))
        else:
            elbo_est = elbo_est.sum(0)
            loss = -torch.mean(elbo_est)
        return loss

    def training_step(self, batch, batch_idx):
        x, _ = batch
        z, mu, logvar = self.enc_rep(x, self.num_samples)
        x = repeat_data(x, self.num_samples)
        z_transformed, sum_log_weights, loss_sm, all_acceptance = self.run_transitions(z=z,
                                                                                       x=x,
                                                                                       mu=mu,
                                                                                       logvar=logvar)
        loss = self.loss_function(sum_log_weights) + loss_sm.sum(1).mean()
        if (self.grad_skip_val != 0.) or (self.grad_clip_val != 0.):
            optimizer = self.optimizers()
            self.manual_backward(loss, optimizer)
            all_params = list(self.decoder_net.parameters()) + list(self.encoder_net.parameters()) + list(
                self.transitions.parameters())
            ## If we are using cloned decoder to approximate the true one, we add its params to inference optimizer
            if self.use_cloned_decoder:
                all_params += list(self.cloned_decoder.parameters())

            grad_norm = torch.nn.utils.clip_grad_norm_(all_params, self.grad_clip_val).item()
            if self.grad_skip_val != 0.:
                if grad_norm < self.grad_skip_val:
                    optimizer.step()
            else:
                optimizer.step()
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        output = self.step(batch)
        d = {"val_loss": output[0], "acceptance_rate": output[1].mean(1), "val_loss_score_match": output[2]}
        if self.current_epoch % 10 == 9:
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
        if self.current_epoch % 10 == 9:
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
