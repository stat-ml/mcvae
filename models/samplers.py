import torch
import torch.nn as nn


class HMC(nn.Module):
    def __init__(self, n_leapfrogs, step_size, use_barker=False, partial_ref=False, learnable=False):
        '''
        :param n_leapfrogs: number of leapfrog iterations
        :param step_size: stepsize for leapfrog
        :param use_barker: If True -- Barker ratios applied. MH otherwise
        :param partial_ref: whether use partial refresh or not
        :param learnable: whether learnable (usage for Met model) or not
        '''
        super().__init__()
        self.n_leapfrogs = n_leapfrogs
        self.use_barker = use_barker
        self.partial_ref = partial_ref
        self.learnable = learnable
        self.register_buffer('zero', torch.tensor(0., dtype=torch.float32))
        self.register_buffer('one', torch.tensor(1., dtype=torch.float32))
        self.alpha_logit = nn.Parameter(self.zero, requires_grad=learnable)
        self.log_stepsize = nn.Parameter(torch.log(torch.tensor(step_size, dtype=torch.float32)),
                                         requires_grad=learnable)

    @property
    def step_size(self):
        return torch.exp(self.log_stepsize)

    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_logit)

    def _forward_step(self, z_old, x=None, target=None, p_old=None):
        p_ = p_old + self.step_size / 2. * self.get_grad(z=z_old, target=target,
                                                         x=x)
        z_ = z_old
        for l in range(self.n_leapfrogs):
            z_ = z_ + self.step_size * p_
            if (l != self.n_leapfrogs - 1):
                p_ = p_ + self.step_size * self.get_grad(z=z_, target=target,
                                                         x=x)
        p_ = p_ + self.step_size / 2. * self.get_grad(z=z_, target=target,
                                                      x=x)
        return z_, p_

    def _make_transition(self, z_old, target, p_old=None, x=None):
        uniform = torch.distributions.Uniform(low=self.zero, high=self.one)
        std_normal = torch.distributions.Normal(loc=self.zero, scale=self.one)

        if self.partial_ref:
            log_jac = p_old.shape[1] * torch.log(self.alpha) * torch.ones_like(z[:, 0])
        else:
            log_jac = torch.zeros_like(z_old[:, 0])
        ############ Then we compute new points and densities ############
        z_upd, p_upd = self.forward_step(z_old=z_old, p_old=p_old, target=target, x=x)

        target_log_density_f = target(z=z_upd, x=x) + std_normal.log_prob(p_upd).sum(-1)
        target_log_density_old = target(z=z_old, x=x) + std_normal.log_prob(p_old).sum(-1)

        log_t = target_log_density_f - target_log_density_old
        log_1_t = torch.logsumexp(torch.cat([torch.zeros_like(log_t).view(-1, 1),
                                             log_t.view(-1, 1)], dim=-1), dim=-1)  # log(1+t)
        if self.use_barker:
            current_log_alphas_pre = log_t - log_1_t
        else:
            current_log_alphas_pre = torch.min(torch.zeros_like(log_t), log_t)

        log_probs = torch.log(uniform.sample((z_upd.shape[0],)))
        a = torch.where(log_probs < current_log_alphas_pre, torch.ones_like(log_probs), torch.zeros_like(log_probs))

        if self.use_barker:
            current_log_alphas = torch.where((a == 0.), -log_1_t, current_log_alphas_pre)
        else:
            expression = torch.log(self.one - torch.exp(log_t) + 1e-8)
            current_log_alphas = torch.where((a == 0), expression, current_log_alphas_pre)

        z_new = torch.where((a == 0.)[:, None], z_old, z_upd)
        p_new = torch.where((a == 0.)[:, None], -p_old, -p_upd)  ##

        return z_new, p_new, a, current_log_alphas

    def make_transition(self, z, target, x=None, p=None):
        if p is None:
            p = torch.randn_like(z.shape)
        if self.partial_ref:
            p = p * self.alpha + torch.sqrt(self.one - self.alpha ** 2) * torch.randn_like(p)
        z_new, p_new, a, current_log_alphas = self._make_transition(z_old=z,
                                                                    target=target, p_old=p, x=x)
        return z_new, p_new, a, current_log_alphas

    def forward_step(self, z_old, x=None, target=None, p_old=None):
        z_, p_ = self._forward_step(z_old=z_old, x=x, target=target, p_old=p_old)
        if not self.learnable:
            z_.requires_grad_(False)
            p_.requires_grad_(False)
        return z_, p_

    def get_grad(self, z, target, x=None):
        z = z.detach().requires_grad_(True)
        with torch.enable_grad():
            grad = self._get_grad(z=z, target=target, x=x)
            return grad

    def _get_grad(self, z, target, x=None):
        s = target(x=x, z=z)
        grad = torch.autograd.grad(s.sum(), z)[0]
        return grad

    def run_chain(self, z_init, target, x, n_steps=100, return_trace=False, burnin=0):
        samples = z_init
        if not return_trace:
            for _ in range(n_steps):
                samples = self.make_transition(z=samples, target=target, x=x)[0]
            return samples
        else:
            final = torch.tensor([], device=self.one.device, dtype=torch.float32)
            for i in range(burnin + n_steps):
                samples = self.make_transition(z=samples, target=target, x=x)[0]
                if i >= burnin:
                    final = torch.cat([final, samples])
            return final
