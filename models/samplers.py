import torch
import torch.nn as nn


def acceptance_ratio(log_t, log_1_t, use_barker, return_pre_alphas=False):
    if use_barker:
        current_log_alphas_pre = log_t - log_1_t
    else:
        current_log_alphas_pre = torch.min(log_t, torch.zeros_like(log_t))

    log_probs = torch.log(torch.rand_like(log_t))
    a = log_probs <= current_log_alphas_pre

    if use_barker:
        current_log_alphas = current_log_alphas_pre
        current_log_alphas[~a] = (-log_1_t)[~a]
    else:
        expression = torch.ones_like(current_log_alphas_pre) - torch.exp(current_log_alphas_pre)
        corr_expression = torch.log(expression + 1e-8)
        current_log_alphas = current_log_alphas_pre
        current_log_alphas[~a] = corr_expression[~a]

    if not return_pre_alphas:
        return a, current_log_alphas
    else:
        return a, current_log_alphas, current_log_alphas_pre


def compute_grad(z, target, x):
    flag = z.requires_grad  # True, if requires grad (means that we propagate gradients to some parameters)
    if not flag:
        z_ = z.detach().requires_grad_(True)
    else:
        z_ = z.requires_grad_(True)  ##  Do I need to clone it?
    with torch.enable_grad():
        grad = _get_grad(z=z_, target=target, x=x)
        if not flag:
            grad = grad.detach()
            z_.requires_grad_(False)
        return grad


def _get_grad(z, target, x=None):
    s = target(x=x, z=z)
    grad = torch.autograd.grad(s.sum(), z, create_graph=True, only_inputs=True)[0]
    return grad


def run_chain(kernel, z_init, target, x=None, n_steps=100, return_trace=False, burnin=0):
    samples = z_init
    if not return_trace:
        for _ in range(burnin + n_steps):
            samples = kernel.make_transition(z=samples, target=target, x=x)[0].detach()
        return samples
    else:
        final = torch.tensor([], device=z_init.device, dtype=torch.float32)
        for i in range(burnin + n_steps):
            samples = kernel.make_transition(z=samples, target=target, x=x)[0].detach()
            if i >= burnin:
                final = torch.cat([final, samples])
        return final


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
        std_normal = torch.distributions.Normal(loc=self.zero, scale=self.one)

        ############ Then we compute new points and densities ############
        z_upd, p_upd = self.forward_step(z_old=z_old, p_old=p_old, target=target, x=x)

        target_log_density_f = target(z=z_upd, x=x) + std_normal.log_prob(p_upd).sum(-1)
        target_log_density_old = target(z=z_old, x=x) + std_normal.log_prob(p_old).sum(-1)

        log_t = target_log_density_f - target_log_density_old
        log_1_t = torch.logsumexp(torch.cat([torch.zeros_like(log_t).view(-1, 1),
                                             log_t.view(-1, 1)], dim=-1), dim=-1)  # log(1+t)

        a, current_log_alphas = acceptance_ratio(log_t=log_t, log_1_t=log_1_t, use_barker=self.use_barker)

        z_new = z_upd
        z_new[~a] = z_old[~a]

        p_new = -p_upd
        p_new[~a] = -p_old[~a]

        return z_new, p_new, a.to(torch.float32), current_log_alphas

    def make_transition(self, z, target, x=None, p=None):
        if p is None:
            p = torch.randn_like(z)
        if self.partial_ref:
            p = p * self.alpha + torch.sqrt(self.one - self.alpha ** 2) * torch.randn_like(p)
        z_new, p_new, a, current_log_alphas = self._make_transition(z_old=z,
                                                                    target=target, p_old=p, x=x)
        return z_new, p_new, a, current_log_alphas

    def forward_step(self, z_old, x=None, target=None, p_old=None):
        z_, p_ = self._forward_step(z_old=z_old, x=x, target=target, p_old=p_old)
        return z_, p_

    def get_grad(self, z, target, x=None):
        grad = compute_grad(z, target, x)
        return grad


class MALA(nn.Module):
    def __init__(self, step_size, use_barker, learnable):
        '''
        :param step_size: stepsize for leapfrog
        :param use_barker: If True -- Barker ratios applied. MH otherwise
        :param learnable: whether learnable (usage for Met model) or not
        '''
        super().__init__()
        self.use_barker = use_barker  # if use barker ratio
        self.learnable = learnable  # if stepsize are learnable
        self.register_buffer('zero', torch.tensor(0., dtype=torch.float32))
        self.register_buffer('one', torch.tensor(1., dtype=torch.float32))
        self.log_stepsize = nn.Parameter(torch.log(torch.tensor(step_size, dtype=torch.float32)),
                                         requires_grad=learnable)

    @property
    def step_size(self):
        return torch.exp(self.log_stepsize)

    def _forward_step(self, z_old, x=None, target=None):
        eps = torch.randn_like(z_old)
        forward_grad = self.get_grad(z=z_old,
                                     target=target,
                                     x=x)
        update = torch.sqrt(2 * self.step_size) * eps + self.step_size * forward_grad
        return z_old + update, update, eps, forward_grad

    def make_transition(self, z, target, x=None):
        """
        Input:
        z_old - current position
        target - target distribution
        x - data object (optional)
        Output:
        z_new - new position
        current_log_alphas - current log_alphas, corresponding to sampled decision variables
        a - decision variables (0 or +1)
        """
        ############ Then we compute new points and densities ############
        std_normal = torch.distributions.Normal(loc=self.zero, scale=self.one)

        z_upd, update, eps, forward_grad = self._forward_step(z_old=z, x=x, target=target)

        target_log_density_upd = target(z=z_upd, x=x)
        target_log_density_old = target(z=z, x=x)

        eps_reverse = (-update - self.step_size * self.get_grad(z=z_upd, target=target, x=x)) / torch.sqrt(
            2 * self.step_size)
        proposal_density_numerator = std_normal.log_prob(eps_reverse).sum(1)
        proposal_density_denominator = std_normal.log_prob(eps).sum(1)

        log_t = target_log_density_upd - target_log_density_old - proposal_density_denominator + proposal_density_numerator  # - (eps_reverse.detach() * eps_reverse).sum(1)
        log_1_t = torch.logsumexp(torch.cat([torch.zeros_like(log_t).view(-1, 1),
                                             log_t.view(-1, 1)], dim=-1), dim=-1)  # log(1+t)

        a, current_log_alphas = acceptance_ratio(log_t, log_1_t, use_barker=self.use_barker)

        z_new = torch.empty_like(z_upd)
        z_new[a] = z_upd[a]
        z_new[~a] = z[~a]

        return z_new, a.to(torch.float32), current_log_alphas, forward_grad

    def get_grad(self, z, target, x=None):
        grad = compute_grad(z, target, x)
        return grad


class ULA(nn.Module):
    def __init__(self, step_size, learnable=False, transforms=None, ula_skip_threshold=0.0):
        '''
        :param step_size: stepsize for leapfrog
        :param learnable: whether learnable (usage for Met model) or not
        '''
        super().__init__()
        self.learnable = learnable
        self.ula_skip_threshold = ula_skip_threshold
        self.register_buffer('zero', torch.tensor(0., dtype=torch.float32))
        self.register_buffer('one', torch.tensor(1., dtype=torch.float32))
        self.log_stepsize = nn.Parameter(torch.log(torch.tensor(step_size, dtype=torch.float32)),
                                         requires_grad=learnable)
        self.transforms = False
        self.add_nn = None
        self.scale_nn = None
        self.score_matching = False
        if transforms is not None:
            self.transforms = True
            self.add_nn = transforms()
            self.scale_nn = transforms()  ###just test with step size at the moment
            # self.scale_nn = lambda z, sign: 1.
            self.score_matching = True

    @property
    def step_size(self):
        return torch.exp(self.log_stepsize)

    def _forward_step(self, z_old, x=None, target=None):
        eps = torch.randn_like(z_old)
        self.log_jac = torch.zeros_like(z_old[:, 0])
        if not self.transforms:
            add = torch.zeros_like(z_old)
            forward_grad = self.get_grad(
                z=z_old,
                target=target,
                x=x)
            update = torch.sqrt(2 * self.step_size) * eps + self.step_size * forward_grad
            z_new = z_old + update
            eps_reverse = (z_old - z_new - self.step_size * self.get_grad(z=z_new, target=target, x=x)) / torch.sqrt(
                2 * self.step_size)
            score_match_cur = add
        else:
            add = self.add_nn(z=z_old, x=x)
            z_new = z_old + self.step_size * add + torch.sqrt(2 * self.step_size) * eps
            eps_reverse = (z_old - z_new - self.step_size * self.add_nn(z=z_new, x=x)) / torch.sqrt(2 * self.step_size)
            score_match_cur = (add - self.get_grad(z=z_old, target=target, x=x)) ** 2
            forward_grad = add
        return z_new, eps, eps_reverse, score_match_cur, forward_grad

    def scale_transform(self, z, sign='+'):
        S = torch.sigmoid(self.scale_nn(z))
        sign = {"+": 1., "-": -1.}[sign]
        self.log_jac += torch.sum(torch.log(S), dim=1) * sign
        return S

    def make_transition(self, z, target, x=None, reverse_kernel=None, mu_amortize=None):
        """
        Input:
        z_old - current position
        target - target distribution
        x - data object (optional)
        Output:
        z_new - new position
        current_log_alphas - current log_alphas, corresponding to sampled decision variables
        a - decision variables (0 or +1)
        """

        ############ Then we compute new points and densities ############
        std_normal = torch.distributions.Normal(loc=self.zero, scale=self.one)

        z_upd, eps, eps_reverse, score_match_cur, forward_grad = self._forward_step(z_old=z, x=x, target=target)

        if reverse_kernel is None:
            proposal_density_numerator = std_normal.log_prob(eps_reverse).sum(1)
        else:
            mu, logvar = reverse_kernel(torch.cat([z_upd, mu_amortize], dim=1))
            proposal_density_numerator = torch.distributions.Normal(loc=mu + z_upd, scale=torch.exp(0.5 * logvar)).log_prob(
                z).sum(1)

        proposal_density_denominator = std_normal.log_prob(eps).sum(1)

        z_new = z_upd

        ###
        with torch.no_grad():
            target_log_density_upd = target(z=z_upd, x=x)
            target_log_density_old = target(z=z, x=x)
            log_t = target_log_density_upd + proposal_density_numerator - target_log_density_old - proposal_density_denominator + self.log_jac
            log_1_t = torch.logsumexp(torch.cat([torch.zeros_like(log_t).view(-1, 1),
                                                 log_t.view(-1, 1)], dim=-1), dim=-1)  # log(1+t)
            if self.ula_skip_threshold > 0.:
                a, _, current_log_alphas_pre = acceptance_ratio(log_t, log_1_t, use_barker=False,
                                                                return_pre_alphas=True)
                acceptance_probs = torch.exp(current_log_alphas_pre)
                reject_mask = acceptance_probs <= self.ula_skip_threshold
            else:
                a, _ = acceptance_ratio(log_t, log_1_t, use_barker=False, return_pre_alphas=False)
                reject_mask = torch.zeros_like(a) < -1.
        ###
        if reject_mask.sum():
            # z_new[reject_mask] = z[reject_mask]
            z_new = torch.where(reject_mask[..., None], z, z_new)
            proposal_density_numerator[reject_mask] = torch.zeros_like(proposal_density_numerator[reject_mask])
            proposal_density_denominator[reject_mask] = torch.zeros_like(proposal_density_denominator[reject_mask])
            score_match_cur[reject_mask] = torch.zeros_like(score_match_cur[reject_mask])

        return z_new, proposal_density_numerator - proposal_density_denominator + self.log_jac, a.to(
            torch.float32), score_match_cur, forward_grad

    def get_grad(self, z, target, x=None):
        grad = compute_grad(z, target, x)
        return grad
