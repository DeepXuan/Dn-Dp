import math
import torch
from torch import device, nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm


def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        image_size,
        channels=3,
        loss_type='l1',
        mean_type='x0', # 'x0' or 'epsilon'
        conditional=True,
        schedule_opt=None
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        # self.denoise_fn = denoise_fn
        self.denoise_fn = None
        self.loss_type = loss_type
        self.mean_type = mean_type
        self.conditional = conditional
        if schedule_opt is not None:
            pass
            # self.set_new_noise_schedule(schedule_opt)

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='none').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='none').to(device)
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise
    
    def predict_noise_from_start(self, x_t, t, x0):
        return (self.sqrt_recip_alphas_cumprod[t] * x_t - x0) \
            / self.sqrt_recipm1_alphas_cumprod[t]
    
    def predict_start_from_mu(self, x_t, t, mu):
        return (  
            1.0 / self.posterior_mean_coef1[t] * mu
            - self.posterior_mean_coef2[t] / self.posterior_mean_coef1[t] * x_t
        )
    
    def predict_start_from_v(self, x_t, t, v):
        #applies the formula x0_pred = sqrt(alpha) * x_t - sqrt(1-alpha) * v_pred
        assert x_t.shape == v.shape
        return (
            self.sqrt_alphas_cumprod[t] * x_t
            - self.sqrt_one_minus_alphas_cumprod[t] * v
        )
    
    def predict_start_from_epsilon_and_start(self, x_t, t, x0_and_epsilon):
        #applies the formula x0_pred = sqrt(alpha) * x_t - sqrt(1-alpha) * v_pred
        c = x0_and_epsilon.shape[1]
        assert c % 2 == 0
        x0 = x0_and_epsilon[:, : c // 2, ...]
        epsilon = x0_and_epsilon[:, c // 2:, ...]
        return (
            self.sqrt_alphas_cumprod[t] * (x_t - self.sqrt_one_minus_alphas_cumprod[t] * epsilon)
            + (self.sqrt_one_minus_alphas_cumprod[t] ** 2) * x0
        )
    
    def _predict_xstart_from_epsilon_and_start(self, x_t, t, x0_and_epsilon):
        #applies the formula x0_pred = sqrt(alpha) * x_t - sqrt(1-alpha) * v_pred
        c = x0_and_epsilon.shape[1]
        assert c % 2 == 0
        x0 = x0_and_epsilon[:, : c // 2, ...]
        epsilon = x0_and_epsilon[:, c // 2:, ...]
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * 
            (x_t - _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * epsilon)
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod ** 2, t, x_t.shape) * x0
        )
    
    def _predict_xstart_from_mu(self, x_t, t, mu):
        assert x_t.shape == mu.shape
        return (  # (mu - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * mu
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )
    
    def _predict_xstart_from_v(self, x_t, t, v):
        #applies the formula x0_pred = sqrt(alpha) * x_t - sqrt(1-alpha) * v_pred
        assert x_t.shape == v.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )
    
    def _predict_xstart_from_noise(self, x_t, t, noise):
        assert x_t.shape == noise.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * \
            x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t+1]]).repeat(batch_size, 1).to(x.device)
        if condition_x is not None:
            if self.mean_type == 'x0':
                x_recon = self.denoise_fn((condition_x, x), noise_level)
                # x_recon = self.denoise_fn(torch.cat([condition_x, x], dim=1), torch.FloatTensor([t]).to(x.device))
            elif self.mean_type == 'epsilon':
                x_recon = self.predict_start_from_noise(
                    x, t=t, noise=self.denoise_fn((condition_x, x), noise_level))
            elif self.mean_type == 'mu':
                x_recon = self.predict_start_from_mu(
                    x, t=t, mu=self.denoise_fn((condition_x, x), noise_level))
            elif self.mean_type == 'v':
                x_recon = self.predict_start_from_v(
                    x, t=t, v=self.denoise_fn((condition_x, x), noise_level))
            elif self.mean_type == 'x0_and_epsilon':
                x_recon = self.predict_start_from_epsilon_and_start(
                    x, t=t, x0_and_epsilon=self.denoise_fn((condition_x, x), noise_level))
        else:
            if self.mean_type == 'x0':
                x_recon = self.denoise_fn(x, noise_level)
            elif self.mean_type == 'epsilon':
                x_recon = self.predict_start_from_noise(
                    x, t=t, noise=self.denoise_fn(x, noise_level))
            elif self.mean_type == 'mu':
                x_recon = self.predict_start_from_mu(
                    x, t=t, mu=self.denoise_fn(x, noise_level))
            elif self.mean_type == 'v':
                x_recon = self.predict_start_from_v(
                    x, t=t, v=self.denoise_fn(x, noise_level))
            elif self.mean_type == 'x0_and_epsilon':
                x_recon = self.predict_start_from_epsilon_and_start(
                    x, t=t, x0_and_epsilon=self.denoise_fn((condition_x, x), noise_level))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance, x_recon

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None):
        model_mean, model_log_variance, x_recon = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return {"mean": model_mean,
                "log_variance": model_log_variance,
                "sample": model_mean + noise * (0.5 * model_log_variance).exp()}

    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False):
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps//10))
        if not self.conditional:
            shape = x_in
            img = torch.randn(shape, device=device)
            ret_img = img
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img = self.p_sample(img, i)["sample"]
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        else:
            x = x_in
            shape = (x.shape[0], x.shape[1], self.image_size, self.image_size)
            img = torch.randn(shape, device=device)
            ret_img = img
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img = self.p_sample(img, i, condition_x=x)["sample"]
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        if continous:
            return ret_img
        else:
            return ret_img[-1]

    @torch.no_grad()
    def ddim_sample(self, x, t, tau, eta, clip_denoised=True, condition_x=None):
        tau_t = tau[t]
        if t > 0:
            tau_t_pre = tau[t-1]
            sigma = (
            eta
            * torch.sqrt((1 - self.alphas_cumprod[tau_t_pre]) / (1 - self.alphas_cumprod[tau_t]))
            * torch.sqrt(1 - self.alphas_cumprod[tau_t] / self.alphas_cumprod[tau_t_pre])
        )
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[tau_t+1]]).repeat(batch_size, 1).to(x.device)
        if condition_x is not None:
            if self.mean_type == 'x0':
                x_recon = self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level)
                noise = self.predict_noise_from_start(x, t=tau_t, x0=x_recon)
                # x_recon = self.denoise_fn(torch.cat([condition_x, x], dim=1), torch.FloatTensor([t]).to(x.device))
            elif self.mean_type == 'epsilon':
                noise=self.denoise_fn((condition_x, x), noise_level)
                x_recon = self.predict_start_from_noise(x, t=tau_t, noise=noise)
            elif self.mean_type == 'mu':
                x_recon = self.predict_start_from_mu(x, tau_t, self.denoise_fn((condition_x, x), noise_level))
                noise = self.predict_noise_from_start(x, t=tau_t, x0=x_recon)
            elif self.mean_type == 'v':
                x_recon = self.predict_start_from_v(x, tau_t, self.denoise_fn((condition_x, x), noise_level))
                noise = self.predict_noise_from_start(x, t=tau_t, x0=x_recon)
            elif self.mean_type == 'x0_and_epsilon':
                x_recon = self.predict_start_from_epsilon_and_start(x, tau_t, self.denoise_fn((condition_x, x), noise_level))
                noise = self.predict_noise_from_start(x, t=tau_t, x0=x_recon)
            if clip_denoised:
                x_recon.clamp_(-1., 1.)
        else:
            if self.mean_type == 'x0':
                x_recon = self.denoise_fn(x, noise_level)
                noise = self.predict_noise_from_start(x, t=tau_t, x0=x_recon)
            elif self.mean_type == 'epsilon':
                noise=self.denoise_fn(x, noise_level)
                x_recon = self.predict_start_from_noise(x, t=tau_t, noise=noise)
            elif self.mean_type == 'mu':
                x_recon = self.predict_start_from_mu(x, tau_t, self.denoise_fn(x, noise_level))
                noise = self.predict_noise_from_start(x, t=tau_t, x0=x_recon)
            elif self.mean_type == 'v':
                x_recon = self.predict_start_from_v(x, tau_t, self.denoise_fn(x, noise_level))
                noise = self.predict_noise_from_start(x, t=tau_t, x0=x_recon)
            elif self.mean_type == 'x0_and_epsilon':
                x_recon = self.predict_start_from_epsilon_and_start(x, tau_t, self.denoise_fn(x, noise_level))
                noise = self.predict_noise_from_start(x, t=tau_t, x0=x_recon)
            if clip_denoised:
                x_recon.clamp_(-1., 1.)
        if tau_t > 0:
            model_mean = torch.sqrt(self.alphas_cumprod_prev[tau_t_pre+1])*x_recon + \
                        torch.sqrt(1-self.alphas_cumprod_prev[tau_t_pre+1]-sigma**2)*noise
            model_log_SD = torch.log(sigma)
            n = torch.randn_like(x)
                # n = torch.randn_like(x) if tau_t > 0 else torch.zeros_like(x)
            return {"mean": model_mean,
                    "predict_start": x_recon,
                    "model_log_SD": model_log_SD,
                    "sample": model_mean + n * (model_log_SD).exp()}
        else:
            return {"mean": x_recon,
                    "predict_start": x_recon,
                    "log_variance": None,
                    "sample": x_recon}

    @torch.no_grad()
    def ddim_sample_loop(self, x_in, timestep_map, eta, continous=False):
        device = self.betas.device
        num_timesteps = len(timestep_map)
        sample_inter = (1 | (num_timesteps//10))
        if not self.conditional:
            shape = x_in
            img = torch.randn(shape, device=device)
            ret_img = img
            for i in tqdm(reversed(range(0, num_timesteps)), desc='sampling loop time step', total=num_timesteps):
                img = self.ddim_sample(img, i, timestep_map, eta)['sample']
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        else:
            x = x_in
            shape = (x.shape[0], x.shape[1], self.image_size, self.image_size)
            img = torch.randn(shape, device=device)
            ret_img = img
            for i in tqdm(reversed(range(0, num_timesteps)), desc='sampling loop time step', total=num_timesteps):
                img = self.ddim_sample(img, i, timestep_map, eta, condition_x=x)['sample']
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        if continous:
            return ret_img
        else:
            return ret_img[-1]
        
    @torch.no_grad()
    def ddim_sample_loop_inversion(self, x_in, timestep_map, eta, noisy, lambda1=0.0075, a=1.5, b=-0.01, c=0.3, resume=3, mode=0, continous=False):
        device = self.betas.device
        num_timesteps = len(timestep_map)
        # sample_inter = (1 | (num_timesteps//10))
        sample_inter = 1
        if not self.conditional:
            shape = x_in
            bs = shape[0]
            img = torch.randn(shape, device=device)
            ret_img = img
            for i in tqdm(reversed(range(0, num_timesteps)), desc='sampling loop time step', total=num_timesteps):
            # for i in reversed(range(0, num_timesteps)):
                continuous_sqrt_alpha_cumprod = torch.FloatTensor([self.sqrt_alphas_cumprod_prev[timestep_map[i-1]]]).repeat(bs, 1).to(device)
                continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(bs, -1)
                noisy_t = self.q_sample(noisy, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1))
                
                # lam = torch.full(shape, lambda1).to(device) * self.sqrt_alphas_cumprod[timestep_map[i]]
                lam = lambda1 * self.sqrt_alphas_cumprod[timestep_map[i]]
                
                out = self.ddim_sample(img, i, timestep_map, eta)
                if i > 0:
                    img = (noisy_t + lam / torch.exp(out["model_log_SD"]) * out["mean"]) / (1 + lam / torch.exp(out["model_log_SD"]))
                else: 
                    img = out["mean"]

                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
            for k in range(1):
                pred_noise = torch.abs(noisy - img)
                noise_std = torch.std(pred_noise, axis=(2, 3)).unsqueeze(-1).unsqueeze(-1).expand(*shape) 
                # print(torch.std(pred_noise, axis=(2, 3)))

                if mode == 0:
                    resume = 0
                if mode == 1:
                    lam = noise_std * a  + b
                elif mode == 2:
                    lam = pred_noise * c  # best 0.3 Liver; 1.2 Chest
                elif mode == 3:
                    lam = (noise_std * a  + b) / lambda1 * pred_noise * c   
                if mode != 0:
                    img = ret_img[-bs*(resume+1):-bs*resume, ...]           
                for i in tqdm(reversed(range(0, resume)), desc='sampling loop time step', total=resume):
                    continuous_sqrt_alpha_cumprod = torch.FloatTensor([self.sqrt_alphas_cumprod_prev[timestep_map[i-1]]]).repeat(bs, 1).to(device)
                    continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(bs, -1)
                    noisy_t = self.q_sample(noisy, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1))

                    out = self.ddim_sample(img, i, timestep_map, eta)
                    if i > 0:
                        img = (noisy_t + lam / torch.exp(out["model_log_SD"]) * out["mean"]) / (1 + lam / torch.exp(out["model_log_SD"]))
                    else:
                        img = out['mean']
                    if i % sample_inter == 0:
                        ret_img = torch.cat([ret_img, img], dim=0)
        else:
            x = x_in
            shape = (x.shape[0], x.shape[1], self.image_size, self.image_size)
            bs = shape[0]
            img = torch.randn(shape, device=device)
            ret_img = img
            for i in tqdm(reversed(range(0, num_timesteps)), desc='sampling loop time step', total=num_timesteps):
            # for i in reversed(range(0, num_timesteps)):
                continuous_sqrt_alpha_cumprod = torch.FloatTensor([self.sqrt_alphas_cumprod_prev[timestep_map[i-1]]]).repeat(bs, 1).to(device)
                continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(bs, -1)
                noisy_t = self.q_sample(noisy, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1))
        
                # lam = torch.full(shape, lambda1).to(device) * self.sqrt_alphas_cumprod[timestep_map[i]]
                lam = lambda1 * self.sqrt_alphas_cumprod[timestep_map[i]]

                out = self.ddim_sample(img, i, timestep_map, eta, condition_x=x)
                if i > 0:
                    img = (noisy_t + lam / torch.exp(out["model_log_SD"]) * out["mean"]) / (1 + lam / torch.exp(out["model_log_SD"]))
                else: 
                    img = out["mean"]
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
            noise_std_list = [torch.zeros_like(noisy)]
            pred_noise_list = [torch.zeros_like(noisy)]
            
            for k in range(1):
                pred_noise = torch.abs(noisy - img)
                noise_std = torch.std(pred_noise, axis=(2, 3)).unsqueeze(-1).unsqueeze(-1).expand(*shape) 
                noise_std_list.append(noise_std)
                pred_noise_list.append(pred_noise)
                # print(torch.std(pred_noise, axis=(2, 3)).flatten())
                
                if mode == 0:
                    resume = 0
                if mode == 1:
                    lam = noise_std * a  + b
                    # lam = lam + (noise_std_list[k+1] - noise_std_list[k]) * 1
                elif mode == 2:
                    lam = pred_noise * c
                    # lam = lam + (pred_noise_list[k+1] - pred_noise_list[k]) * c
                elif mode == 3:
                    lam = (noise_std * a  + b) / lambda1 * pred_noise * c 
                if mode != 0:
                    img = ret_img[-bs*(resume+1):-bs*resume, ...]   

                # print(lam[:, :, 0, 0].flatten())

                for i in tqdm(reversed(range(0, resume)), desc='resuming', total=resume):
                    continuous_sqrt_alpha_cumprod = torch.FloatTensor([self.sqrt_alphas_cumprod_prev[timestep_map[i-1]]]).repeat(bs, 1).to(device)
                    continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(bs, -1)
                    noisy_t = self.q_sample(noisy, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1))
                    
                    out = self.ddim_sample(img, i, timestep_map, eta, condition_x=x)
                    if i > 0:
                        img = (noisy_t + lam / torch.exp(out["model_log_SD"]) * out["mean"]) / (1 + lam / torch.exp(out["model_log_SD"]))
                    else:
                        img = out['mean']
                    if i % sample_inter == 0:
                        ret_img = torch.cat([ret_img, img], dim=0)

        if continous:
            return ret_img
        else:
            return ret_img[-bs:, ...]

    
    @torch.no_grad()
    def p_sample_loop_inversion(self, x_in, noisy, lambda1=0.0075, a=1.5, b=-0.01, c=0.3, resume=3, mode=0, continous=False):
        device = self.betas.device
        sample_inter = 1 # (1 | (self.num_timesteps//10))
        if not self.conditional:
            shape = x_in
            bs = shape[0]
            img = torch.randn(shape, device=device)
            ret_img = img
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='resuming', total=self.num_timesteps):

                continuous_sqrt_alpha_cumprod = torch.FloatTensor(
                np.random.uniform(
                    self.sqrt_alphas_cumprod_prev[i-1],
                    self.sqrt_alphas_cumprod_prev[i],
                    size=bs
                )).to(noisy.device)
                continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(bs, -1)
                noisy_t = self.q_sample(noisy, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1))
                # lam = torch.full(shape, lambda1).to(device) * self.sqrt_alphas_cumprod[i]
                lam = lambda1 * self.sqrt_alphas_cumprod[i]
                out = self.p_sample(img, i)
                if i > 0:
                    img = (noisy_t + lam / torch.exp(0.5 * out["log_variance"]) * out["mean"]) / (1 + lam / torch.exp(0.5 * out["log_variance"]))
                else: 
                    img = out["mean"]
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
            for k in range(1):
                pred_noise = torch.abs(noisy - img)
                noise_std = torch.std(pred_noise, axis=(2, 3)).unsqueeze(-1).unsqueeze(-1).expand(*shape) 
            
                if mode == 0:
                    resume = 0
                if mode == 1:
                    lam = noise_std * a  + b
                elif mode == 2:
                    lam = pred_noise * c
                elif mode == 3:
                    lam = (noise_std * a  + b) / lambda1 * pred_noise * c  
                if mode != 0:
                    img = ret_img[-bs*(resume+1):-bs*resume, ...]  
                for i in tqdm(reversed(range(0, resume)), desc='resuming', total=resume):
                    continuous_sqrt_alpha_cumprod = torch.FloatTensor(
                    np.random.uniform(
                        self.sqrt_alphas_cumprod_prev[i-1],
                        self.sqrt_alphas_cumprod_prev[i],
                        size=bs
                    )).to(noisy.device)
                    continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(bs, -1)
                    noisy_t = self.q_sample(noisy, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1))
        
                    out = self.p_sample(img, i)
                    if i > 0:
                        img = (noisy_t + lam / torch.exp(0.5 * out["log_variance"]) * out["mean"]) / (1 + lam / torch.exp(0.5 * out["log_variance"]))
                    else: 
                        img = out["mean"]
                    if i % sample_inter == 0:
                        ret_img = torch.cat([ret_img, img], dim=0)

        else:
            x = x_in
            shape = (x.shape[0], x.shape[1], self.image_size, self.image_size)
            bs = shape[0]
            img = torch.randn(shape, device=device)
            ret_img = img
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='resuming', total=self.num_timesteps):
                
                continuous_sqrt_alpha_cumprod = torch.FloatTensor(
                np.random.uniform(
                    self.sqrt_alphas_cumprod_prev[i-1],
                    self.sqrt_alphas_cumprod_prev[i],
                    size=bs
                )).to(noisy.device)
                continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(bs, -1)
                
                noisy_t = self.q_sample(noisy, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1))
                # lam = torch.full(shape, lambda1).to(device) * self.sqrt_alphas_cumprod[i]
                lam = lambda1 * self.sqrt_alphas_cumprod[i]
                out = self.p_sample(img, i, condition_x=x)
                if i > 0:
                    img = (noisy_t + lam / torch.exp(0.5 * out["log_variance"]) * out["mean"]) / (1 + lam / torch.exp(0.5 * out["log_variance"]))
                else: 
                    img = out["mean"]
                
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
            for k in range(1):
                pred_noise = torch.abs(noisy - img)
                noise_std = torch.std(pred_noise, axis=(2, 3)).unsqueeze(-1).unsqueeze(-1).expand(*shape) 
            
                if mode == 0:
                    resume = 0
                if mode == 1:
                    lam = noise_std * a  + b
                elif mode == 2:
                    lam = pred_noise * c
                elif mode == 3:
                    lam = (noise_std * a  + b) / lambda1 * pred_noise * c  
                if mode != 0:
                    img = ret_img[-bs*(resume+1):-bs*resume, ...]  

                for i in tqdm(reversed(range(0, resume)), desc='resuming', total=resume):
                    continuous_sqrt_alpha_cumprod = torch.FloatTensor(
                    np.random.uniform(
                        self.sqrt_alphas_cumprod_prev[i-1],
                        self.sqrt_alphas_cumprod_prev[i],
                        size=bs
                    )).to(noisy.device)
                    continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(bs, -1)
                    noisy_t = self.q_sample(noisy, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1))
                    out = self.p_sample(img, i, condition_x=x)
                    if i > 0:
                        img = (noisy_t + lam / torch.exp(0.5 * out["log_variance"]) * out["mean"]) / (1 + lam / torch.exp(0.5 * out["log_variance"]))
                    else: 
                        img = out["mean"]
                    if i % sample_inter == 0:
                        ret_img = torch.cat([ret_img, img], dim=0)
        if continous:
            return ret_img
        else:
            return ret_img[-bs:, ...]
        
    @torch.no_grad()
    def sample(self, *args, batch_size=1, continous=False, ddim=False):
        image_size = self.image_size
        channels = self.channels
        if not ddim:
            return self.p_sample_loop((batch_size, channels, image_size, image_size), continous)
        else:
            return self.ddim_sample_loop((batch_size, channels, image_size, image_size), *args, continous=continous)

    @torch.no_grad()
    def inversion(self, noisy, sr, *args, batch_size=1, ddim=False, lambda1=0.0075, a=1.5, b=-0.01, c=0.3, resume=3, mode=0, continous=False):
        image_size = self.image_size
        channels = self.channels
        if not self.conditional:
            if ddim:
                return self.ddim_sample_loop_inversion((batch_size, channels, image_size, image_size), *args, noisy, 
                                                       lambda1=lambda1, a=a, b=b, c=c, resume=resume, mode=mode, continous=continous)
            else:
                return self.p_sample_loop_inversion((batch_size, channels, image_size, image_size), noisy, 
                                                    lambda1=lambda1, a=a, b=b, c=c, resume=resume, mode=mode, continous=continous)
        else:
            if ddim:
                return self.ddim_sample_loop_inversion(sr, *args, noisy, 
                                                       lambda1=lambda1, a=a, b=b, c=c, resume=resume, mode=mode, continous=continous)
            else:
                return self.p_sample_loop_inversion(sr, noisy, 
                                                    lambda1=lambda1, a=a, b=b, c=c, resume=resume, mode=mode, continous=continous)

    @torch.no_grad()
    def super_resolution(self, x_in, *args, continous=False, ddim=False):
        if not ddim:
            return self.p_sample_loop(x_in, continous)
        else:
            return self.ddim_sample_loop(x_in, *args, continous=continous)

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # random gama
        return (
            continuous_sqrt_alpha_cumprod * x_start +
            (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise
        )

    def p_losses(self, x_in, noise=None):
        x_start = x_in['HR']
        [b, c, h, w] = x_start.shape
        t = np.random.randint(0, self.num_timesteps, size=b)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                [self.sqrt_alphas_cumprod_prev[ti] for ti in t],
                [self.sqrt_alphas_cumprod_prev[ti+1] for ti in t]
            )
        ).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
            b, -1)
        t_tensor = torch.FloatTensor(t).to(x_start.device)

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(
            x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)

        if not self.conditional:
            x_recon = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
        else:
            x_recon = self.denoise_fn(
                (x_in['SR'], x_noisy), continuous_sqrt_alpha_cumprod)
            
        self.snrs = self.alphas_cumprod / self.sqrt_one_minus_alphas_cumprod**2
        weightings = (torch.sqrt(self.snrs)).cpu().numpy()

        if self.mean_type == 'x0':  
            loss = _extract_into_tensor(weightings, t_tensor.long(), x_start.shape) * self.loss_func(x_start, x_recon)
        elif self.mean_type == 'epsilon': 
            x_recon = self._predict_xstart_from_noise(x_noisy, t_tensor.long(), x_recon)
            loss = _extract_into_tensor(weightings, t_tensor.long(), x_start.shape) * self.loss_func(x_start, x_recon)
        elif self.mean_type == 'mu': 
            x_recon = self._predict_xstart_from_mu(x_noisy, t_tensor.long(), x_recon)
            loss = _extract_into_tensor(weightings, t_tensor.long(), x_start.shape) * self.loss_func(x_start, x_recon)
        elif self.mean_type == 'v': 
            x_recon = self._predict_xstart_from_v(x_noisy, t_tensor.long(), x_recon)
            loss = _extract_into_tensor(weightings, t_tensor.long(), x_start.shape) * self.loss_func(x_start, x_recon)
        elif self.mean_type == 'x0_and_epsilon': 
            x_recon = self._predict_xstart_from_epsilon_and_start(x_noisy, t_tensor.long(), x_recon)
            loss = _extract_into_tensor(weightings, t_tensor.long(), x_start.shape) * self.loss_func(x_start, x_recon)
        return loss

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if type(arr) == np.ndarray:
        res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    elif type(arr) == torch.Tensor:
        res = arr.to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
