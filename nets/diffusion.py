import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import math
from fastprogress import progress_bar

from functools import partial
from copy import deepcopy


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class EMA():
    def __init__(self, decay):
        self.decay = decay
    
    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.decay + (1 - self.decay) * new

    def update_model_average(self, ema_model, current_model):
        for current_params, ema_params in zip(current_model.parameters(), ema_model.parameters()):
            old, new = ema_params.data, current_params.data
            ema_params.data = self.update_average(old, new)

class GaussianDiffusion(nn.Module):
    def __init__(
        self, model, img_size, img_channels, num_classes=None, betas=[], loss_type="l2", ema_decay=0.9999, ema_start=2000, ema_update_rate=1,
    ):
        super().__init__()
        self.model      = model
        self.ema_model  = deepcopy(model)

        self.ema                = EMA(ema_decay)
        self.ema_decay          = ema_decay
        self.ema_start          = ema_start
        self.ema_update_rate    = ema_update_rate
        self.step               = 0

        self.img_size       = img_size
        self.img_channels   = img_channels
        self.num_classes    = num_classes

        # l1或者l2损失
        if loss_type not in ["l1", "l2", "rlhf"]:
            raise ValueError("__init__() got unknown loss type")

        self.loss_type      = loss_type
        self.num_timesteps  = len(betas)

        alphas              = 1.0 - betas
        alphas_cumprod      = np.cumprod(alphas)

        # 转换成torch.tensor来处理
        to_torch = partial(torch.tensor, dtype=torch.float32)

        # betas             [0.0001, 0.00011992, 0.00013984 ... , 0.02]
        self.register_buffer("betas", to_torch(betas))
        # alphas            [0.9999, 0.99988008, 0.99986016 ... , 0.98]
        self.register_buffer("alphas", to_torch(alphas))
        # alphas_cumprod    [9.99900000e-01, 9.99780092e-01, 9.99640283e-01 ... , 4.03582977e-05]
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))

        # sqrt(alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        # sqrt(1 - alphas_cumprod)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1 - alphas_cumprod)))
        # sqrt(1 / alphas)
        self.register_buffer("reciprocal_sqrt_alphas", to_torch(np.sqrt(1 / alphas)))

        self.register_buffer("remove_noise_coeff", to_torch(betas / np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("sigma", to_torch(np.sqrt(betas)))

    def update_ema(self):
        self.step += 1
        if self.step % self.ema_update_rate == 0:
            if self.step < self.ema_start:
                self.ema_model.load_state_dict(self.model.state_dict())
            else:
                self.ema.update_model_average(self.ema_model, self.model)

    @torch.no_grad()
    def remove_noise(self, x, t, y, use_ema=True, ax_feature=None):
        if use_ema:
            return (
                (x - extract(self.remove_noise_coeff, t, x.shape) * self.ema_model(x, t, y, ax_feature)) *
                extract(self.reciprocal_sqrt_alphas, t, x.shape)
            )
        else:
            return (
                (x - extract(self.remove_noise_coeff, t, x.shape) * self.model(x, t, y, ax_feature)) *
                extract(self.reciprocal_sqrt_alphas, t, x.shape)
            )

    @torch.no_grad()
    def sample(self, batch_size, device, y=None, use_ema=True, ax_feature=None):
        if y is not None and batch_size != len(y):
            raise ValueError("sample batch size different from length of given y")

        x = torch.randn(batch_size, self.img_channels, *self.img_size, device=device)
        
        for t in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            x = self.remove_noise(x, t_batch, y, use_ema, ax_feature)

            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)
        
        return x.cpu().detach()

    @torch.no_grad()
    def sample_diffusion_sequence(self, batch_size, device, y=None, use_ema=True):
        if y is not None and batch_size != len(y):
            raise ValueError("sample batch size different from length of given y")

        x = torch.randn(batch_size, self.img_channels, *self.img_size, device=device)
        diffusion_sequence = [x.cpu().detach()]
        
        for t in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            x = self.remove_noise(x, t_batch, y, use_ema)

            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)
            
            diffusion_sequence.append(x.cpu().detach())
        
        return diffusion_sequence

    @torch.no_grad()
    def ddim_sample(self, batch_size, device, y=None, use_ema=True, ax_feature=None, ddim_step=20, eta=0, simple_var=True):
        if y is not None and batch_size != len(y):
            raise ValueError("sample batch size different from length of given y")
        
        x = torch.randn(batch_size, self.img_channels, *self.img_size, device=device)
        ts = torch.linspace(self.num_timesteps, 0, (ddim_step+1)).to(device).to(torch.long)
        
        for t in range(1, ddim_step+1):
            cur_t = ts[t-1] - 1
            prev_t = ts[t] - 1
            t_batch = torch.tensor([cur_t], device=device).repeat(batch_size)

            model = self.ema_model if use_ema else self.model
            eps = model(x, t_batch, y, ax_feature)

            alpha_bar = self.alphas_cumprod[cur_t]
            alpha_bar_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else 1

            noise = torch.rand_like(x)
            var = eta * (1 - alpha_bar_prev) / (1 - alpha_bar) * (1 - alpha_bar / alpha_bar_prev)
            first = torch.sqrt(alpha_bar_prev / alpha_bar) * x
            second = (torch.sqrt(1 - alpha_bar_prev - var) - torch.sqrt(alpha_bar_prev * (1 - alpha_bar) / alpha_bar)) * eps
            third = torch.sqrt(1 - alpha_bar / alpha_bar_prev) * noise if simple_var else torch.sqrt(var) * noise

            x = first + second + third

        return x.cpu().detach()

    def calculate_log_probs(self, prev_sample, prev_sample_mean, std_dev_t):
        std_dev_t = torch.clip(std_dev_t, 1e-6)
        log_probs = -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * std_dev_t ** 2) - torch.log(std_dev_t) - math.log(math.sqrt(2 * math.pi))
        return log_probs

    @torch.no_grad()
    def ddim_sample_rlhf(self, batch_size, device, y=None, use_ema=False, ax_feature=None, ddim_step=25, eta=0, simple_var=False):
        if y is not None and batch_size != len(y):
            raise ValueError("sample batch size different from length of given y")
        
        x = torch.randn(batch_size, self.img_channels, *self.img_size, device=device)
        ts = torch.linspace(self.num_timesteps, 0, (ddim_step+1)).to(device).to(torch.long)

        all_x = [x]
        log_probs = []
        
        for t in range(1, ddim_step+1):
            cur_t = ts[t-1] - 1
            prev_t = ts[t] - 1
            t_batch = torch.tensor([cur_t], device=device).repeat(batch_size)

            model = self.ema_model if use_ema else self.model
            eps = model(x, t_batch, y, ax_feature)

            alpha_bar = self.alphas_cumprod[cur_t]
            alpha_bar_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else 1

            noise = torch.rand_like(x)
            var = (1 - alpha_bar_prev) / (1 - alpha_bar) * (1 - alpha_bar / alpha_bar_prev)
            first = torch.sqrt(alpha_bar_prev / alpha_bar) * x
            second = (torch.sqrt(1 - alpha_bar_prev - eta * var) - torch.sqrt(alpha_bar_prev * (1 - alpha_bar) / alpha_bar)) * eps
            third = torch.sqrt(1 - alpha_bar / alpha_bar_prev) * noise if simple_var else torch.sqrt(eta * var) * noise

            x_mean = first + second 
            x = x_mean + third
            log_probs.append(self.calculate_log_probs(x, x_mean, eta*torch.sqrt(var)).mean(dim=tuple(range(1, x_mean.ndim))))
            all_x.append(x)

        return x, torch.stack(all_x), torch.stack(log_probs)
    
    def rlhf_loss(self, batch_size, x_t, original_log_probs, advantages, clip_advantages, clip_ratio, ax_feature, num_inference_steps, eta, device, y=None, use_ema=True, simple_var=False):
        loss_value = 0.
        ts = torch.linspace(self.num_timesteps, 0, (num_inference_steps+1)).to(device).to(torch.long)
        for i, t in enumerate(range(num_inference_steps)):
            clipped_advantages = torch.clip(advantages, -clip_advantages, clip_advantages).detach()

            cur_t = ts[t-1] - 1
            prev_t = ts[t] - 1
            t_batch = torch.tensor([cur_t], device=device).repeat(batch_size)

            model = self.ema_model if use_ema else self.model
            eps = model(x_t[i], t_batch, y, ax_feature)

            alpha_bar = self.alphas_cumprod[cur_t]
            alpha_bar_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else 1

            # noise = torch.rand_like(x_t[i])
            var = (1 - alpha_bar_prev) / (1 - alpha_bar) * (1 - alpha_bar / alpha_bar_prev)
            first = torch.sqrt(alpha_bar_prev / alpha_bar) * x_t[i]
            second = (torch.sqrt(1 - alpha_bar_prev - eta * var) - torch.sqrt(alpha_bar_prev * (1 - alpha_bar) / alpha_bar)) * eps
            # third = torch.sqrt(1 - alpha_bar / alpha_bar_prev) * noise if simple_var else torch.sqrt(eta * var) * noise

            x_mean = first + second 
            # x = x_mean + third
            current_log_probs = self.calculate_log_probs(x_t[i+1].detach(), x_mean, eta * torch.sqrt(var)).mean(dim=tuple(range(1, x_mean.ndim)))

            ratio = torch.exp(current_log_probs - original_log_probs[i].detach())
            unclipped_loss = -clip_advantages * ratio
            clipped_loss = -clipped_advantages * torch.clip(ratio, 1. - clip_ratio, 1. + clip_ratio)
            loss = torch.max(unclipped_loss, clipped_loss).mean()
            loss.backward()

            loss_value += loss.item()
        return loss_value

    def perturb_x(self, x, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t,  x.shape) * x +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise
        )   

    def get_losses(self, x, t, y, ax_feature):
        # x, noise [batch_size, 3, 64, 64]
        noise           = torch.randn_like(x)

        perturbed_x     = self.perturb_x(x, t, noise)
        estimated_noise = self.model(perturbed_x, t, y, ax_feature)

        if self.loss_type == "l1":
            loss = F.l1_loss(estimated_noise, noise)
        elif self.loss_type == "l2":
            loss = F.mse_loss(estimated_noise, noise)
        return loss

    def forward(self, x, y=None, ax_feature=None, all_step_preds=None, log_probs=None, advantages=None, clip_advantages=None, clip_ratio=None, all_prompts=None, num_infer_step=None, eta=None):
        b, c, h, w  = x.shape
        device      = x.device
        if self.loss_type == 'rlhf':
            return self.rlhf_loss(b, all_step_preds, log_probs, advantages, clip_advantages, clip_ratio, all_prompts, num_infer_step, eta, device)
        else: 
            if h != self.img_size[0]:
                raise ValueError("image height does not match diffusion parameters")
            if w != self.img_size[0]:
                raise ValueError("image width does not match diffusion parameters")
            
            t = torch.randint(0, self.num_timesteps, (b,), device=device)
            return self.get_losses(x, t, y, ax_feature)

def generate_cosine_schedule(T, s=0.008):
    def f(t, T):
        return (np.cos((t / T + s) / (1 + s) * np.pi / 2)) ** 2
    
    alphas = []
    f0 = f(0, T)

    for t in range(T + 1):
        alphas.append(f(t, T) / f0)
    
    betas = []

    for t in range(1, T + 1):
        betas.append(min(1 - alphas[t] / alphas[t - 1], 0.999))
    
    return np.array(betas)

def generate_linear_schedule(T, low, high):
    return np.linspace(low, high, T)