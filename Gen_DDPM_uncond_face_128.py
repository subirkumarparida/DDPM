#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np

#os.environ['CUDA_VISIBLE_DEVICES'] = '0' #before import torch
#CUDA_LAUNCH_BLOCKING = '1'

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter

from PIL import Image
from matplotlib import pyplot as plt
import logging
from tqdm import tqdm
    
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    '''
    1 - Setting up a noising schedule
    2 - Function for noising images
    3 - Sampling images
    '''
    
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=128, device="cuda:0"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        
    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - self.alpha_hat[t])[:, None, None, None]
        eps = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps, eps
    
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
    
    def sample(self, model, n):
        #logging.info(f"Sampling {n} new images .... \n")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i>1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha)/(torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
                            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                            nn.GroupNorm(1, mid_channels),
                            nn.GELU(),
                            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                            nn.GroupNorm(1, out_channels),
                            )
        
    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
                            nn.MaxPool2d(2),
                            DoubleConv(in_channels, in_channels, residual=True),
                            DoubleConv(in_channels, out_channels),
                            )
        
        self.emb_layer = nn.Sequential(
                            nn.SiLU(),
                            nn.Linear(emb_dim, out_channels),
                            )
        
    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
                    DoubleConv(in_channels, in_channels, residual=True),
                    DoubleConv(in_channels, out_channels, in_channels//2),
                    )
        self.emb_layer = nn.Sequential(
                            nn.SiLU(),
                            nn.Linear(emb_dim, out_channels),
                            )
        
    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class SelfAttention(nn.Module):
    def __init__(self, channels, height):
        super().__init__()
        self.channels = channels
        self.height = height
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
                        nn.LayerNorm([channels]),
                        nn.Linear(channels, channels),
                        nn.GELU(),
                        nn.Linear(channels, channels),
                        )
        
    def forward(self, x):
        x = x.view(-1, self.channels, self.height * self.height).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.height, self.height)


class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, device="cuda:0"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(in_channels=c_in, out_channels=64)
        self.down1 = Down(in_channels=64, out_channels=128)
        self.sa1 = SelfAttention(channels=128, height=64)
        self.down2 = Down(in_channels=128, out_channels=256)
        self.sa2 = SelfAttention(channels=256, height=32)
        self.down3 = Down(in_channels=256, out_channels=512)
        self.sa3 = SelfAttention(channels=512, height=16)
        self.down4 = Down(in_channels=512, out_channels=512)
        self.sa4 = SelfAttention(channels=512, height=8)
        
        self.bot1 = DoubleConv(in_channels=512, out_channels=1024)
        self.bot2 = DoubleConv(in_channels=1024, out_channels=1024)
        self.bot3 = DoubleConv(in_channels=1024, out_channels=512)
        
        self.up1 = Up(in_channels=1024, out_channels=256)
        self.sa5 = SelfAttention(channels=256, height=16)
        self.up2 = Up(in_channels=512, out_channels=128)
        self.sa6 = SelfAttention(channels=128, height=32)
        self.up3 = Up(in_channels=256, out_channels=64)
        self.sa7 = SelfAttention(channels=64, height=64)
        self.up4 = Up(in_channels=128, out_channels=64)
        self.sa8 = SelfAttention(channels=64, height=128)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)
        
    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels))
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a ,pos_enc_b], dim=-1)
        return pos_enc
    
    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.inc(x)        #Input: h,w,3 | Output: h, w, 64
        x2 = self.down1(x1, t)  #Input: h,w,64 + t | Output: h/2, w/2, 128
        x2 = self.sa1(x2)       #Input: h/2, w/2, 128 | Output: h/2, w/2, 128
        x3 = self.down2(x2, t)  #Input: h/2, w/2, 128 + t | Output: h/4, w/4, 256
        x3 = self.sa2(x3)       #Input: h/4, w/4, 256 | Output: h/4, w/4, 256
        x4 = self.down3(x3, t)  #Input: h/4, w/4, 256 + t | Output: h/8, w/8, 512
        x4 = self.sa3(x4)       #Input: h/8, w/8, 512 | Output: h/8, w/8, 512
        x5 = self.down4(x4, t)  #Input: h/8, w/8, 512 + t | Output: h/16, w/16, 512
        x5 = self.sa4(x5)       #Input: h/16, w/16, 512 | Output: h/16, w/16, 512

        x5 = self.bot1(x5)      #Input: h/16, w/16, 512  | Output: h/16, w/16, 1024
        x5 = self.bot2(x5)      #Input: h/16, w/16, 1024 | Output: h/16, w/16, 1024
        x5 = self.bot3(x5)      #Input: h/16, w/16, 1024 | Output: h/16, w/16, 512

        x = self.up1(x5, x4, t) #Input: h/16, w/16, 512 + h/8, w/8, 512 + t | Output: h/8, w/8, 256
        x = self.sa5(x)         #Input: h/8, w/8, 256 | Output: h/8, w/8, 256
        x = self.up2(x, x3, t) #Input: h/8, w/8, 256 + h/4, w/4, 256 + t | Output: h/4, w/4, 128
        x = self.sa6(x)         #Input: h/4, w/4, 128 | Output: h/4, w/4, 128
        x = self.up3(x, x2, t)  #Input: h/4, w/4, 128 + h/2, w/2, 128 + t | Output: h/2, w/2, 64
        x = self.sa7(x)         #Input: h/2, w/2, 64 | Output: h/2, w/2, 64
        x = self.up4(x, x1, t)  #Input: h/2, w/2, 64 + h, w, 64 + t | Output: h, w, 64
        x = self.sa8(x)         #Input: h, w, 64 | Output: h, w, 64
        output = self.outc(x)   #Input: h, w, 64 | Output: h, w, 3
        return output


class EMA:
    def __init__(self, beta):
        self.beta = beta
        self.step = 0

    def update_model_average(self, ema_model, model):
        for current_param, ema_param in zip(model.parameters(), ema_model.parameters()):
            old_weight, new_weight = ema_param.data, current_param.data
            ema_param.data = self.update_average(old_weight, new_weight)

    def update_average(self, old, new):
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=3000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return

        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())
        

def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1,2,0).to("cpu").numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def sample_and_test(args):
    device = args.device
    model = UNet().to(device)
    ckpt = torch.load('./models/DDPM_Unconditional_Face_128-CelebA-HQ/ckpt.pt', map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    #ema_model = UNet().to(device)
    #ema = EMA(beta=0.995)
    #ckpt_ema = torch.load('./models/DDPM_Unconditional_Face_128-CelebA-HQ/ckpt_ema.pt', map_location=device)
    #ema_model.load_state_dict(ckpt_ema)
    #ema_model.eval()
    
    diffusion = Diffusion(img_size=args.image_size, device=device)

    iters_needed = 5000 // args.batch_size
    save_dir = "./generated_samples/{}".format(args.run_name)
    
    if not os.path.exists(save_dir):
    	os.makedirs(save_dir)
    
    for i in range(iters_needed):
        logging.info(f"generating batch {i} \n")
    	#print('generating batch ', i)
        
        sampled_images = diffusion.sample(model, n=args.batch_size)
        save_images(sampled_images, os.path.join(save_dir, f"{i}.jpg"))
    	
    	#ema_sampled_images = diffusion.sample(ema_model, args.batch_size)
    	#save_images(ema_sampled_images, os.path.join(save_dir, f"{i}_ema.jpg"))
        
        
def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_CelebA_ep-256"
    args.batch_size = 1
    args.image_size = 128
    args.device = "cuda:0"
    sample_and_test(args)

if __name__ == "__main__":
    launch()
