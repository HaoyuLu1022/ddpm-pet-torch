import itertools
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn

from nets import (GaussianDiffusion, UNet, generate_cosine_schedule,
                  generate_linear_schedule)
from utils.utils import postprocess_output, show_config


class Diffusion(object):
    _defaults = {
        #-----------------------------------------------#
        #   model_path指向logs文件夹下的权值文件
        #-----------------------------------------------#
        "model_path"        : 'model_data/Diffusion_Flower.pth',
        #-----------------------------------------------#
        #   卷积通道数的设置
        #-----------------------------------------------#
        "channel"           : 128,
        #-----------------------------------------------#
        #   输入图像大小的设置
        #-----------------------------------------------#
        "input_shape"       : (128, 128),
        #-----------------------------------------------#
        #   betas相关参数
        #-----------------------------------------------#
        "schedule"          : "linear",
        "num_timesteps"     : 1000,
        "schedule_low"      : 1e-4,
        "schedule_high"     : 0.02,
        #-------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        #-------------------------------#
        "cuda"              : True,
    }

    #---------------------------------------------------#
    #   初始化Diffusion
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)  
            self._defaults[name] = value 
        self.generate()

        show_config(**self._defaults)

    def generate(self):
        #----------------------------------------#
        #   创建Diffusion模型
        #----------------------------------------#
        if self.schedule == "cosine":
            betas = generate_cosine_schedule(self.num_timesteps)
        else:
            betas = generate_linear_schedule(
                self.num_timesteps,
                self.schedule_low * 1000 / self.num_timesteps,
                self.schedule_high * 1000 / self.num_timesteps,
            )
            
        self.net    = GaussianDiffusion(UNet(1, condition=True, guide_channels=32, base_channels=self.channel), self.input_shape, 1, betas=betas)

        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model loaded.'.format(self.model_path))

        if self.cuda:
            self.net = self.net.cuda()

    #---------------------------------------------------#
    #   Diffusion5x5的图片
    #---------------------------------------------------#
    def generate_5x5_image(self, save_path):
        with torch.no_grad():
            randn_in    = torch.randn((1, 1)).cuda() if self.cuda else torch.randn((1, 1))

            test_images = self.net.sample(25, randn_in.device)

            size_figure_grid = 5
            fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
            for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
                ax[i, j].get_xaxis().set_visible(False)
                ax[i, j].get_yaxis().set_visible(False)

            for k in range(5*5):
                i = k // 5
                j = k % 5
                ax[i, j].cla()
                ax[i, j].imshow(np.uint8(postprocess_output(test_images[k].cpu().data.numpy().transpose(1, 2, 0))))

            label = 'predict_5x5_results'
            fig.text(0.5, 0.04, label, ha='center')
            plt.savefig(save_path)

    #---------------------------------------------------#
    #   Diffusion1x1的图片
    #---------------------------------------------------#
    def generate_1x1_image(self, save_path, condition=None):
        with torch.no_grad():
            randn_in    = torch.randn((1, 1)).cuda() if self.cuda else torch.randn((1, 1))
            if condition is not None:
                condition = condition.cuda()

            test_images = self.net.sample(1, randn_in.device, use_ema=False, ax_feature=condition)
            test_images = postprocess_output(test_images[0].cpu().data.numpy().transpose(1, 2, 0))

            # Image.fromarray(test_images[:, :, 0]).save(save_path)
            plt.imshow(test_images[:, :, 0], cmap='gray', origin='lower')
            plt.axis('off')
            plt.savefig(save_path)

    def show_result(self, device, result_dir, gt, ax_feature=None, mode='ddpm'):
        print("Generating images...")

        if mode == 'ddpm':
            test_images = self.net.sample(len(ax_feature), device, ax_feature=ax_feature, use_ema=False)
        elif mode == 'ddim':
            ddim_step = int(input('Input your sampling step for DDIM: '))
            test_images = self.net.ddim_sample(len(ax_feature), device, ax_feature=ax_feature, ddim_step=ddim_step, eta=0, use_ema=False, simple_var=False)
            # seems that simple_var must be False to provide good results

        size_figure_grid_r = 3
        size_figure_grid_c = 4
        fig, ax = plt.subplots(size_figure_grid_r, size_figure_grid_c, figsize=(10, 10), constrained_layout=True)
        low_imgs = [
            postprocess_output(np.expand_dims((ax_feature[0][15]).cpu().data.numpy(), axis=0).transpose(2, 1, 0)), 
            postprocess_output(np.expand_dims(ax_feature[1][15].cpu().data.numpy(),axis=0).transpose(2, 1, 0)),
            postprocess_output(np.expand_dims(ax_feature[2][15].cpu().data.numpy(), axis=0).transpose(2, 1, 0)),
            postprocess_output(np.expand_dims(ax_feature[3][15].cpu().data.numpy(), axis=0).transpose(2, 1, 0)),]
        for i, j in itertools.product(range(size_figure_grid_r), range(size_figure_grid_c)):
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)
        for k in range(size_figure_grid_c):
            ax[0, k].cla()
            ax[0, k].imshow(low_imgs[k], cmap='gray', origin='lower')
        for k in range(size_figure_grid_c):
            # i = k // size_figure_grid_c
            # j = k % size_figure_grid_c
            ax[1, k].cla()
            ax[1, k].imshow(postprocess_output(test_images[k].cpu().data.numpy().transpose(2, 1, 0)), cmap='gray', origin='lower')
        for k in range(size_figure_grid_c): 
            ax[2, k].cla()
            ax[2, k].imshow(postprocess_output(gt[k].transpose(2, 1, 0)), cmap='gray', origin='lower')

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        plt_path = f"{result_dir}/{mode}_{ddim_step}_results.png" if mode == 'ddim' else f"{result_dir}/{mode}_results.png"
        plt.savefig(plt_path)
        plt.close('all')  #避免内存泄漏