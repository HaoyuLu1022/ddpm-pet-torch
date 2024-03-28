import itertools
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
import scipy.ndimage as ndimage
from numpy.lib.stride_tricks import sliding_window_view
from skimage.metrics import structural_similarity, normalized_root_mse, peak_signal_noise_ratio
from skimage.exposure import rescale_intensity
from prettytable import PrettyTable
from tqdm import tqdm
import math

from nets import (GaussianDiffusion, UNet, generate_cosine_schedule,
                  generate_linear_schedule)
from utils.utils import postprocess_output, show_config, preprocess_input


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
        "guide_channels"    : 32,
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
            
        self.net    = GaussianDiffusion(UNet(1, condition=True, guide_channels=self.guide_channels, base_channels=self.channel), self.input_shape, 1, betas=betas)

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
            postprocess_output(np.expand_dims((ax_feature[i][15]).cpu().data.numpy(), axis=0).transpose(2, 1, 0)) for i in range(ax_feature.shape[0])]
        predict_images = [
            postprocess_output(test_images[i].cpu().data.numpy().transpose(2, 1, 0)) for i in range(size_figure_grid_c)
        ]
        gt_images = [
            postprocess_output(gt[i].transpose(2, 1, 0)) for i in range(size_figure_grid_c)
        ]
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
            ax[1, k].imshow(predict_images[k], cmap='gray', origin='lower')
        for k in range(size_figure_grid_c): 
            ax[2, k].cla()
            ax[2, k].imshow(gt_images[k], cmap='gray', origin='lower')
        
        table = PrettyTable(['Metrics', '1st slice', '2nd slice', '3rd slice', '4th slice'])
        psnr = []
        ssim = []
        nrmse = []
        for i in range(size_figure_grid_c): 
            psnr.append(peak_signal_noise_ratio(predict_images[i], gt_images[i], data_range=0.005)) 
            ssim.append(structural_similarity(predict_images[i], gt_images[i], data_range=0.005, channel_axis=2))
            nrmse.append(normalized_root_mse(rescale_intensity(predict_images[i]), rescale_intensity(gt_images[i]), normalization='euclidean'))
        table.add_row(['PSNR', f"{psnr[0]:.3f}", f"{psnr[1]:.3f}", f"{psnr[2]:.3f}", f"{psnr[3]:.3f}"])
        table.add_row(['SSIM', f"{ssim[0]:.3f}", f"{ssim[1]:.3f}", f"{ssim[2]:.3f}", f"{ssim[3]:.3f}"])
        table.add_row(['NRMSE', f"{nrmse[0]:.3f}", f"{nrmse[1]:.3f}", f"{nrmse[2]:.3f}", f"{nrmse[3]:.3f}"])
        print(table)

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        plt_path = f"{result_dir}/{mode}_{ddim_step}_results.png" if mode == 'ddim' else f"{result_dir}/{mode}_results.png"
        plt.savefig(plt_path)
        plt.close('all')  #避免内存泄漏

    def show_result_3d(self, device, result_dir, guide_channels, mode='ddim'):
        img_shape = (256, 256, 256)
        target_shape = (128, 128, 128)
        files = "test_lines.txt"
        scale = 1e4
        with open(files) as f:
           lines = f.readlines()
        full_dir, low_dir = lines[0].split()
        #full_dir = '/media/bld/e644a83d-65c3-4f55-a408-bea0bee7f43e/haoyu/cropped/fulldose/fulldoseP-67292.img'
        #low_dir = '/media/bld/e644a83d-65c3-4f55-a408-bea0bee7f43e/haoyu/cropped/lowdose2/P-67292.img'
        full_img = np.fromfile(full_dir, dtype=np.float32).reshape(img_shape) * scale
        # full_img = rescale_intensity(full_img, out_range=(-1, 1))
        full_img, invalid_z_list = preprocess_input(full_img)
        low_img = np.fromfile(low_dir, dtype=np.float32).reshape(img_shape) * scale
        low_img = np.delete(low_img, invalid_z_list, 0)
        low_img /= 0.004*scale #5e3
        low_img -= 0.5
        low_img /= 0.5
        # low_img = preprocess_input(low_img)
        # low_img = rescale_intensity(low_img, out_range=(-1, 1))
        low_img = ndimage.zoom(low_img, [target_shape/img_shape for target_shape, img_shape in zip(target_shape, low_img.shape)])
        if guide_channels > 1: 
            low_img = np.pad(low_img, ((math.ceil((guide_channels-1)/2), math.floor((guide_channels-1)/2)), (0, 0), (0, 0)), mode='constant', constant_values=0)
        low_img_neighbor_slices = sliding_window_view(low_img, window_shape=guide_channels, axis=0).transpose(0, 3, 1, 2)
        low_img_neighbor_slices = np.split(low_img_neighbor_slices, low_img_neighbor_slices.shape[0], axis=0)
        low_img_neighbor_slices = [torch.from_numpy(slice.copy()).cuda(device) for slice in low_img_neighbor_slices]
        # low_img_neighbor_slices = np.concatenate([np.repeat(np.expand_dims(low_img_neighbor_slices[0, :, :, :], axis=0), 16, axis=0), low_img_neighbor_slices, np.repeat(np.expand_dims(low_img_neighbor_slices[-1, :, :, :], axis=0), 15, axis=0)], axis=0)
        # low_img_neighbor_slices = [torch.Tensor(low_img_neighbor_slices[i, :, :, :]).unsqueeze(0).cuda() for i in range(len(low_img_neighbor_slices))]
        fulldose = []
        if mode == 'ddpm':
            print("Generating images...")
            for i in tqdm(range(0, len(low_img_neighbor_slices)), total=len(low_img_neighbor_slices)):
                fulldose.append(self.net.sample(1, device, ax_feature=low_img_neighbor_slices[i], use_ema=False).squeeze(0, 1))
        elif mode == 'ddim': 
            ddim_step = int(input('Input your sampling step for DDIM (default to 25): '))
            print("Generating images...")
            for i in tqdm(range(0, len(low_img_neighbor_slices)), total=len(low_img_neighbor_slices)):
                fulldose.append(self.net.ddim_sample(1, device, ax_feature=low_img_neighbor_slices[i], ddim_step=ddim_step, eta=0, use_ema=False, simple_var=False).squeeze(0, 1))
        fulldose = torch.stack(fulldose, dim=0)
        fulldose = postprocess_output(fulldose.cpu().data.numpy())
        fulldose = ndimage.zoom(fulldose, [shape/target_shape for target_shape, shape in zip(target_shape, full_img.shape)])

        full_img = postprocess_output(full_img)
        ssim = structural_similarity(full_img, fulldose, data_range=0.008*scale) #1e4
        psnr = peak_signal_noise_ratio(full_img, fulldose, data_range=0.008*scale) #1e4
        nrmse = normalized_root_mse(full_img, fulldose, normalization='euclidean')

        print(f"SSIM: {ssim}\nPSNR: {psnr}\nNRMSE: {nrmse}")

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        img_path = f"{result_dir}/{mode}_{ddim_step}_results.img" if mode == 'ddim' else f"{result_dir}/{mode}_results.img"
        
        fulldose.tofile(img_path)

    def show_result_3d_loop(self, device, result_dir, guide_channels, mode='ddim', ddim_step=25):
        img_shape = (256, 256, 256)
        target_shape = (128, 128, 128)
        files = "test_lines.txt"
        scale = 1e4
        with open(files) as f:
            lines = f.readlines()
        for line in lines:
            full_dir, low_dir = line.split()
            idx = full_dir[74:-8]
            #full_dir = '/media/bld/e644a83d-65c3-4f55-a408-bea0bee7f43e/haoyu/cropped/fulldose/fulldoseP-67292.img'
            #low_dir = '/media/bld/e644a83d-65c3-4f55-a408-bea0bee7f43e/haoyu/cropped/lowdose2/P-67292.img'
            full_img = np.fromfile(full_dir, dtype=np.float32).reshape(img_shape) * scale
            # full_img = rescale_intensity(full_img, out_range=(-1, 1))
            full_img, invalid_z_list = preprocess_input(full_img)
            low_img = np.fromfile(low_dir, dtype=np.float32).reshape(img_shape) * scale
            low_img = np.delete(low_img, invalid_z_list, 0)
            low_img /= 0.004*scale #5e3
            low_img -= 0.5
            low_img /= 0.5
            # low_img = preprocess_input(low_img)
            # low_img = rescale_intensity(low_img, out_range=(-1, 1))
            low_img = ndimage.zoom(low_img, [target_shape/img_shape for target_shape, img_shape in zip(target_shape, low_img.shape)])
            if guide_channels > 1: 
                low_img = np.pad(low_img, ((math.ceil((guide_channels-1)/2), math.floor((guide_channels-1)/2)), (0, 0), (0, 0)), mode='constant', constant_values=0)
            low_img_neighbor_slices = sliding_window_view(low_img, window_shape=guide_channels, axis=0).transpose(0, 3, 1, 2)
            low_img_neighbor_slices = np.split(low_img_neighbor_slices, low_img_neighbor_slices.shape[0], axis=0)
            low_img_neighbor_slices = [torch.from_numpy(slice.copy()).cuda(device) for slice in low_img_neighbor_slices]
            # low_img_neighbor_slices = np.concatenate([np.repeat(np.expand_dims(low_img_neighbor_slices[0, :, :, :], axis=0), 16, axis=0), low_img_neighbor_slices, np.repeat(np.expand_dims(low_img_neighbor_slices[-1, :, :, :], axis=0), 15, axis=0)], axis=0)
            # low_img_neighbor_slices = [torch.Tensor(low_img_neighbor_slices[i, :, :, :]).unsqueeze(0).cuda() for i in range(len(low_img_neighbor_slices))]
            fulldose = []
            if mode == 'ddpm':
                print("Generating images...")
                for i in tqdm(range(0, len(low_img_neighbor_slices)), total=len(low_img_neighbor_slices)):
                    fulldose.append(self.net.sample(1, device, ax_feature=low_img_neighbor_slices[i], use_ema=False).squeeze(0, 1))
            elif mode == 'ddim': 
                # ddim_step = int(input('Input your sampling step for DDIM (default to 25): '))
                print("Generating images...")
                for i in tqdm(range(0, len(low_img_neighbor_slices)), total=len(low_img_neighbor_slices)):
                    fulldose.append(self.net.ddim_sample(1, device, ax_feature=low_img_neighbor_slices[i], ddim_step=ddim_step, eta=0, use_ema=False, simple_var=False).squeeze(0, 1))
            fulldose = torch.stack(fulldose, dim=0)
            fulldose = postprocess_output(fulldose.cpu().data.numpy())
            fulldose = ndimage.zoom(fulldose, [shape/target_shape for target_shape, shape in zip(target_shape, full_img.shape)])

            full_img = postprocess_output(full_img)
            ssim = structural_similarity(full_img, fulldose, data_range=0.008*scale) #1e4
            psnr = peak_signal_noise_ratio(full_img, fulldose, data_range=0.008*scale) #1e4
            nrmse = normalized_root_mse(full_img, fulldose, normalization='euclidean')

            print(f"for {idx}:\nSSIM: {ssim}\nPSNR: {psnr}\nNRMSE: {nrmse}")

            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            img_path = f"{result_dir}/{mode}_{ddim_step}_{idx}_results.img" if mode == 'ddim' else f"{result_dir}/{mode}_{idx}_results.img"
            
            fulldose.tofile(img_path)