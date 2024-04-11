import itertools
import math
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from typing import Tuple, List
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, normalized_root_mse
from skimage.exposure import rescale_intensity
from prettytable import PrettyTable

maxn = 0.004
scale = 1e4

#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

#----------------------------------------#
#   预处理训练图片
#----------------------------------------#
# def preprocess_input(x):
#     x /= 255
#     x -= 0.5
#     x /= 0.5
#     return x
def preprocess_input(x: np.ndarray) -> Tuple[np.ndarray, List]: 
    invalid_z_list = [k for k in range(x.shape[0]) if np.max(x[k, :, :]) > maxn*scale]
    x_rev = np.delete(x, invalid_z_list, 0)

    x_rev /= maxn*scale
    x_rev -= 0.5
    x_rev /= 0.5

    return x_rev, invalid_z_list


def postprocess_output(x):
    x *= 0.5
    x += 0.5
    x *= maxn*scale
    return x

def show_result(num_epoch, net, device, result_dir, gt, ax_feature=None):
    test_images = [net.ddim_sample(1, device, ax_feature=ax_feature[i], use_ema=False, ddim_step=25, eta=0, simple_var=False).squeeze(0) for i in range(len(ax_feature))]

    size_figure_grid_r = 3
    size_figure_grid_c = 4
    fig, ax = plt.subplots(size_figure_grid_r, size_figure_grid_c, figsize=(10, 10), constrained_layout=True)
    low_imgs = [
        postprocess_output((ax_feature[i][0][ax_feature.shape[1]//2]).cpu().data.numpy()) for i in range(ax_feature.shape[0])]
    predict_images = [
        postprocess_output(test_images[i][0].cpu().data.numpy()) for i in range(size_figure_grid_c)
    ]
    gt_images = [
        postprocess_output(gt[i][0].copy()) for i in range(size_figure_grid_c)
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
        psnr.append(peak_signal_noise_ratio(predict_images[i], gt_images[i], data_range=2*maxn*scale)) 
        ssim.append(structural_similarity(predict_images[i], gt_images[i], data_range=2*maxn*scale))
        nrmse.append(normalized_root_mse(rescale_intensity(predict_images[i]), rescale_intensity(gt_images[i]), normalization='euclidean'))
    table.add_row(['PSNR', f"{psnr[0]:.3f}", f"{psnr[1]:.3f}", f"{psnr[2]:.3f}", f"{psnr[3]:.3f}"])
    table.add_row(['SSIM', f"{ssim[0]:.3f}", f"{ssim[1]:.3f}", f"{ssim[2]:.3f}", f"{ssim[3]:.3f}"])
    table.add_row(['NRMSE', f"{nrmse[0]:.3f}", f"{nrmse[1]:.3f}", f"{nrmse[2]:.3f}", f"{nrmse[3]:.3f}"])
    print(table)

    # label = 'Epoch {0}'.format(num_epoch)
    # fig.text(0.5, -0.5, label, ha='center')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    plt.savefig(f"{result_dir}/epoch_{str(num_epoch)}_results.png")
    plt.close('all')  #避免内存泄漏

def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)

#---------------------------------------------------#
#   获得学习率
#---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10, power=8.0):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr
    
    def poly_lr(base_lr, num_epochs, iters):
        lr = base_lr * (1-iters/num_epochs)**power
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr
        return lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    elif lr_decay_type == "poly":
        func = partial(poly_lr, lr, total_iters)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
