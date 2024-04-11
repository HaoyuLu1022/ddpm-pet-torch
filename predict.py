#-------------------------------------#
#   运行predict.py可以生成图片
#   生成1x1的图片和5x5的图片
#-------------------------------------#
import numpy as np
import time
import torch
import os
import scipy.ndimage as ndimage
from numpy.lib.stride_tricks import sliding_window_view
import re

from ddpm import Diffusion
from utils.utils import preprocess_input

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
ax_channel_num = 32

if __name__ == "__main__":
    model_path = input("Please specify the model weight directory: ")
    pattern1 = r'loss_(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})'
    pattern2 = r'Diffusion_Epoch(\d+)-GLoss([\d.]+)\.pth'
    match1 = re.search(pattern1, model_path)
    match2 = re.search(pattern2, model_path)
    folder1 = match1.group(0)
    folder2 = match2.group(0)
    save_path = f"results/predict_out/{folder1}/{folder2[:-4]}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # save_path_5x5 = f"{save_path}/predict_5x5_results.png"
    # save_path_1x1 = f"{save_path}/predict_1x1_results.png"

    ddpm = Diffusion(model_path=model_path, guide_channels=ax_channel_num)
    batch_size = 4
    torch.manual_seed(114514)
    init = torch.randn(batch_size, 1, *(128, 128), device=torch.device("cuda"))
    while True:
        mode = input('Choose generation mode (ddpm, ddim, dpm, or press Q to quit): ') # mixed_sample currently unavailable
        if mode == 'q' or mode == 'Q':
            print("Generation exited. ")
            break
        # ddpm.generate_1x1_image(save_path_1x1, condition=low_slice)
        start = time.perf_counter()
        # ddpm.show_result(test_low.device, save_path, test_full_list, test_low, mode)
        ddpm.show_result_3d(batch_size, torch.device('cuda'), save_path, ax_channel_num, mode, init) 
        end = time.perf_counter()
        print(f"Generation done, consuming {(end-start)}s.")
        
        # print("Generate_5x5_image")
        # ddpm.generate_5x5_image(save_path_5x5)
        # print("Generate_5x5_image Done")