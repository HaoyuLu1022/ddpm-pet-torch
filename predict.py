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

img_shape = (256, 200, 200)
target_shape = (128, 128, 128)
test_path = 'test_lines.txt'
with open(test_path) as f:
    lines = f.readlines()
test_full_dir, test_low_dir = lines[0].split()
test_full = np.fromfile(test_full_dir, dtype=np.float32)
test_full, invalid_z_list = preprocess_input(test_full.reshape(img_shape))
test_full = ndimage.zoom(test_full, [t_shape/img_shape for t_shape, img_shape in zip(target_shape, test_full.shape)])
test_full_list = np.split(test_full, test_full.shape[0], axis=0)
test_full_list = [test_full_list[i] for i in range(0, len(test_full_list), len(test_full_list)//4)]

test_low = np.fromfile(test_low_dir, dtype=np.float32).reshape(img_shape)
test_low = np.delete(test_low, invalid_z_list, 0)
test_low /= 5e3
test_low -= 0.5
test_low /= 0.5
test_low = ndimage.zoom(test_low, [t_shape/img_shape for t_shape, img_shape in zip(target_shape, test_low.shape)])
test_low_list = sliding_window_view(test_low, window_shape=32, axis=0).transpose(0, 3, 1, 2)
test_low_list = np.concatenate([np.repeat(np.expand_dims(test_low_list[0, :, :, :], axis=0), 16, axis=0), test_low_list, np.repeat(np.expand_dims(test_low_list[-1, :, :, :], axis=0), 15, axis=0)], axis=0)
test_low_list = [test_low_list[i, :, :, :] for i in range(len(test_low_list))]
test_low = torch.stack([torch.Tensor(test_low_list[i]) for i in range(0, len(test_low_list), len(test_low_list)//4)], dim=0)
with torch.no_grad():
    test_low = test_low.cuda()

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

    ddpm = Diffusion(model_path=model_path)
    while True:
        mode = input('Choose generation mode (ddpm, ddim, or press Q to quit): ')
        if mode == 'q' or mode == 'Q':
            break
        # ddpm.generate_1x1_image(save_path_1x1, condition=low_slice)
        start = time.perf_counter()
        # ddpm.show_result(test_low.device, save_path, test_full_list, test_low, mode)
        ddpm.show_result_3d(test_low.device, save_path, test_full_list, test_low, mode) 
        end = time.perf_counter()
        print(f"Generation done, consuming {(end-start)}s.")
        
        # print("Generate_5x5_image")
        # ddpm.generate_5x5_image(save_path_5x5)
        # print("Generate_5x5_image Done")