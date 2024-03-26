import numpy as np
import os
from PIL import Image
import scipy.ndimage as ndimage
from numpy.lib.stride_tricks import sliding_window_view
import multiprocessing as mp
from tqdm import tqdm

from utils.utils import cvtColor, preprocess_input

img_shape = (256, 256, 256)
target_shape = (128, 128, 128)
files = "train_lines.txt"
scale = 1e4
full_save_dir = "/media/bld/e644a83d-65c3-4f55-a408-bea0bee7f43e/haoyu/slices_12s/fulldose"
if not os.path.exists(full_save_dir):
    os.makedirs(full_save_dir)
low_save_dir = "/media/bld/e644a83d-65c3-4f55-a408-bea0bee7f43e/haoyu/slices_12s/lowdose"
if not os.path.exists(low_save_dir):
    os.makedirs(low_save_dir)
with open(files) as f:
    lines = f.readlines()

def slice_processing(line):
    full_dir, low_dir = line.split()
    full_img = np.fromfile(full_dir, dtype=np.float32) * scale
    full_img, invalid_z_list = preprocess_input(full_img.reshape(img_shape))
    full_img = ndimage.zoom(full_img, [target_shape/img_shape for target_shape, img_shape in zip(target_shape, full_img.shape)])
    full_img_slices = np.split(full_img, full_img.shape[0], axis=0)
    
    low_img = np.fromfile(low_dir, dtype=np.float32).reshape(img_shape) * scale
    low_img = np.delete(low_img, invalid_z_list, 0)
    low_img /= 0.004*scale #5e3
    low_img -= 0.5
    low_img /= 0.5
    # low_img = preprocess_input(low_img)
    low_img = ndimage.zoom(low_img, [target_shape/img_shape for target_shape, img_shape in zip(target_shape, low_img.shape)])
    # low_img_slices = np.split(low_img, low_img.shape[2], axis=2)
    low_img = np.pad(low_img, ((16, 15), (0, 0), (0, 0)), mode='constant', constant_values=0)
    low_img_neighbor_slices = sliding_window_view(low_img, window_shape=32, axis=0).transpose(0, 3, 1, 2)
    # low_img_neighbor_slices = np.concatenate([np.repeat(np.expand_dims(low_img_neighbor_slices[0, :, :, :], axis=0), 16, axis=0), low_img_neighbor_slices, np.repeat(np.expand_dims(low_img_neighbor_slices[-1, :, :, :], axis=0), 15, axis=0)], axis=0)
    low_img_neighbor_slices = [low_img_neighbor_slices[i, :, :, :] for i in range(len(low_img_neighbor_slices))]

    for i, (full_slice, low_slice) in enumerate(zip(full_img_slices, low_img_neighbor_slices)):
        np.save(f"{full_save_dir}/{full_dir[74:-4]}-{i}.npy", full_slice)
        np.save(f"{low_save_dir}/{low_dir[76:-4]}-{i}.npy", low_slice)

if __name__ == '__main__':
    # for line in tqdm(lines):
    #     slice_processing(line)

    with mp.Pool(14) as p:
        p.map(slice_processing, lines)
