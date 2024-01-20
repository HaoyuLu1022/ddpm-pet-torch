import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import scipy.ndimage as ndimage

from utils.utils import cvtColor, preprocess_input


class DiffusionDataset(Dataset):
    def __init__(self, annotation_lines, model_input_shape, img_shape):
        super(DiffusionDataset, self).__init__()

        self.annotation_lines   = annotation_lines
        self.length             = len(annotation_lines)
        self.model_input_shape        = model_input_shape
        self.img_shape = img_shape
        self.target_shape = (128, 128, 128)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        full_dir, low_dir = self.annotation_lines[index].split()
        full_slice = np.load(full_dir)
        low_slice = np.load(low_dir)
        return full_slice, low_slice
        # full_img = np.fromfile(full_dir, dtype=np.float32)
        # full_img, invalid_z_list = preprocess_input(full_img.reshape(self.img_shape))
        # full_img = ndimage.zoom(full_img, [target_shape/img_shape for target_shape, img_shape in zip(self.target_shape, full_img.shape)])
        # full_img_slices = np.split(full_img, full_img.shape[0], axis=0)
        
        # low_img = np.fromfile(low_dir, dtype=np.float32).reshape(self.img_shape)
        # low_img = np.delete(low_img, invalid_z_list, 0)
        # low_img /= 5e3
        # low_img -= 0.5
        # low_img /= 0.5
        # low_img = ndimage.zoom(low_img, [target_shape/img_shape for target_shape, img_shape in zip(self.target_shape, low_img.shape)])
        # # low_img_slices = np.split(low_img, low_img.shape[2], axis=2)
        # low_img_neighbor_slices = sliding_window_view(low_img, window_shape=32, axis=0).transpose(0, 3, 1, 2)
        # low_img_neighbor_slices = np.concatenate([np.repeat(np.expand_dims(low_img_neighbor_slices[0, :, :, :], axis=0), 16, axis=0), low_img_neighbor_slices, np.repeat(np.expand_dims(low_img_neighbor_slices[-1, :, :, :], axis=0), 15, axis=0)], axis=0)
        # low_img_neighbor_slices = [low_img_neighbor_slices[i, :, :, :] for i in range(len(low_img_neighbor_slices))]
        # # return full_img, low_img
        # return full_img_slices, low_img_neighbor_slices

def Diffusion_dataset_collate(batch):
    full_imgs = []
    low_imgs = []
    for images in batch:
        full_imgs.append(images[0])
        low_imgs.append(images[1])
    full_imgs = torch.from_numpy(np.array(full_imgs, dtype=np.float32))
    low_imgs = torch.from_numpy(np.array(low_imgs, dtype=np.float32))
    return {"fulldose": full_imgs, "lowdose": low_imgs}
