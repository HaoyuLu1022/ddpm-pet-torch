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
        # image   = Image.open(self.annotation_lines[index].split()[0])
        # image   = cvtColor(image).resize([self.model_input_shape[1], self.model_input_shape[0]], Image.BICUBIC)
        
        # image   = np.array(image, dtype=np.float32)
        # image   = np.transpose(preprocess_input(image), (2, 0, 1))
        # model_input_shape = (400, 400, 400)
        image = np.fromfile(self.annotation_lines[index].split()[0], dtype=np.float32)
        image = preprocess_input(image.reshape(self.img_shape).transpose(1, 2, 0))
        z_scale = 128/image.shape[2]
        image = ndimage.zoom(image, [0.64, 0.64, z_scale])

        # valid_img = image[18:(18+363), 18:(18+363), 68:332] # center 363 pixels along x and y axes, and center 264 slices along z axis
        # standardization to rescale images ito [-1, 1]
        return image

def Diffusion_dataset_collate(batch):
    images = []
    for image in batch:
        images.append(image)
    images = torch.from_numpy(np.array(images, dtype=np.float32))
    return images
