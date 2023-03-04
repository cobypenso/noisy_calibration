import torch
from torchvision import transforms
import numpy as np 
import os
import pandas as pd
import random
import medmnist
import PIL
from medmnist import INFO

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

## Create Dataset
def get_path_mnist_data(input_size = 28, batch_size = 32, download = False, as_rgb = False):
    info = INFO['pathmnist']
    DataClass = getattr(medmnist, info['python_class'])

    if input_size == 28:
        data_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])])
    elif input_size == 29:
        data_transform = transforms.Compose(
            [transforms.Resize((29, 29), interpolation=PIL.Image.NEAREST), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])])
    else:
        raise Exception()

     
    train_dataset = DataClass(split='train', transform=data_transform, download=download, as_rgb=as_rgb)
    val_dataset = DataClass(split='val', transform=data_transform, download=download, as_rgb=as_rgb)
    test_dataset = DataClass(split='test', transform=data_transform, download=download, as_rgb=as_rgb)

    return train_dataset, val_dataset, test_dataset