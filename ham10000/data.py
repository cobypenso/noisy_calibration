from glob import glob

import os, cv2, itertools
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
from torchvision import models,transforms
from torch.utils.data import DataLoader,Dataset
import numpy as np
from PIL import Image
from tqdm import tqdm
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

num_classes = 7

def add_noise_to_labels(labels, n_classes = 7, epsilon = 0.1):
    # randomly sample epsilon label idxs.
    idxs = random.sample(range(len(labels)), int(len(labels) * epsilon))
    
    # randomly sample labels
    random_labels = random.choices(range(n_classes), weights = torch.ones(n_classes), k = int(len(labels) * epsilon))

    for idx1, idx2 in enumerate(idxs):
        labels[idx2] = random_labels[idx1]
    
    return labels


class HAM10000(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Load data and get label
        if index >= len(self.df):
            import ipdb; ipdb.set_trace()
        X = Image.open(self.df['path'][index])
        y = torch.tensor(int(self.df['cell_type_idx'][index]))

        if self.transform:
            X = self.transform(X)

        return X, y


def get_ham10000(batch_size = 32, train_epsilon = 0, valid_epsilon = 0):
    norm_mean = (0.49139968, 0.48215827, 0.44653124)
    norm_std = (0.24703233, 0.24348505, 0.26158768)
    df_train = pd.read_csv('datasets/ham10000/train.csv')
    df_valid = pd.read_csv('datasets/ham10000/valid.csv')
    df_test = pd.read_csv('datasets/ham10000/test.csv')

    if train_epsilon > 0:
        df_train['cell_type_idx'] = add_noise_to_labels(df_train['cell_type_idx'], n_classes=7, epsilon=train_epsilon)

    if valid_epsilon > 0:
        df_valid['cell_type_idx'] = add_noise_to_labels(df_valid['cell_type_idx'], n_classes=7, epsilon=valid_epsilon)

    train_transform = transforms.Compose([transforms.Resize((224,224)),transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip(),transforms.RandomRotation(20),
                                      transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
                                        transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)])
    # define the transformation of the val images.
    val_transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
                                        transforms.Normalize(norm_mean, norm_std)])

    train_set = HAM10000(df_train, transform=train_transform)
    valid_set = HAM10000(df_valid, transform=val_transform)
    test_set = HAM10000(df_test, transform=val_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader

def preprocess_ham10000(seed = 42):
    data_dir = 'datasets/ham10000'
    all_image_path = glob(os.path.join(data_dir, '*', '*.jpg'))
    imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_path}
    lesion_type_dict = {
        'nv': 'Melanocytic nevi',
        'mel': 'dermatofibroma',
        'bkl': 'Benign keratosis-like lesions ',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc': 'Vascular lesions',
        'df': 'Dermatofibroma'
    }

    norm_mean,norm_std = compute_img_mean_std(all_image_path)


    df_original = pd.read_csv(os.path.join(data_dir, 'HAM10000_metadata.csv'))
    df_original['path'] = df_original['image_id'].map(imageid_path_dict.get)
    df_original['cell_type'] = df_original['dx'].map(lesion_type_dict.get)
    df_original['cell_type_idx'] = pd.Categorical(df_original['cell_type']).codes

    df_undup = df_original.groupby('lesion_id').count()
    # now we filter out lesion_id's that have only one image associated with it
    df_undup = df_undup[df_undup['image_id'] == 1]
    df_undup.reset_index(inplace=True)

    # here we identify lesion_id's that have duplicate images and those that have only one image.
    def get_val_rows(x):
        # create a list of all the lesion_id's in the val set
        val_list = list(df_val['image_id'])
        if str(x) in val_list:
            return 'val'
        else:
            return 'train'

    def get_duplicates(x):
        unique_list = list(df_undup['lesion_id'])
        if x in unique_list:
            return 'unduplicated'
        else:
            return 'duplicated'
            
    # create a new colum that is a copy of the lesion_id column
    df_original['duplicates'] = df_original['lesion_id']
    # apply the function to this new column
    df_original['duplicates'] = df_original['duplicates'].apply(get_duplicates)


    # now we filter out images that don't have duplicates
    df_undup = df_original[df_original['duplicates'] == 'unduplicated']
    # now we create a val set using df because we are sure that none of these images have augmented duplicates in the train set
    y = df_undup['cell_type_idx']
    _, df_val = train_test_split(df_undup, test_size=0.2, random_state=seed, stratify=y)

    # identify train and val rows
    # create a new colum that is a copy of the image_id column
    df_original['train_or_val'] = df_original['image_id']
    # apply the function to this new column
    df_original['train_or_val'] = df_original['train_or_val'].apply(get_val_rows)
    # filter out train rows
    df_train = df_original[df_original['train_or_val'] == 'train']

    y = df_val['cell_type_idx']
    df_val, df_test = train_test_split(df_val, test_size=0.5, random_state=seed, stratify=y)

    df_train.to_csv('datasets/ham10000/train.csv')
    df_val.to_csv('datasets/ham10000/valid.csv')
    df_test.to_csv('datasets/ham10000/test.csv')

    return df_train, df_val, df_test

def compute_img_mean_std(image_paths):
    """
        computing the mean and std of three channel on the whole dataset,
        first we should normalize the image from 0-255 to 0-1
    """

    img_h, img_w = 224, 224
    imgs = []
    means, stdevs = [], []

    for i in tqdm(range(len(image_paths))):
        img = cv2.imread(image_paths[i])
        img = cv2.resize(img, (img_h, img_w))
        imgs.append(img)

    imgs = np.stack(imgs, axis=3)
    print(imgs.shape)

    imgs = imgs.astype(np.float32) / 255.

    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()  # resize to one row
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    means.reverse()  # BGR --> RGB
    stdevs.reverse()

    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))
    return means,stdevs


# if __name__ == "__main__":
#     preprocess_ham10000(seed = 42)
