import torch
from torchvision import transforms
import numpy as np 
import os
import pandas as pd
import random
## Create Dataset

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

def add_noise_to_labels(labels, n_classes = 15, epsilon = 0.1):
    # randomly sample epsilon label idxs.
    idxs = random.sample(range(len(labels)), int(len(labels) * epsilon))
    
    # randomly sample labels
    random_labels = random.choices(range(n_classes), weights = torch.ones(n_classes), k = int(len(labels) * epsilon))

    for idx1, idx2 in enumerate(idxs):
        labels[idx2] = random_labels[idx1]
    
    return labels

from PIL import Image
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_frame, transforms=None):
        self.data_frame = data_frame
        self.transforms = transforms
        self.len = data_frame.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        row = self.data_frame.iloc[index]
        address = row['path']
        x = Image.open(address).convert('RGB')
        
        # vec = np.array(row['disease_vec'], dtype=np.float)
        # y = torch.FloatTensor(vec)
        y = row['label']
        if self.transforms:
            x = self.transforms(x)
        return x, y
    

train_transform = transforms.Compose([ 
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(224, scale=(0.63, 1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])

test_transform = transforms.Compose([ 
    transforms.Resize(230),
    transforms.CenterCrop(224),
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)

def get_cxr14_data(batch_size = 16, train_epsilon = 0, valid_epsilon = 0):
    train_df = pd.read_csv('/cxr14/train.csv')
    valid_df = pd.read_csv('/cxr14/valid.csv')
    test_df = pd.read_csv('/cxr14/test.csv')

    if train_epsilon > 0:
        train_df['label'] = add_noise_to_labels(train_df['label'], n_classes=15, epsilon=train_epsilon)

    if valid_epsilon > 0:
        valid_df['label'] = add_noise_to_labels(valid_df['label'], n_classes=15, epsilon=valid_epsilon)

    dsetTrain = CustomDataset(train_df, train_transform) 
    dsetVal = CustomDataset(valid_df, test_transform) 
    dsetTest = CustomDataset(test_df, test_transform)

    trainloader = torch.utils.data.DataLoader( dataset = dsetTrain, batch_size = batch_size, shuffle = True, num_workers = 4 )
    valloader = torch.utils.data.DataLoader( dataset = dsetVal, batch_size = batch_size, shuffle = False, num_workers = 4 )
    testloader = torch.utils.data.DataLoader( dataset = dsetTest, batch_size = batch_size, shuffle = False, num_workers = 4 )

    return trainloader, valloader, testloader


def get_cxr14_datasets(batch_size = 16, train_epsilon = 0, valid_epsilon = 0):
    train_df = pd.read_csv('/cxr14/train.csv')
    valid_df = pd.read_csv('/cxr14/valid.csv')
    test_df = pd.read_csv('/cxr14/test.csv')

    dsetTrain = CustomDataset(train_df, train_transform) 
    dsetVal = CustomDataset(valid_df, test_transform) 
    dsetTest = CustomDataset(test_df, test_transform)

    return dsetTrain, dsetVal, dsetTest



def prepare_data():
    all_xray_df = pd.read_csv('/cxr14/Data_Entry_2017.csv')
    all_image_paths = {os.path.basename(x): x for x in glob(os.path.join('/datasets/cxr14', 'images*', '*', '*.png'))}
    print('Scans found:', len(all_image_paths), ', Total Headers', all_xray_df.shape[0])
    all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)
    label_counts = all_xray_df['Finding Labels'].value_counts()[:15]
    labels_list = ['Cardiomegaly', 
            'Emphysema', 
            'Effusion', 
            'Hernia', 
            'Infiltration', 
            'Mass', 
            'Nodule', 
            'Atelectasis',
            'Pneumothorax',
            'Pleural_Thickening', 
            'Pneumonia', 
            'Fibrosis', 
            'Edema', 
            'Consolidation']

    drop_column = ['Patient Age','Patient Gender','View Position','Follow-up #',
    'OriginalImagePixelSpacing[x','y]','OriginalImage[Width','Height]','Unnamed: 11']


    all_xray_df['Finding Labels'] = all_xray_df['Finding Labels'].map(lambda x: x.replace('No Finding', ''))
    from itertools import chain
    all_labels = np.unique(list(chain(*all_xray_df['Finding Labels'].map(lambda x: x.split('|')).tolist())))
    all_labels = [x for x in all_labels if len(x)>0]
    print('All Labels ({}): {}'.format(len(all_labels), all_labels))
    for c_label in all_labels:
        if len(c_label)>1: # leave out empty labels
            all_xray_df[c_label] = all_xray_df['Finding Labels'].map(lambda finding: 1 if c_label in finding else 0)


    all_xray_df = all_xray_df.drop(drop_column,axis=1)


    all_xray_df['disease_vec'] = all_xray_df.apply(lambda x: [x[all_labels].values], 1).map(lambda x: x[0])
    # Filter examples with no disease or only one disease
    all_xray_df['category_count'] = all_xray_df['disease_vec'].apply(lambda x: x.sum())
    all_xray_df_filtered = all_xray_df[all_xray_df['category_count'] <= 1].copy()
    all_xray_df_filtered['disease_vec_pad'] = all_xray_df_filtered['disease_vec'].apply(lambda x: np.append([0],x))
    all_xray_df_filtered['label'] = all_xray_df_filtered['disease_vec_pad'].apply(lambda x: np.argmax(x))

    train_df, valid_df, test_df = np.split(all_xray_df_filtered.sample(frac=1),
                                        [int(.6*len(all_xray_df_filtered)), int(.8*len(all_xray_df_filtered))])

    print('train', train_df.shape[0], 'validation', valid_df.shape[0], 'test', test_df.shape[0])

    if False:
        train_labels = []
        ds_len = train_df.shape[0]

        for inx in range(ds_len):
            row = train_df.iloc[inx]
            vec = np.array(row['disease_vec'], dtype=np.int)
            train_labels.append(vec)

        freq_pos, freq_neg = compute_class_freqs(train_labels)

        pos_weights = freq_neg
        neg_weights = freq_pos
        pos_contribution = freq_pos * pos_weights 
        neg_contribution = freq_neg * neg_weights


        # Prevent data leakage
        ids_train = train_df['Patient ID'].values
        ids_valid = valid_df['Patient ID'].values

        # Create a "set" datastructure of the training set id's to identify unique id's
        ids_train_set = set(ids_train)
        print(f'There are {len(ids_train_set)} unique Patient IDs in the training set')
        # Create a "set" datastructure of the validation set id's to identify unique id's
        ids_valid_set = set(ids_valid)
        print(f'There are {len(ids_valid_set)} unique Patient IDs in the training set')

        # Identify patient overlap by looking at the intersection between the sets
        patient_overlap = list(ids_train_set.intersection(ids_valid_set))
        n_overlap = len(patient_overlap)

        train_overlap_idxs = []
        valid_overlap_idxs = []
        for idx in range(n_overlap):
            train_overlap_idxs.extend(train_df.index[train_df['Patient ID'] == patient_overlap[idx]].tolist())
            valid_overlap_idxs.extend(valid_df.index[valid_df['Patient ID'] == patient_overlap[idx]].tolist())

        valid_df.drop(valid_overlap_idxs, inplace=True)
        # Extract patient id's for the validation set
        ids_valid = valid_df['Patient ID'].values
        # Create a "set" datastructure of the validation set id's to identify unique id's
        ids_valid_set = set(ids_valid)
        print(f'There are {len(ids_valid_set)} unique Patient IDs in the training set')
        # Identify patient overlap by looking at the intersection between the sets
        patient_overlap = list(ids_train_set.intersection(ids_valid_set))
        n_overlap = len(patient_overlap)
        print(f'There are {n_overlap} Patient IDs in both the training and validation sets')
        print("leakage between train and test: {}".format(check_for_leakage(train_df, valid_df, 'Patient ID')))

    train_df = train_df.drop('category_count', 1)
    valid_df = valid_df.drop('category_count', 1)
    test_df = test_df.drop('category_count', 1)
    train_df.to_csv('/cxr14/train.csv')
    valid_df.to_csv('/cxr14/valid.csv')
    test_df.to_csv('/cxr14/test.csv')

