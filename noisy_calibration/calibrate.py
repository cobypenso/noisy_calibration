import sys
sys.path.append('../')
sys.path.append('../cxr14')
sys.path.append('../path_mnist')
sys.path.append('../ham10000')
import os
from sklearn.model_selection import train_test_split
import argparse
import matplotlib.pyplot as plt
import random
import torch
import numpy as np

from utils import TempScalingOnECE, TempScalingOnAdaECE, ECELoss
from ham10000.data import  get_ham10000
from cxr14.data import get_cxr14_datasets
from path_mnist.data import get_path_mnist_data

use_cuda = True
device = 'cuda' if torch.cuda.is_available() and use_cuda else  'cpu'


def add_noise_to_labels(labels, n_classes = 10, epsilon = 0.1):
    # randomly sample epsilon label idxs.
    idxs = random.sample(range(len(labels)), int(len(labels) * epsilon))

    if type(labels) == torch.Tensor:
        labels = labels.detach()
        new_labels = labels.clone().detach()
    else:
        new_labels = labels.copy()

    for _, idx in enumerate(idxs):
        if type(new_labels[idx]) == int:
            old_label = new_labels[idx]
        else:
            old_label = new_labels[idx].item()
            
        optional_labels = list(range(n_classes))
        optional_labels.remove(old_label)
        
        new_labels[idx] = random.choices(optional_labels, weights = torch.ones(n_classes - 1))[0]

    
    return new_labels


def calibrate(valid_logits, 
              val_labels, 
              test_logits, 
              test_labels,
              n_classes, 
              epsilon,
              tmp_scaler):
        
    # Generate Noise   
    noisy_val_labels = add_noise_to_labels(val_labels,
                                            n_classes = n_classes,
                                            epsilon=epsilon)

    # Calibration and Accuracy of original model
    val_acc = (torch.argmax(valid_logits, dim = -1) == torch.tensor(val_labels)).sum() / len(val_labels)
    noisy_val_acc = (torch.argmax(valid_logits, dim = -1) == torch.tensor(noisy_val_labels)).sum() / len(noisy_val_labels)
    test_acc = (torch.argmax(test_logits, dim = -1) == torch.tensor(test_labels)).sum() / len(test_labels)
    ece_loss = ECELoss()
    
    # Original ECE Before calibration
    val_ece_before = ece_loss(valid_logits, torch.tensor(noisy_val_labels))
    ece_before = ece_loss(test_logits, torch.tensor(test_labels))

    # TS-Clean
    calib_model = tmp_scaler()
    oracle_opt_temp = calib_model.find_best_T(valid_logits, torch.tensor(val_labels), optimizer='brute')
    ece_after_oracle = ece_loss(test_logits/ oracle_opt_temp, torch.tensor(test_labels))
    val_ece_after_oracle = ece_loss(valid_logits/ oracle_opt_temp, torch.tensor(val_labels))

    # TS-noise
    calib_model = tmp_scaler()
    noisy_opt_temp = calib_model.find_best_T(valid_logits, torch.tensor(noisy_val_labels), optimizer='brute')
    ece_with_noise_val = ece_loss(valid_logits / noisy_opt_temp, torch.tensor(noisy_val_labels))
    ece_with_noise = ece_loss(test_logits/ noisy_opt_temp, torch.tensor(test_labels))

    # NTS
    calib_model = tmp_scaler(noisy_labels = True, epsilon = epsilon)
    ours_opt_temp = calib_model.find_best_T(valid_logits, torch.tensor(noisy_val_labels), optimizer='brute')
    ece_with_NECE = ece_loss(test_logits/ ours_opt_temp, torch.tensor(test_labels))
    ece_with_NECE_val = ece_loss(valid_logits/ ours_opt_temp, torch.tensor(noisy_val_labels))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--result_path", type=str, default="models_sweep_noisy_calibration")
    parser.add_argument("--opt_critic", type = str, default='adaece', choices= ['ece', 'adaece'])
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()
    
    print (f'---> Noise level {args.epsilon}')


    # ------------- Get configs -------------
    if args.dataset == 'ham10000':
        get_data = get_ham10000
        n_classes = 7
    elif args.dataset == 'cxr14':
        get_data = get_cxr14_datasets
        n_classes = 15
    elif args.dataset == 'path_mnist':
        get_data = get_path_mnist_data
        n_classes = 9
    else:
        raise Exception(f'Dataset {args.dataset} is not supported')


    if args.dataset == 'ham10000':
        train_data_loader, valid_data_loader, test_data_loader =  get_data(batch_size=args.batch_size)
        valid_dataset = valid_data_loader.dataset
        test_dataset = test_data_loader.dataset
        val_labels = valid_dataset.df['cell_type_idx'].to_list()
        test_labels = test_dataset.df['cell_type_idx'].to_list()
    elif args.dataset == 'cxr14':
        train_data, valid_dataset, test_dataset =  get_data(batch_size=args.batch_size)
        val_labels = torch.tensor((valid_dataset.data_frame['label']).tolist())
        test_labels = test_dataset.data_frame['label']
    elif args.dataset == 'path_mnist':
        if args.model_name == 'densenet121':
            input_size = 29
        else:
            input_size = 28
        _, _, test_dataset =  get_data(input_size = input_size)
        test_labels = torch.tensor([item[1].item() for item in test_dataset])
        valid_dataset, test_dataset = train_test_split(test_dataset, train_size=0.5, stratify=test_labels)
        val_labels = torch.tensor([item[1].item() for item in valid_dataset])
        test_labels = torch.tensor([item[1].item() for item in test_dataset])



    # ------------- Type of ECE Critiria to optimize by ------------
    if args.opt_critic == 'ece':
        print ('using ECE')
        tmp_scaler = TempScalingOnECE
    elif args.opt_critic == 'adaece':
        print ('using AdaECE')
        tmp_scaler = TempScalingOnAdaECE
    else:
        tmp_scaler = None

    print (f'Working on {args.model_path}')


    print ('load logits')
    valid_logits = torch.load(args.result_path.replace('ece/', '') + 'valid_logits.pt') 
    test_logits = torch.load(args.result_path.replace('ece/', '')  + 'test_logits.pt')
    print ('Valid size:', len(valid_logits))
    print ('Test size:', len(test_logits))
    
    # ------- Calibrate --------- #
    print (f'--> seed {args.seed}')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    calibrate(valid_logits=valid_logits, 
        val_labels=val_labels, 
        test_logits=test_logits, 
        test_labels=test_labels, 
        n_classes=n_classes, 
        epsilon=args.epsilon,
        tmp_scaler=tmp_scaler)