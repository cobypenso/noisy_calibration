import numpy as np


import wandb
# pytorch libraries
import torch
from torch import optim,nn
import time
import random
from data import *
from configs import get_args
from models import *

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model, train_data_loader, valid_data_loader, optimizer,
                n_epochs, criterion, batch_size, lr, epsilon, model_name, verbose = False):
    best_acc_val = 0
    start_time = time.time()
    for epoch in range(1, n_epochs+1):
        epoch_time = time.time()
        epoch_loss = 0
        correct = 0
        total=0
        if verbose:
            print("Epoch {} / {}".format(epoch, n_epochs))

        model.train()
        
        for inputs, labels in train_data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad() # zeroed grads
            outputs = model(inputs) # forward pass
            loss = criterion(outputs, labels) # softmax + cross entropy
            loss.backward() # back pass
            optimizer.step() # updated params
            epoch_loss += loss.item() # train loss
            _, pred = torch.max(outputs, dim=1)
            correct += (pred.cpu() == labels.cpu()).sum().item()
            total += labels.shape[0]
        acc = correct / total
        
        model.eval()
        a=0
        pred_val=0
        correct_val=0
        total_val=0
        with torch.no_grad():
            for inp_val, lab_val in valid_data_loader:
                inp_val = inp_val.to(device)
                lab_val = lab_val.to(device)
                out_val = model(inp_val)
                loss_val = criterion(out_val, lab_val)
                a += loss_val.item()
                _, pred_val = torch.max(out_val, dim=1)
                correct_val += (pred_val.cpu()==lab_val.cpu()).sum().item()
                total_val += lab_val.shape[0]
            acc_val = correct_val / total_val
        epoch_time2 = time.time()

        train_loss = epoch_loss/len(labels)
        valid_loss = a/len(lab_val)
        wandb.log({'epoch': epoch,
                   'epoch_time': epoch_time2-epoch_time,
                   'train_acc': acc,
                   'train_loss': train_loss,
                   'valid_acc': acc_val,
                   'valid_loss': valid_loss})
        if verbose:                   
            print("Duration: {:.0f}s, Train Loss: {:.4f}, Train Acc: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.4f}"
              .format(epoch_time2-epoch_time, epoch_loss/len(labels), acc, a/len(lab_val), acc_val))

        if acc_val > best_acc_val:
            best_acc_val = acc_val
            best_ckpt_epoch = epoch
            best_model_path = f'ham10000/ckpts/eps_{epsilon}/new_model_{model_name}_epoch_{epoch}_{n_epochs}_lr_{lr}_bs_{batch_size}.pth'
            torch.save(model, best_model_path)

    end_time = time.time()
    total_time = end_time - start_time
    wandb.log({'total_time':total_time})

    if verbose:
        print("Total Time:{:.0f}s".format(end_time-start_time))
    return best_model_path

def test_model(model, test_data_loader, verbose = False):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_data_loader:
            images, labels = data
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = float(correct)/float(total)*100

    if verbose:
        print(f"Accuracy of the network on the test images: {test_acc:.4f}")
    
    wandb.log({'test_acc': test_acc})

    return test_acc


def main(args):
    print (f'---> Noise level {args.epsilon}')
    train_data_loader, valid_data_loader, test_data_loader =  get_ham10000(batch_size=args.batch_size, 
                                                                            train_epsilon=args.epsilon,
                                                                            valid_epsilon=args.epsilon)
    model, _ = get_model(args.model_name)
    model.to(device)

    
    criterion = nn.CrossEntropyLoss().to(device)
    for lr in [0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001]:
        args.lr = lr
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        with wandb.init(project='ham10000_with_noisy_labels'):
            wandb.config.update(args)
            print ('=== Start Training ===')
            best_model_path = train_model(model = model, 
                                train_data_loader = train_data_loader, 
                                valid_data_loader = valid_data_loader, 
                                optimizer = optimizer,
                                n_epochs = args.n_epochs, 
                                criterion = criterion, 
                                batch_size = args.batch_size,
                                lr=args.lr,
                                epsilon=args.epsilon,
                                model_name= args.model_name)
            print ('=== End Training ===')
            print ('=== Start Testing ===')
            model = torch.load(best_model_path)
            acc = test_model(model, test_data_loader)
            wandb.log({'test_accuracy': acc})
            print ('=== End Testing ===')


if __name__ == "__main__":
    args = get_args()
    main(args)
