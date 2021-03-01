from __future__ import print_function # future proof
import argparse
import sys
import os
import json

import pandas as pd

# pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

# import model
from model import MLP

def model_fn(model_dir):
    print("Loading model...")
    model_info={}
    model_info_path=os.path.join(model_dir,'model_info.pth')
    with open(model_info_path,'rb') as f:
        model_info=torch.load(f)
    print('Model info: {}'.format(model_info))
    
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=MLP(model_info['dim_input'],model_info['dim_hidden'],model_info['dim_output'])
    
    model_path=os.path.join(model_dir,'model.pth')
    with open(model_path,'rb'):
        model.load_state_dict(torch.load(f))
    
    return model.to(device)

def _get_train_loader(batch_size,data_dir):
    print("Get data loader")

    train_data = pd.read_csv(os.path.join(data_dir, "moon_train.csv"), header=None, names=None)

    train_y = torch.from_numpy(train_data[[0]].values).float().squeeze()
    train_x = torch.from_numpy(train_data.drop([0], axis=1).values).float()
    train_ds = torch.utils.data.TensorDataset(train_x, train_y)

    return torch.utils.data.DataLoader(train_ds, batch_size=batch_size)
    
# Provided train function
def train(model, train_loader, epochs, optimizer, criterion, device):
    """
    This is the training method that is called by the PyTorch training script.
    
    Parameters:
        model        - The PyTorch model that we wish to train.
        train_loader - The PyTorch DataLoader that should be used during training.
        epochs       - The total number of epochs to train for.
        optimizer    - The optimizer to use during training.
        criterion    - The loss function used for training. 
        device       - Where the model and data should be loaded (gpu or cpu).
    """
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader, 1):
            # prep data
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad() # zero accumulated gradients
            # get output of SimpleNet
            output = model(data)
            # calculate loss and perform backprop
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
            total_loss += loss.item()
        
        print("Epoch: {}, Loss: {}".format(epoch, total_loss / len(train_loader)))

    save_model(model, args.model_dir)
    

def save_model(model, model_dir):
    print("Saving the model.")
    path = os.path.join(model_dir, 'model.pth')
    torch.save(model.cpu().state_dict(), path)
    
def save_model_params(model, model_dir):
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'dim_input': args.dim_input,
            'dim_hidden': args.dim_hidden,
            'dim_output': args.dim_output
        }
        torch.save(model_info, f)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    # Training Parameters, given
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
  
    # Model parameters

    parser.add_argument('--dim_input', type=int, default=2, metavar='IN',
                        help='dimension of input')
        
    parser.add_argument('--dim_hidden', type=int, default=10, metavar='H',
                        help='dimension of hidden layer')
        
    parser.add_argument('--dim_output', type=int, default=1, metavar='OUT',
                        help='dimension of output layer')
    
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        
    train_loader = _get_train_loader(args.batch_size, args.data_dir)
    
    model = MLP(args.dim_input, args.dim_hidden, args.dim_output).to(device)
    
    save_model_params(model, args.model_dir)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    train(model, train_loader, args.epochs, optimizer, criterion, device)
    