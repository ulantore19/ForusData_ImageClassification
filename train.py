import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
from torchvision import transforms


def train(model, num_epochs, train_dl, valid_dl, loss_fn, optimizerm, model_name="model.pth"):

    # Arrays to keep track of loss and accuracy for train and validation set
    loss_hist_train = [0] * num_epochs
    accuracy_hist_train = [0] * num_epochs
    loss_hist_valid = [0] * num_epochs
    accuracy_hist_valid = [0] * num_epochs
    
    for epoch in range(num_epochs):
        model.train()
        for x_batch, y_batch in train_dl:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device) 
            # forward propagation
            pred = model(x_batch)[:, 0]
            loss = loss_fn(pred, y_batch.float())
            # backward propagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            loss_hist_train[epoch] += loss.item()*y_batch.size(0)
            is_correct = ((pred>=0.5).float() == y_batch).float()
            accuracy_hist_train[epoch] += is_correct.sum().cpu()

        loss_hist_train[epoch] /= len(train_dl.dataset)
        accuracy_hist_train[epoch] /= len(train_dl.dataset)
        
        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in valid_dl:
                x_batch = x_batch.to(device) 
                y_batch = y_batch.to(device) 
                pred = model(x_batch)[:, 0]
                loss = loss_fn(pred, y_batch.float())
                loss_hist_valid[epoch] += loss.item()*y_batch.size(0) 
                is_correct = ((pred>=0.5).float() == y_batch).float()
                accuracy_hist_valid[epoch] += is_correct.sum().cpu()

        loss_hist_valid[epoch] /= len(valid_dl.dataset)
        accuracy_hist_valid[epoch] /= len(valid_dl.dataset)
        
        print(f'Epoch {epoch+1} accuracy: {accuracy_hist_train[epoch]:.4f} val_accuracy: {accuracy_hist_valid[epoch]:.4f}')
    
    torch.save(model.state_dict(), model_name)  # Model Saving


    return loss_hist_train, loss_hist_valid, accuracy_hist_train, accuracy_hist_valid


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model_base = EfficientNet.from_pretrained('efficientnet-b2')
        self.fc =  torch.nn.Sequential(
            nn.Linear(1000, 512),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, 256), 
            nn.Dropout(0.5),
            nn.ReLU(), 
            nn.Linear(256, 1), 
        )
    

    def forward(self, x):
        x = self.model_base(x)
        return self.fc(x)
