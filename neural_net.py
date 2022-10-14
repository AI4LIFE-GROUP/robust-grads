import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from art.utils import load_mnist

import adversarial

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    correct = 0
    for batch, (X, y) in enumerate(dataloader):
        y = y[:,None].clone().detach()
        
        # Compute prediction and loss
        X = X.float()
        pred = model(X)
        
        y = torch.squeeze(y)
        if len(pred.shape) == 1:
            y = y.float()
        else:
            y = y.long()

        loss = loss_fn(pred, y)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        # compute training accuracy
        pred = torch.argmax(pred, dim=1)
        correct += (pred == y).type(torch.float).sum().item()
    correct /= size
    print(f"Training Accuracy: {(100*correct):>0.1f}\n")
    return correct


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.float()
            pred = model(X)
            y = y[:,None].clone().detach()# torch.tensor(y[:,None], dtype=torch.float)
            y = torch.squeeze(y)
            if len(pred.shape) == 1:
                y = y.float()
                test_loss += loss_fn(pred, y).item()
            else:
                y = y.long()
                test_loss += loss_fn(pred, y).item()
                pred = torch.argmax(pred, dim=1)
            correct += (pred == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct, pred

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()       
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )       
         # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output

class NeuralNetwork(nn.Module):
    def __init__(self, num_feat, num_classes=2, activation = nn.ReLU(), nodes_per_layer = 5, num_layers = 3, p = 0.0):
        super(NeuralNetwork, self).__init__()
        self.num_feat = num_feat
        self.num_classes = num_classes
        self.nodes_per_layer = nodes_per_layer
        self.num_layers = num_layers
        
        self.activation = activation
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            nn.Linear(self.num_feat, self.nodes_per_layer),
            self.activation,
        )

        for i in range(self.num_layers - 1):
            self.stack.add_module("linear"+str(i), nn.Linear(self.nodes_per_layer, self.nodes_per_layer))
            self.stack.add_module("activation"+str(i), self.activation)
        
        # add a dropout layer 
        self.stack.add_module("dropout", nn.Dropout(p=p))
        self.stack.add_module("final", nn.Linear(self.nodes_per_layer, self.num_classes))

    def forward(self, x):
        x = self.flatten(x)
        logits = self.stack(x)
        logits = np.squeeze(logits)
        return logits

def dnn_adversarial(train, test, params, dataset, random_state):
    torch.manual_seed(params.manual_seed)
    train_dataloader = DataLoader(train, batch_size=params.batch_size, shuffle=True)
    test_dataloader = DataLoader(test, batch_size=params.batch_size, shuffle=False)
    
    orig_labels = copy.deepcopy(train.labels)
    train.labels = np.transpose(np.array(train.labels))[0]
    orig_train = copy.deepcopy(train)
    

    if dataset == 'mnist':
        model = CNN()
    else:
        model = NeuralNetwork(params.num_feat, params.num_classes, params.activation, 
                          params.nodes_per_layer, params.num_layers, params.dropout)

    if params.optimizer == 'amsgrad':
        optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate, amsgrad=True)
    elif params.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=params.learning_rate)
    
    orig_lr = params.learning_rate
    for t in range(params.epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_acc = train_loop(train_dataloader, model, params.loss_fn, optimizer)
        if dataset in ['income', 'compas', 'whobin']:
            adv_examples, _ = adversarial.get_adversarial_example(params, model, orig_train.data, orig_train.labels, params.epsilon)
        else:
            adv_examples, _ = adversarial.get_adversarial_example_reg(params, model, orig_train.data, orig_train.labels, params.epsilon)
        
        test_acc, preds = test_loop(test_dataloader, model, params.loss_fn)
        print("Done!")

        train.data = adv_examples
        train_dataloader = DataLoader(train, batch_size=params.batch_size, shuffle=True)

        if params.lr_decay is not None:
            params.learning_rate = params.learning_rate * params.lr_decay
    
    params.learning_rate = orig_lr # reset for iterating on next random seed
    return model, test_acc, train_acc, preds

def dnn(train_dataloader, test_dataloader, params, dataset):
    torch.manual_seed(params.manual_seed)
    if dataset == 'mnist':
        model = CNN()
    else:
        model = NeuralNetwork(params.num_feat, params.num_classes, params.activation, 
                          params.nodes_per_layer, params.num_layers, params.dropout)
    epochs = params.epochs
    if params.optimizer == 'amsgrad':
        optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate, amsgrad=True)
    elif params.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=params.learning_rate)
    
    orig_lr = params.learning_rate
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_acc = train_loop(train_dataloader, model, params.loss_fn, optimizer)
        test_acc, preds = test_loop(test_dataloader, model, params.loss_fn)
        print("Done!")
        if params.lr_decay is not None:
            params.learning_rate = params.learning_rate * params.lr_decay
    
    params.learning_rate = orig_lr
    return model, test_acc, train_acc, preds