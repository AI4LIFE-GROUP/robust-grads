import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import neural_net

class Params():
    def __init__(self, lr, lr_decay, epochs, batch_size, loss_fn, num_feat, num_classes, activation, nodes_per_layer, 
                 num_layers, optimizer = None, seed = 0, epsilon=0.5, dropout=0.0, weight_decay=0.0, step_size=40, gamma=0.95):
        self.learning_rate = lr
        self.lr_decay = lr_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.num_feat = num_feat
        self.num_classes = num_classes
        self.activation = activation
        self.nodes_per_layer = nodes_per_layer
        self.num_layers = num_layers
        self.optimizer = optimizer
        self.manual_seed = seed
        self.epsilon = epsilon
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.step_size = step_size
        self.gamma = gamma

def eval_model(test_dataloader, model, start_str, random_state):
    logits = 0
    
    for X,y in test_dataloader:
        X = X.float()
        X.requires_grad = True
        if type(logits) == type(1):
            # Gradients wrt output
            new_log = model(X)
            grads = torch.autograd.grad(outputs=new_log, inputs=X, grad_outputs=torch.ones_like(new_log))[0]
            
            # Gradients wrt loss
            new_log = model(X)  # requires new variable for gradients
            soft = torch.nn.functional.softmax(new_log, dim=1)
            losses = -(torch.log(soft[:, 1])*y+torch.log(soft[:,0])*(1-y))
            loss_grads = torch.autograd.grad(outputs=losses, inputs=X, grad_outputs=torch.ones_like(losses))[0]
            
            logits = new_log.detach().numpy()
            labels = y
        else:
            # Gradients wrt output
            new_log = model(X)
            new_grads = torch.autograd.grad(outputs=new_log, inputs=X, grad_outputs=torch.ones_like(new_log))[0]
            
            # Gradients wrt loss
            new_log = model(X)
            soft = torch.nn.functional.softmax(new_log, dim=1)
            losses = -(torch.log(soft[:, 1])*y+torch.log(soft[:,0])*(1-y))
            new_loss_grads = torch.autograd.grad(outputs=losses, inputs=X, grad_outputs=torch.ones_like(losses))[0]
            
            # Append new arrays
            grads = np.concatenate((grads, new_grads), axis=0)
            loss_grads = np.concatenate((loss_grads, new_loss_grads), axis=0)
            logits = np.concatenate((logits, new_log.detach().numpy()), axis=0)
            labels = np.concatenate((labels, y), axis=0)
    if len(logits.shape) == 1:
        preds = logits
    else:
        preds = np.argmax(logits,axis=1)
    
    return preds, logits, grads
    #end_str = str(random_state) + '.npy'
    #np.save(start_str + 'preds' + end_str, preds)
    #np.save(start_str + 'logits' + end_str, logits)
    #np.save(start_str + 'gradients' + end_str, grads)
        

def get_activation_term(activation):
    if type(activation) == type(nn.ReLU()):
        act = 'relu'
    elif type(activation) == type(nn.Softplus(beta = 5)):
        act = 'soft'
    else:
        act = 'leak'
    return act

def train_adv_nn(params, train, test, random_state, dataset, threshold=None, plot=False, print_terms=True, base=None):
    outputs = neural_net.dnn_adversarial(train, test, params, dataset, plot=plot, print_terms=print_terms, base=base)
    return outputs

def train_nn(params, train, test, dataset, plot=False, print_terms=True, base=None, adversarial=False):
    if adversarial:
        outputs = neural_net.dnn_adversarial(train, test, params, dataset, plot=plot,
                                             print_terms=print_terms, base=base)
    else:
        train_dataloader = DataLoader(train, batch_size=params.batch_size, shuffle=True)
        test_dataloader = DataLoader(test, batch_size=params.batch_size, shuffle=False)
        outputs = neural_net.dnn(train_dataloader, test_dataloader, params, dataset, plot=plot,
                                 print_terms=print_terms, base=base)
    return outputs