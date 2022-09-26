import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


import neural_net
import linear_model

class Params():
    def __init__(self, lr, lr_decay, epochs, batch_size, loss_fn, num_feat, num_classes, activation, nodes_per_layer, 
                 num_layers, optimizer = None, seed = 0, epsilon=0.5):
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


def train_linear_models(args, train, test, random_state):
    train_dataloader = DataLoader(train, batch_size=len(train), shuffle=True)
    test_dataloader = DataLoader(test, batch_size=len(test), shuffle=False)

    X, y = next(iter(train_dataloader))
    lr_model_l2 = linear_model.logistic_reg_l2(X, y)
    lr_model_l1 = linear_model.logistic_reg_l1(X, y)
    lr_model_none = linear_model.logistic_reg_none(X, y)
    
    preds_lr_l2 = lr_model_l2.predict(next(iter(test_dataloader))[0])
    preds_lr_l1 = lr_model_l1.predict(next(iter(test_dataloader))[0])
    preds_lr_none = lr_model_none.predict(next(iter(test_dataloader))[0])
    logits_lr_l2 = lr_model_l2.predict_proba(next(iter(test_dataloader))[0])
    logits_lr_l1 = lr_model_l1.predict_proba(next(iter(test_dataloader))[0])
    logits_lr_none = lr_model_none.predict_proba(next(iter(test_dataloader))[0])

    start_str = args.dataset + "_lr"
    end_str = str(random_state) + '.npy'
    np.save(start_str + 'l1_preds' + end_str, preds_lr_l1)
    np.save(start_str + 'l2_preds' + end_str, preds_lr_l2)
    np.save(start_str + 'none_preds' + end_str, preds_lr_none)
    np.save(start_str + 'l1_logits' + end_str, logits_lr_l1)
    np.save(start_str + 'l1_gradients' + end_str, lr_model_l1.coef_[0])
    np.save(start_str + 'l2_logits' + end_str, logits_lr_l2)
    np.save(start_str + 'l2_gradients' + end_str, lr_model_l2.coef_[0])
    np.save(start_str + 'none_logits' + end_str, logits_lr_none)
    np.save(start_str + 'none_gradients' + end_str, lr_model_none.coef_[0])

def eval_model(test_dataloader, model, start_str, random_state):
    logits = 0
    for X,y in test_dataloader:
        X = X.float()
        X.requires_grad = True
        if type(logits) == type(1):
            new_log = model(X)
            new_grads = torch.autograd.grad(outputs=new_log, inputs=X, grad_outputs=torch.ones_like(new_log))[0]
            logits = new_log.detach().numpy()
            grads = new_grads
            labels = y
        else:
            new_log = model(X)
            new_grads = torch.autograd.grad(outputs=new_log, inputs=X, grad_outputs=torch.ones_like(new_log))[0]
            logits = np.concatenate((logits, new_log.detach().numpy()), axis=0)
            grads = np.concatenate((grads, new_grads), axis=0)
            labels = np.concatenate((labels, y), axis=0)
    if len(logits.shape) == 1:
        preds_nn = logits
    else:
        preds_nn = np.argmax(logits,axis=1)

    end_str = str(random_state) + '.npy'
    np.save(start_str + 'preds' + end_str, preds_nn)
    np.save(start_str + 'logits' + end_str, logits)
    np.save(start_str + 'gradients' + end_str, grads)

def get_activation_term(activation):
    if type(activation) == type(nn.ReLU()):
        act = 'relu'
    elif type(activation) == type(nn.Softplus(beta = 5)):
        act = 'soft'
    else:
        act = 'leak'
    return act

def train_adv_nn(params, train, test, random_state, dataset, threshold=None):
    test_dataloader = DataLoader(test, batch_size=params.batch_size, shuffle=False)
    
    nn_model, test_acc, train_acc, preds = neural_net.dnn_adversarial(train, test, params, dataset, random_state)

    act = get_activation_term(params.activation)
    start_str = dataset + "_adv_nn" + str(params.num_layers) + "_" + act +  "_" 
    if threshold is not None:
        start_str = start_str + "t" + str(threshold) + "_"
    eval_model(test_dataloader, nn_model, start_str, random_state)

    return nn_model, test_acc, train_acc, preds

def train_nn(params, train, test, random_state, dataset, threshold=None):
    train_dataloader = DataLoader(train, batch_size=params.batch_size, shuffle=True)
    test_dataloader = DataLoader(test, batch_size=params.batch_size, shuffle=False)
    nn_model, test_acc, train_acc, preds = neural_net.dnn(train_dataloader, test_dataloader, params, dataset)

    act = get_activation_term(params.activation)
    start_str = dataset + "_nn" + str(params.num_layers) + "_" + act +  "_" 
    if threshold is not None:
        start_str = start_str + "t" + str(threshold) + "_"
    eval_model(test_dataloader, nn_model, start_str, random_state)
    return nn_model, test_acc, train_acc, preds