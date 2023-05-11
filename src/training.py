import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import neural_net
import linear_model

class Params():
    def __init__(self, lr, lr_decay, epochs, lime_epochs, batch_size, loss_fn, num_feat, num_classes, activation, nodes_per_layer, 
                 num_layers, optimizer = None, seed = 0, epsilon=0.2, dropout=0.0, weight_decay=0, step_size = 40, gamma=0.95):
        self.learning_rate = lr
        self.lr_decay = lr_decay
        self.epochs = epochs
        self.lime_epochs = lime_epochs
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



def train_adv_nn(params, train, test, random_state, dataset, output_dir, run_id, secondary_dataset=None, finetune = False, finetune_base = False):
    test_dataloader = DataLoader(test, batch_size=params.batch_size, shuffle=False)
    train_dataloader = DataLoader(train, batch_size=params.batch_size, shuffle=True)
    
    nn_model, test_acc, train_acc, secondary_acc, test_loss, train_loss = neural_net.dnn_adversarial(train, train_dataloader, test_dataloader, params, dataset, random_state, output_dir, run_id, secondary_dataset, finetune, finetune_base)

    return nn_model, test_acc, train_acc, secondary_acc, test_loss, train_loss

def train_nn(params, train, test, random_state, dataset, output_dir, run_id, secondary_dataset=None, finetune = False, finetune_base = False):
    
    train_dataloader = DataLoader(train, batch_size=params.batch_size, shuffle=True)
    test_dataloader = DataLoader(test, batch_size=params.batch_size, shuffle=False)

    nn_model, test_acc, train_acc, secondary_acc, test_loss, train_loss = neural_net.dnn(train_dataloader, test_dataloader, params, dataset, output_dir, secondary_dataset, random_state, run_id, finetune, finetune_base)

    return nn_model, test_acc, train_acc, secondary_acc, test_loss, train_loss