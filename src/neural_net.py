import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from captum.attr import Saliency
from captum.attr import NoiseTunnel
from captum.attr import Lime
from captum.attr import KernelShap

def train_loop(dataloader, model, loss_fn, optimizer, scheduler, printmode=False):
    size = len(dataloader.dataset)
    train_loss, correct = 0, 0
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

        losscur = loss_fn(pred, y)
        train_loss += losscur
        
        # Backpropagation
        optimizer.zero_grad()
        losscur.backward()
        optimizer.step()
        scheduler.step()

        if batch % 100 == 0:
            if printmode:
                losscur, current = losscur.item(), batch * len(X)
                print(f"loss: {losscur:>7f}  [{current:>5d}/{size:>5d}]")

        # compute training accuracy
        pred = torch.argmax(pred, dim=1)
        correct += (pred == y).type(torch.float).sum().item()
    correct /= size
    train_loss /= size
    if printmode:
        print(f"Training Accuracy: {(100*correct):>0.1f}\n")
    return correct, train_loss


def test_loop(dataloader, model, loss_fn, printmode=False):
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
    correct /= size
    test_loss /= size
    if printmode:
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct, test_loss

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


def get_activation_term(activation):
    if type(activation) == type(nn.ReLU()):
        act = 'relu'
    elif type(activation) == type(nn.Softplus(beta = 5)):
        act = 'soft'
    else:
        act = 'leak'
    return act

class Modified_Model_Lime(nn.Module):
    def __init__(self, model, y):
        super(Modified_Model_Lime, self).__init__()
        self.model = model
        self.y = y
        self.loss_fn = nn.CrossEntropyLoss()
    def forward(self, x):
        logits = self.model(x)
        # take softmax of logits
        sm = torch.nn.functional.softmax(logits, dim=0) # shape (batch_size, 2)
        loss = self.loss_fn(sm, self.y)
        return loss

class Modified_Model(nn.Module):
    def __init__(self, model, y):
        super(Modified_Model, self).__init__()
        self.model = model
        self.y = y
        self.loss_fn = nn.CrossEntropyLoss()
    def forward(self, x):
        logits = self.model(x)
        # take softmax of logits
        sm = torch.nn.functional.softmax(logits, dim=1) # shape (batch_size, 2)
        loss = torch.zeros(len(x))
        i=0
        for input, label in zip(sm,self.y):
            # update index (1,i) with loss of x,y
            loss[i] = self.loss_fn(input, label)
            i+=1
        return loss

def eval_model(test_dataloader, model, start_str, random_state,
               test_lime, total_lime_samples=100, n_samples=100):
    total_samples = len(test_dataloader.dataset)
    total_features = len(test_dataloader.dataset[0][0])
    batch_size = test_dataloader.batch_size

    logits = torch.zeros(total_samples, 2)
    labels = torch.zeros(total_samples)
    grads = torch.zeros(total_samples, total_features)
    salience = torch.zeros(total_samples, total_features)
    smoothgrad = torch.zeros(total_samples, total_features)
    lime_grads = torch.zeros(total_lime_samples, total_features)
    kernel_shap = torch.zeros(total_lime_samples, total_features)
    
    batch = 0
    total_lime_queries = 0
    for X,y in test_dataloader:
        mm = Modified_Model(model, y)
        sal = Saliency(mm)
        sg = NoiseTunnel(sal)

        X = X.float()
        X.requires_grad = True

        new_log = model(X)
        end_num = min(batch*batch_size + batch_size, total_samples)

        logits[(batch*batch_size):end_num] = new_log.detach()
        labels[(batch*batch_size):end_num] = y
        grads[(batch*batch_size):end_num] = torch.autograd.grad(outputs=new_log, inputs=X, grad_outputs=torch.ones_like(new_log))[0]
        salience[(batch*batch_size):end_num] = sal.attribute(X, abs=False)
        smoothgrad[(batch*batch_size):end_num] = sg.attribute(X, nt_type = 'smoothgrad', nt_samples = 10, abs=False)

        if test_lime and (total_lime_queries < total_lime_samples):
            i = 0
            for x in X:
                if total_lime_queries >= total_lime_samples:
                    break
                x = x.unsqueeze(0)
                mm_lime = Modified_Model_Lime(model, y[i])
                lime = Lime(mm_lime)
                ks = KernelShap(mm_lime)

                lime_grads[total_lime_queries] = lime.attribute(x, n_samples=n_samples)
                kernel_shap[total_lime_queries] = ks.attribute(x)

                total_lime_queries += 1
                i += 1
        
        batch += 1
    if len(logits.shape) == 1:
        preds_nn = logits
    else:
        preds_nn = np.argmax(logits,axis=1)

    end_str = str(random_state) + '.npy'
    np.save(start_str + 'gradients' + end_str, grads)
    np.save(start_str + 'preds' + end_str, preds_nn)
    np.save(start_str + 'logits' + end_str, logits)
    np.save(start_str + 'salience' + end_str, salience)
    np.save(start_str + 'smoothgrad' + end_str, smoothgrad)
    if test_lime:
        np.save(start_str + 'lime' + end_str, lime_grads)
        np.save(start_str + 'shap' + end_str, kernel_shap)

    outputs = [preds_nn, logits, grads, salience, smoothgrad]
    if test_lime:
        outputs += [lime_grads, kernel_shap]
    return outputs

def dnn(train_dataloader, test_dataloader, params, dataset, output_dir, secondary_dataset, random_state, run_id, finetune, finetune_base):
    torch.manual_seed(params.manual_seed)
    if dataset == 'mnist':
        model = CNN()
    else:
        model = NeuralNetwork(params.num_feat, params.num_classes, params.activation, 
                          params.nodes_per_layer, params.num_layers, params.dropout)
        if finetune:
            print("loading model state with seed", params.manual_seed," and program random state", random_state)
            model.load_state_dict(torch.load(output_dir + 'model_' + str(params.manual_seed) + '.pt'))

    epochs = params.epochs
    lime_epochs = params.lime_epochs
    if params.optimizer == 'amsgrad':
        optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate, amsgrad=True, weight_decay=params.weight_decay)
    elif params.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params.step_size, gamma=params.gamma, last_epoch=-1, verbose=False)
    orig_lr = params.learning_rate
    
    # for real-world dataset shift
    secondary_dataloader = None
    if secondary_dataset is not None:
        secondary_dataloader = DataLoader(secondary_dataset, batch_size=params.batch_size, shuffle=False)

    all_acc_train, all_acc_test, all_sec_acc = np.zeros(max(epochs)), np.zeros(max(epochs)), None
    all_loss_train, all_loss_test = np.zeros(max(epochs)), np.zeros(max(epochs))
    if secondary_dataset is not None:
        all_sec_acc = np.zeros(max(epochs))
    
    for t in range(max(epochs)):
        printmode = False
        if ((t+1)%5 == 0):
            print(f"Epoch {t+1}\n-------------------------------")
            printmode = True
        all_acc_train[t], all_loss_train[t] = train_loop(train_dataloader, model, params.loss_fn, optimizer, scheduler, printmode)
        all_acc_test[t], all_loss_test[t] = test_loop(test_dataloader, model, params.loss_fn, printmode)

        if ((t + 1) in epochs):
            start_str = output_dir + "/results_" + run_id + "_e" + str(t+1) + "_" 
            # for real-world dataset shift, compute test stats on the both the original and shifted data
            test_lime = False
            if ((t + 1) in lime_epochs):
                test_lime = True
            if secondary_dataset is not None:
                if 'orig' in dataset:
                    eval_model(test_dataloader, model, start_str, random_state, test_lime) # original, partial dataset
                    eval_model(secondary_dataloader, model, start_str + "full_", random_state, test_lime) # full, shifted data
                    all_sec_acc[t], _ = test_loop(secondary_dataloader, model, params.loss_fn, test_lime)
                else:
                    eval_model(secondary_dataloader, model, start_str, random_state, test_lime) # original, partial dataset
                    eval_model(test_dataloader, model, start_str + "full_", random_state, test_lime) # full, shifted data
                    all_sec_acc[t], _ = test_loop(secondary_dataloader, model, params.loss_fn)
            else:
                eval_model(test_dataloader, model, start_str, random_state, test_lime)

    params.learning_rate = orig_lr
    if finetune_base:
        torch.save(model.state_dict(), output_dir + 'model_' + str(params.manual_seed) + '.pt')
    elif finetune:
        orig_model = NeuralNetwork(params.num_feat, params.num_classes, params.activation, 
                          params.nodes_per_layer, params.num_layers, params.dropout)
        orig_model.load_state_dict(torch.load(output_dir + 'model_' + str(params.manual_seed) + '.pt'))

        diff_squared=0
        for s,m in zip(orig_model.stack, model.stack):
            if type(s) == torch.nn.modules.linear.Linear:
                diff_squared += ((s.weight-m.weight)**2).sum().item()
        diff_squared = diff_squared ** 0.5

        np.save("theta_diff_" + run_id + "_" + str(random_state) + ".npy",diff_squared)
    else:
        # save model so we can compute theta diff later
        torch.save(model.state_dict(), output_dir + '/model_' + run_id + str(random_state) + '.pt')
        
    return model, all_acc_test, all_acc_train, all_sec_acc, all_loss_test, all_loss_train
