from torch.utils.data import DataLoader
import neural_net

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