import numpy as np
import torch

from sklearn.metrics import mean_squared_error

from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from art.estimators.regression import PyTorchRegressor

def get_adversarial_example_reg(params, model, X_test, y_test, epsilon):
    '''
    Get an adversarial example for a regression model
    '''
    if params.optimizer == 'amsgrad':
        optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate, amsgrad=True)
    elif params.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=params.learning_rate)
    criterion = params.loss_fn 
    model = model.float()
    regressor = PyTorchRegressor(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(1, params.num_feat),
    )
    if type(X_test) == type(np.ones(10)):
        X_test = X_test.astype(np.float32)
    else:
        X_test = X_test.type(torch.float32)

    y_test = y_test.astype(np.float32)
    attack = FastGradientMethod(estimator=regressor, eps=epsilon)
    x_test_adv = attack.generate(x=np.array(X_test), y=y_test)

    predictions = regressor.predict(x_test_adv)
    predictions = np.transpose(predictions)[0]
    loss = mean_squared_error(predictions, y_test)

    print("Loss on adversarial test examples: {}".format(loss))
    return x_test_adv, loss


def get_adversarial_example(params, model, X_test, y_test, epsilon, printmode = False):
    '''
    Get an adversarial example for a classification model
    '''
    if params.optimizer == 'amsgrad':
        optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate, amsgrad=True)
    elif params.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=params.learning_rate)
    criterion = params.loss_fn 
    model = model.float()
    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(1, params.num_feat),
        nb_classes=2,
    )
    if type(X_test) == type(np.ones(10)):
        X_test = X_test.astype(np.float32)
    else:
        X_test = X_test.type(torch.float32)
    attack = FastGradientMethod(estimator=classifier, eps=epsilon)
    x_test_adv = attack.generate(x=np.array(X_test))

    predictions = classifier.predict(x_test_adv)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.squeeze(y_test)) / len(y_test)
    if printmode:
        print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))
    return x_test_adv, accuracy

