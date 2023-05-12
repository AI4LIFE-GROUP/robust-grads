from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
import numpy as np

def logistic_reg_l2(X, y, random_state=1129):
    clf = LogisticRegression(random_state=random_state).fit(X, y)
    return clf

def logistic_reg_l1(X, y, random_state=1129):
    clf = LogisticRegression(random_state=random_state, penalty='l1', solver='liblinear').fit(X,y)
    return clf

def logistic_reg_none(X, y, random_state=1129):
    clf = LogisticRegression(random_state=random_state, penalty='none', solver='saga').fit(X,y)
    return clf

def train_linear_models(args, train, test, random_state):
    ''' TODO update start_str, end_str, filenames to match what we use elsewhere so that postprocessing
    files can be called directly
    '''
    train_dataloader = DataLoader(train, batch_size=len(train), shuffle=True)
    test_dataloader = DataLoader(test, batch_size=len(test), shuffle=False)

    X, y = next(iter(train_dataloader))
    lr_model_l2 = logistic_reg_l2(X, y)
    lr_model_l1 = logistic_reg_l1(X, y)
    lr_model_none = logistic_reg_none(X, y)
    
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
