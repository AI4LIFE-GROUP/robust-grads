import argparse
import numpy as np
import pandas as pd

# import decision trees
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler

import utils.metrics as metrics
import utils.datasets as datasets

import lime
import lime.lime_tabular
import shap

# silence the warning 'UserWarning: X has feature names, but StandardScaler was fitted without feature names'
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# Perturb the dataset -- assumes X is a tensor
def perturb_X(X, random_state, std_dv):
    # change X to np matrix
    X = np.asarray(X)

    # create a random matrix of the same shape as X
    random_matrix = np.random.RandomState(random_state).normal(0, std_dv, X.shape)

    # add the random matrix to X
    data_new = X + random_matrix
    return data_new

def explain_lime(model, train_X, test_X, num_to_run):
    explainer = lime.lime_tabular.LimeTabularExplainer(train_X,  class_names=['0', '1'], 
                                                       discretize_continuous=False, random_state=0)
    lime_exps = np.zeros((num_to_run, test_X.shape[1]))

    for idx in range(num_to_run):
        exp = explainer.explain_instance(test_X[idx], model.predict_proba,
                                         num_features = train_X.shape[0], num_samples = 5000)
        # note: EXP has shape (num_features, num_classes)
        feature_idxs = [int(x[0]) for x in exp.as_list()]
        lime_values = [i[1] for i in exp.as_list()]
        lime_exps[idx, feature_idxs] = np.array(lime_values)

    # lime_exps has shape (num_to_run, num_features)
    return np.array(lime_exps)


def explain_shap(model, preds, test_X, num_to_run):
    # note: preds is not necessarily WRT model, but WRT to a base model because otherwise comparing explanations 
    #      doesn't make sense
    explainer = shap.Explainer(model)
    shap_test = explainer(test_X[:num_to_run])

    # XGBoost case - SHAP values are already 2-dimensional
    if len(shap_test.values.shape) == 2:
        return shap_test.values
    # convert shap_test from a (num_to_run, num_features, num_classes) array to a (num_to_run, num_features) array
    # where shap_test_new[i,j] = shap_test[i,j,preds[i]]
    shap_test_new = np.zeros((num_to_run, test_X.shape[1]))
    for i in range(num_to_run):
        for j in range(test_X.shape[1]):
            shap_test_new[i,j] = shap_test.values[i,j,preds[i]]
    return shap_test_new

def save_metric_data(all_met_lime3, all_met_lime5, all_met_shap3, all_met_shap5, argstring, num_trials):
    print("all_met_lime3 shape is ",all_met_lime3.shape)
    print("all_met_shap5 shape", all_met_shap5.shape)
    col_names, results = [], np.zeros([num_trials,12])
    i=0
    for k, name, scores in zip(['3','5','3','5'], ['lime','lime','shap','shap'], 
                            [all_met_lime3, all_met_lime5, all_met_shap3, all_met_shap5]):
        for idx, met in enumerate(['sa','cdc','ssa']):
            
            col_names.append(name + "_top" + k + "_" + met)
            results[:,i] = scores[idx]
            i+=1
    results = np.array(results)
    print("col_names is ",col_names)
    print("results are ",results)
    df = pd.DataFrame(results, columns=col_names)
    df.to_csv(argstring + ".csv", index=False)
    #np.save(argstring + ".npy", np.concatenate([col_names, results]))
    
    
def real_world_shift(args, train_X, train_y, test_X, test_y, argstring):

    sec_file = args.file_base.replace('orig', 'shift')
    sec_train, sec_test = datasets.load_data(sec_file, args.dataset, 0, 0, add_noise=False, label_col=args.label_col)
    sec_train_X, sec_train_y = sec_train.data, sec_train.labels
    sec_test_X, sec_test_y = sec_test.data, sec_test.labels

    if args.model_type == 'dt':
        base_model = DecisionTreeClassifier(random_state=0, max_depth=args.max_depth, 
                                            min_samples_leaf=args.min_samples_leaf)
    else:
        base_model = XGBClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth_xgb, 
                                   learning_rate=args.learning_rate, reg_lambda = args.reg_lambda,
                                   random_state=0)
    base_model.fit(train_X, train_y)

    test_acc = base_model.score(test_X, test_y)
    sec_test_acc = base_model.score(sec_test_X, sec_test_y)
    train_acc = base_model.score(train_X, train_y)

    if args.model_type == 'dt':
        shifted_model = DecisionTreeClassifier(random_state=0, max_depth=args.max_depth, 
                                               min_samples_leaf=args.min_samples_leaf)
    else:
        shifted_model = XGBClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth_xgb, 
                                   learning_rate=args.learning_rate, reg_lambda = args.reg_lambda, 
                                   random_state=0)
    shifted_model.fit(sec_train_X, sec_train_y)

    secondary_acc = shifted_model.score(sec_train_X, sec_train_y)
    secondary_test_acc = shifted_model.score(sec_test_X, sec_test_y)

    np.save(argstring + "_accuracy.npy", np.array([train_acc, test_acc, sec_test_acc, secondary_acc, secondary_test_acc]))

    preds = base_model.predict(sec_test_X[:args.num_lime])
    lime_base = explain_lime(base_model, train_X, sec_test_X, args.num_lime)
    lime_shift = explain_lime(shifted_model, sec_train_X, sec_test_X, args.num_lime)
    shap_base = explain_shap(base_model, preds, sec_test_X, args.num_lime)
    shap_shift = explain_shap(shifted_model, preds, sec_test_X, args.num_lime)
    limes = [lime_base, lime_shift]
    shaps = [shap_base, shap_shift]
    
    col_names, results = [], []
    for technique, scores in zip(['lime','shap'],[limes, shaps]):
        signs = np.sign(scores[0])
        signs_shift = np.sign(scores[1])
        for k in [3,5]:
            top_k = metrics.get_top_k(k, scores[0])
            top_k_shift = metrics.get_top_k(k, scores[1])

            for name, met in zip(['sa', 'cdc', 'ssa'], [metrics.top_k_sa, metrics.top_k_cdc, metrics.top_k_ssd]):
                all_met = met(k,  top_k, top_k_shift, signs, signs_shift)
                results.append(all_met)
                col_names.append(technique + "_" + str(k) + "_" + name)

    np.save(argstring + ".npy", [col_names, results])
    

def synthetic_shift(args, train_X, train_y, test_X, test_y, argstring):
    train_accuracy, test_accuracy = [], []
    orig_scalar = StandardScaler()
    orig_scalar.fit(train_X)
    train_X = orig_scalar.transform(train_X)

    if args.model_type == 'dt':
        base_model = DecisionTreeClassifier(random_state=0, max_depth=args.max_depth, 
                                            min_samples_leaf=args.min_samples_leaf)
    else:
        base_model = XGBClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth_xgb, 
                                   learning_rate=args.learning_rate, reg_lambda = args.reg_lambda, random_state=0)
    base_model.fit(train_X, train_y)
    train_accuracy.append(base_model.score(train_X, train_y))
    test_accuracy.append(base_model.score(test_X, test_y))

    preds = base_model.predict(test_X[:args.num_lime])
    lime_base = explain_lime(base_model, train_X, test_X, args.num_lime)
    shap_base = explain_shap(base_model, preds, test_X, args.num_lime)
    lime_base_sign = np.sign(lime_base)
    shap_base_sign = np.sign(shap_base)
    lime_base_topk_3 = metrics.get_top_k(3, lime_base)
    shap_base_topk_3 = metrics.get_top_k(3, shap_base)
    lime_base_topk_5 = metrics.get_top_k(5, lime_base)
    shap_base_topk_5 = metrics.get_top_k(5, shap_base)

    all_met_lime3, all_met_lime5 = np.zeros((3, args.num_trials)), np.zeros((3, args.num_trials))
    all_met_shap3, all_met_shap5 = np.zeros((3, args.num_trials)), np.zeros((3, args.num_trials))

    #TODO something is broken, all numbers are zero except for lime and k=3
    for r in range(args.num_trials):
        train_X_pert = perturb_X(train_X, r+1, args.threshold)
        train_X_pert = orig_scalar.inverse_transform(train_X_pert)

        if args.model_type == 'dt':
            shifted_model = DecisionTreeClassifier(random_state=0, max_depth=args.max_depth, 
                                                   min_samples_leaf=args.min_samples_leaf)
        else:
            shifted_model = XGBClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth_xgb, 
                                   learning_rate=args.learning_rate, reg_lambda = args.reg_lambda, 
                                   random_state=0)
        shifted_model.fit(train_X_pert, train_y)
        train_accuracy.append(shifted_model.score(train_X_pert, train_y))
        test_accuracy.append(shifted_model.score(test_X, test_y))

        lime_shift = explain_lime(shifted_model, train_X_pert, test_X, args.num_lime)
        shap_shift = explain_shap(shifted_model, preds, test_X, args.num_lime)
        lime_shift_sign = np.sign(lime_shift)
        shap_shift_sign = np.sign(shap_shift)
        lime_shift_topk_3 = metrics.get_top_k(3, lime_shift)
        shap_shift_topk_3 = metrics.get_top_k(3, shap_shift)
        lime_shift_topk_5 = metrics.get_top_k(5, lime_shift)
        shap_shift_topk_5 = metrics.get_top_k(5, shap_shift)

        for idx, met in enumerate([metrics.top_k_sa, metrics.top_k_cdc, metrics.top_k_ssd]):
            all_met_lime3[idx, r] = met(3, lime_base_topk_3, lime_shift_topk_3, lime_base_sign, lime_shift_sign)
            all_met_lime5[idx, r] = met(5, lime_base_topk_5, lime_shift_topk_5, lime_base_sign, lime_shift_sign)
            all_met_shap3[idx, r] = met(3, shap_base_topk_3, shap_shift_topk_3, shap_base_sign, shap_shift_sign)
            all_met_shap5[idx, r] = met(5, shap_base_topk_5, shap_shift_topk_5, shap_base_sign, shap_shift_sign)
        
    np.save(argstring + "accuracy.npy", np.array([train_accuracy, test_accuracy]))
    save_metric_data(all_met_lime3, all_met_lime5, all_met_shap3, all_met_shap5, argstring, args.num_trials)


def main(args):
    # TODO data loading isn't working (probably because of lack of "orig" in whobin filename)
    # FIX by calling into existing data processing code
    if args.dataset_shift:
        args.file_base += "_orig"
    train, test = datasets.load_data(args.file_base, args.dataset, 0, 0, add_noise=False, label_col=args.label_col)
    train_X, train_y = train.data, train.labels
    test_X, test_y = test.data, test.labels

    argstring = args.output_dir + "/" + args.dataset
    if not args.dataset_shift:        
        argstring += "_t" + str(int(args.threshold*100))
    if args.model_type == 'dt':
        argstring += "_dt_depth" + str(args.max_depth)
    else:
        argstring += "_xgb_lambda" + str(args.reg_lambda)
    if args.dataset_shift:
        real_world_shift(args, train_X, train_y, test_X, test_y, argstring)
    else:
        synthetic_shift(args, train_X, train_y, test_X, test_y, argstring)

    return 1

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('file_base', type=str, help='file path of dataset through _train or _test')
    parser.add_argument('--dataset_shift', type=bool, default=False)
    parser.add_argument('--output_dir', type=str, default='.')
    parser.add_argument('--label_col', default='label', type=str)
    parser.add_argument('--max_depth',type=int, default=10)
    parser.add_argument('--reg_lambda', type=int, default=1, help="regularization parameter for XGBoost")
    parser.add_argument('--num_trials', type=int, default=10, help="number of random perturbations to average over for synthetic noise case")
    parser.add_argument('--model_type', type=str, default='dt', help="model type, either dt (decision tree) or xgb (xgboost)")

    parser.add_argument('--threshold', type=float, default = 0.0)
    parser.add_argument('--num_lime', type=int, default=732, help="number of test samples for which to compute lime/shap explanations")
    parser.add_argument('--num_lime_iters', type=int, default=5000, help="number of iterations for lime")
    args = parser.parse_args()

    assert args.model_type in ['dt', 'xgb'], "model_type must be either dt or xgb"

    # set additional arguments for XGBoost
    args.n_estimators = 5
    args.max_depth_xgb = 32
    args.learning_rate = 1

    # set additional args for decision tree
    args.min_samples_leaf = 5
    main(args)


