
from sklearn import tree
import numpy as np

import shap
import lime

import utils.parser_utils as parser_utils
import utils.exp_utils as exp_utils
import utils.data_utils as data_utils
import utils.datasets as datasets

'''
Generate trees and save their lime/shap explanations
'''
# Compute LIME explanations for one base model
def explain_lime(model, X, X_test, num_feat, num_samples=5000):
    explainer = lime.lime_tabular.LimeTabularExplainer(X, class_names=['0', '1'],
                                                       discretize_continuous=False)
    lime_exps = np.zeros((X_test.shape[0], X_test.shape[1]))
    for idx in range(X_test.shape[0]):
        exp = explainer.explain_instance(X_test[idx], model.predict_proba,
                                        num_features=num_feat, num_samples=num_samples)
        feature_idxs = [int(i[0]) for i in exp.as_list()]
        lime_values = [i[1] for i in exp.as_list()]
        lime_exps[idx, feature_idxs] = np.array(lime_values)
    return np.array(lime_exps)

def explain_shap(model, X_test):
    explainer = shap.Explainer(model)
    shap_test = explainer(X_test, check_additivity=False).values
    return shap_test

def main(args):
    dataset, run_id, output_dir = args.dataset, args.run_id, args.output_dir
    variations, base_repeats = args.variations, args.base_repeats
    dataset_shift = args.dataset_shift
    threshold = args.threshold

    # Load data
    random_states = exp_utils.get_random_states(dataset, dataset_shift, 0, variations, base_repeats)
    test_accs = np.zeros((len(args.max_depth), len(args.min_samples_leaf), len(random_states)))
    train_accs = np.zeros((len(args.max_depth), len(args.min_samples_leaf), len(random_states)))

    for i, r in enumerate(random_states):
        add_noise, baseline_model, _ = exp_utils.find_seed(r, dataset_shift, 0, base_repeats, variations)
        # we will use baseline_model for real-world dataset shift I think

        train, test = datasets.load_data(args.file_base, dataset, r, threshold, add_noise, args.label_col)

        for j, depth in enumerate(args.max_depth):
            for k, min_samples_leaf in enumerate(args.min_samples_leaf):
                model = tree.DecisionTreeClassifier(max_depth=depth,
                                        min_samples_leaf=min_samples_leaf)
                model = model.fit(train.data, train.labels.to_numpy()[:, 0])

                # Predict
                pred_te = model.predict(test.data)
                pred_tr = model.predict(train.data)

                # Accuracy
                test_accs[j,k,i] = (pred_te == test.labels.to_numpy()[:, 0]).mean()
                train_accs[j,k,i] = (pred_tr == train.labels.to_numpy()[:,0]).mean()

                # Compute LIME and SHAP on pred_te
                output_file = args.output_dir + "/" + args.run_id + "_d" + str(depth) + "_m" + str(min_samples_leaf) + "_" + str(r) + "_"
                np.save(output_file + "lime.npy", explain_lime(model, train.data, test.data, train.num_features(), args.n_lime_samples))
                np.save(output_file + "shap.npy", explain_shap(model, test.data))

        # save accuracy
        np.save(output_dir + "/accuracy_test_" + run_id + ".npy", test_accs)
        np.save(output_dir + "/accuracy_train_" + run_id + ".npy", train_accs) 

if __name__ == "__main__":
    parser = parser_utils.create_tree_parser()
    args = parser.parse_args()

    main(args) 