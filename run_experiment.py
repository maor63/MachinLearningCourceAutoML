# %%

import pandas as pd
import numpy as np
from sklearn.linear_model import *
from sklearn.ensemble import *
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import *
from sklearn import metrics
from hyperopt import tpe, hp, fmin
import os
from pathlib import Path
from sklearn.model_selection import *
from sklearn.preprocessing import *
import timeit
import lightgbm as lgb
from palobst.palobst import PaloBst
from KiGB.core.scikit.skigb import SKiGB as KiGB
import warnings
from NNDT.NNDT_classifier import DNDT
from PRF import prf as ProbabilisticRandomForest
import gbdtpl

warnings.filterwarnings('ignore', category=FutureWarning)
# %% md

# Define sklearn models

# %%

from sklearn.base import BaseEstimator, ClassifierMixin


# class PaloBstSKlearn(BaseEstimator, PaloBst):
#     def __init__(self, **kwargs):
#         BaseEstimator.__init__(self, **kwargs)
#         PaloBst.__init__(self, **kwargs)
#
#
# PaloBstSKlearn().get_params()

def preprocess_base(X, y):
    X = pd.get_dummies(pd.DataFrame(X)).values
    return X, y


space = {
    'max_depth': hp.randint('max_depth', 2, 8),
    'learning_rate': hp.uniform('learning_rate', 0.1, 0.6),
    'n_estimators': hp.randint('n_estimators', 20, 200),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 10),
    'subsample': hp.uniform('subsample', 0.5, 1.0),
    'num_leaves': 37
}
baseline_model = (lgb.LGBMClassifier, space, preprocess_base, True)

space = {
    'learning_rate': hp.uniform('learning_rate', 0.1, 0.6),
    'n_estimators': hp.randint('n_estimators', 20, 200),
    'subsample': hp.uniform('subsample', 0.5, 1.0),
    'max_depth': hp.randint('max_depth', 2, 8),
    'distribution': "bernoulli",
}
model1 = (PaloBst, space, preprocess_base, False)

space = {
    'lr': hp.uniform('lr', 0.0001, 0.05),
    'cut_count': hp.randint('cut_count', 4, 10),
    'temprature': hp.uniform('temprature', 0.01, 0.8),
    'epochs': hp.randint('epochs', 5, 50),
}
model2 = (DNDT, space, preprocess_base, True)

space = {
    'keep_proba': hp.uniform('keep_proba', 0.001, 0.1),
    'n_estimators_': hp.randint('n_estimators_', 10, 80),
    'max_depth': hp.randint('max_depth', 3, 8),
}
model3 = (ProbabilisticRandomForest, space, preprocess_base, True)

models = [baseline_model, model1, model2, model3]
# models = [model2]

# %% md

# Load Datasets

# %%

folder_path = Path('')
dataset_path = folder_path / 'classification_datasets/'
print('number of datasets:', len(os.listdir(dataset_path)))


def optimize_parameters(model, space, X_train, y_train, multiclass=True):
    def objective(params):
        params = {param: params[param] for param in space}
        if isinstance(model(), KiGB):
            data = pd.DataFrame(np.append(X_train, y_train.reshape(-1,1), axis=1))
            class_corr_coef = list(data.corr().iloc[-1])
            params['advice'] = np.array(class_corr_coef[:-1])
        if multiclass:
            gbm_clf = model(**params)
        else:
            gbm_clf = OneVsRestClassifier(model(**params), n_jobs=3)

        best_score = cross_val_score(gbm_clf, X_train, y_train, scoring='f1_macro', cv=3, n_jobs=3).mean()
        loss = 1 - best_score
        return loss

    # Run the algorithm
    best = fmin(fn=objective, space=space, max_evals=50, rstate=np.random.RandomState(42), algo=tpe.suggest,
                verbose=False)
    for param in space:
        if param not in best:
            best[param] = space[param]
    return best


# %%

def load_dataset(path):
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].values
    y = LabelEncoder().fit_transform(df.iloc[:, -1].values)
    y = y.ravel()
    return X, y


def PR_curve_auc_macro(y_true, y_prob):
    precision = dict()
    recall = dict()
    for i in range(y_prob.shape[1]):
        y_true = np.zeros(y_test.size)
        y_true[y_test == i] = 1
        precision[i], recall[i], _ = metrics.precision_recall_curve(y_true, y_prob[:, i])
    aucs = [metrics.auc(recall, precision) for recall, precision in zip(recall.values(), precision.values())]
    return np.mean(aucs)


def roc_auc_macro(y_true, y_prob):
    tprs = dict()
    fprs = dict()
    for i in range(y_prob.shape[1]):
        y_true = np.zeros(y_test.size)
        y_true[y_test == i] = 1
        fprs[i], tprs[i], _ = metrics.roc_curve(y_true, y_prob[:, i])
    aucs = [metrics.auc(fpr, tpr) for fpr, tpr in zip(fprs.values(), tprs.values())]
    return np.mean(aucs)


def fpr_macro(y_true, y_pred, y_prob):
    fprs = dict()
    for i in range(y_prob.shape[1]):
        y_true = np.zeros(y_test.size)
        y_predict = np.zeros(y_test.size)
        y_true[y_test == i] = 1
        y_predict[y_pred == i] = 1
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_predict).ravel()
        fprs[i] = fp / (fp + tn)
    return np.mean(list(fprs.values()))


def calculate_scores(y_test, y_pred, y_proba):
    acc = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred, average='macro', labels=np.unique(y_test), zero_division=0)
    tpr = recall = metrics.recall_score(y_test, y_pred, average='macro', labels=np.unique(y_test), zero_division=0)
    fpr = fpr_macro(y_test, y_pred, y_proba)
    pr_curve = PR_curve_auc_macro(y_test, y_proba)
    auc = roc_auc_macro(y_test, y_proba)
    return acc, tpr, fpr, precision, auc, pr_curve


def train_classifier(model, best_params, X_train, y_train, multiclass):
    if isinstance(model(), KiGB):
        data = pd.DataFrame(np.append(X_train, y_train.reshape(-1, 1), axis=1))
        class_corr_coef = list(data.corr().iloc[-1])
        best_params['advice'] = np.array(class_corr_coef[:-1])

    if multiclass:
        clf = model(**best_params)
    else:
        clf = OneVsRestClassifier(model(**best_params), n_jobs=2)

    tarin_start_time = timeit.default_timer()
    print(best_params)
    clf.fit(X_train, y_train)
    training_sec = timeit.default_timer() - tarin_start_time
    return clf, training_sec


def predict_using_classifier(clf, X_test):
    infrence_start_time = timeit.default_timer()
    clf.predict(X_test[:1000])
    infrence_1000_sec = timeit.default_timer() - infrence_start_time

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    return y_pred, y_proba, infrence_1000_sec


import warnings

warnings.filterwarnings("ignore")
columns = ['Dataset Name', 'Algorithm Name', 'Cross Validation [1-10]', 'Hyper-Parameters Values',
           'Accuracy', 'TPR_macro', 'FPR_macro', 'Precision_macro', 'AUC_macro', 'PR-Curve_macro',
           'Training Time (sec)', 'Inference Time (sec)']
results = []
false_datasets = []
total_datasets = len(os.listdir(dataset_path))
for i, dataset_csv in enumerate(os.listdir(dataset_path), 1):
    if i > 1:
        break
    X, y = load_dataset(dataset_path / dataset_csv)
    dataset_name = dataset_csv.split('.csv')[0]
    model_count = len(models)
    for model_idx, (model, space, preprocess, multiclass) in enumerate(models, 1):
        algorithm_name = model().__class__.__name__
        X, y = preprocess(X, y)
        kfold = StratifiedKFold(n_splits=10)
        for fold, (train_index, test_index) in enumerate(kfold.split(X, y), 1):
            print(
                f'\r run on dataset {i}/{total_datasets}, Method: {algorithm_name}, {model_idx}/{model_count} fold: {fold}/{10}',
                end='')
            X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]

            best_params = optimize_parameters(model, space, X_train, y_train, multiclass)
            clf, training_sec = train_classifier(model, best_params, X_train, y_train, multiclass)

            y_pred, y_proba, infrence_1000_sec = predict_using_classifier(clf, X_test)

            scores = calculate_scores(y_test, y_pred, y_proba)
            acc, tpr, fpr, precision, auc, pr_curve = scores

            results.append([dataset_name, algorithm_name, fold, best_params,
                            acc, tpr, fpr, precision, auc, pr_curve,
                            training_sec, infrence_1000_sec])

results_df = pd.DataFrame(results, columns=columns)
results_df.to_csv(folder_path / 'results.csv', index=False)

