import timeit

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneOut
from xgboost import XGBClassifier, plot_importance
from sklearn import metrics


def prepare_y(results_df, score, higher_is_better):
    # mean all k folds of each dataset + algo and take the relevant score
    folds_mean = (results_df.groupby((results_df['Algorithm Name'] != results_df['Algorithm Name'].shift()).cumsum()))\
        .mean().reset_index(drop=True)[score]

    # get models names
    algorithm_names = results_df['Algorithm Name'].drop_duplicates().values

    # set the winner for each dataset
    winner_data = []
    for dataset_index in range(0, folds_mean.shape[0] - len(algorithm_names) + 1, len(algorithm_names)):
        winner_score = folds_mean.values[dataset_index]
        winner_algorithm = 0
        for algorithm_index in range(1, len(algorithm_names)):
            algorithm_score = folds_mean.values[dataset_index + algorithm_index]
            if (algorithm_score > winner_score and higher_is_better) or \
                    (algorithm_score < winner_score and not higher_is_better):
                winner_score = algorithm_score
                winner_algorithm = algorithm_index
        winner_data.append(winner_algorithm)

    return np.array(winner_data), algorithm_names


def get_datasets_names(results_df):
    return results_df['Dataset Name'].drop_duplicates().values


def load_meta_features():
    df = pd.read_csv('./ClassificationAllMetaFeatures.csv')
    X = df.iloc[:, :-1].values
    return X


def preprocess_X(X):
    X = pd.get_dummies(pd.DataFrame(X).infer_objects()).values
    X = SimpleImputer(strategy='mean').fit_transform(X)
    return X


def train(X_train, y_train):
    xgb_model = XGBClassifier(learning_rate=0.05, n_estimators=100, max_depth=5)
    start_time = timeit.default_timer()
    xgb_model.fit(X_train, y_train)
    training_sec = timeit.default_timer() - start_time
    return xgb_model, training_sec


def predict(model, X_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    return y_pred, y_proba


def fpr_macro(y_test, y_pred, y_prob):
    fprs = dict()
    for i in range(y_prob.shape[1]):
        y_true = np.zeros(y_test.size)
        y_predict = np.zeros(y_test.size)
        y_true[y_test == i] = 1
        y_predict[y_pred == i] = 1
        con_mat_scores = metrics.confusion_matrix(y_true, y_predict).ravel()
        if len(con_mat_scores) == 1:
            if y_true[0] == 0:
                tn, fp, fn, tp = con_mat_scores[0], 0, 0, 0
            else:
                tn, fp, fn, tp = 0, 0, 0, con_mat_scores[0]
        else:
            tn, fp, fn, tp = con_mat_scores
        if (fp + tn) == 0:
            fprs[i] = 0
        else:
            fprs[i] = fp / (fp + tn)
    return np.mean(list(fprs.values()))


def PR_curve_auc_macro(y_test, y_prob):
    precision = dict()
    recall = dict()
    for i in range(y_prob.shape[1]):
        y_true = np.zeros(y_test.size)
        y_true[y_test == i] = 1
        precision[i], recall[i], _ = metrics.precision_recall_curve(y_true, y_prob[:, i])
    aucs = [metrics.auc(recall, precision) for recall, precision in zip(recall.values(), precision.values())]
    return np.mean(aucs)


def roc_auc_macro(y_test, y_prob):
    tprs = dict()
    fprs = dict()
    for i in range(y_prob.shape[1]):
        y_true = np.zeros(y_test.size)
        y_true[y_test == i] = 1
        fprs[i], tprs[i], _ = metrics.roc_curve(y_true, y_prob[:, i])
    aucs = [metrics.auc(fpr, tpr) for fpr, tpr in zip(fprs.values(), tprs.values())]
    return np.mean(aucs)


def calculate_scores(y_test, y_pred, y_proba):
    acc = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred, average='macro', labels=np.unique(y_test), zero_division=0)
    tpr = recall = metrics.recall_score(y_test, y_pred, average='macro', labels=np.unique(y_test), zero_division=0)
    fpr = fpr_macro(y_test, y_pred, y_proba)
    pr_curve = PR_curve_auc_macro(y_test, y_proba)
    auc = roc_auc_macro(y_test, y_proba)
    return acc, tpr, fpr, precision, auc, pr_curve


results_df = pd.read_csv('./results_T_completed.csv')
y, algorithm_names = prepare_y(results_df, score='Accuracy', higher_is_better=True)
datasets_names = get_datasets_names(results_df)
X = load_meta_features()
X = preprocess_X(X)
loo = LeaveOneOut()

import warnings

warnings.filterwarnings("ignore")
columns = ['Dataset Name', 'Accuracy', 'TPR_macro', 'FPR_macro', 'Precision_macro', 'AUC_macro', 'PR-Curve_macro',
           'Training Time (sec)', 'Predicted Model', 'Predicted Model Probability']

results = []
_total_y_test = []
_total_y_pred = []
_total_y_proba = []
for train_index, test_index in loo.split(X):
    print('Train index:', train_index, 'Test index:', test_index)
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]

    model, training_sec = train(X_train, y_train)
    y_pred, y_proba = predict(model, X_test)

    scores = calculate_scores(y_test, y_pred, y_proba)
    acc, tpr, fpr, precision, auc, pr_curve = scores

    predicted_model = algorithm_names[y_pred][0]
    predicted_model_proba = float(y_proba[0][y_pred][0])
    results.append([datasets_names[test_index][0], acc, tpr, fpr, precision, auc, pr_curve, training_sec,
                    predicted_model, predicted_model_proba])

    # total
    if y_proba.shape == (1, len(algorithm_names)):
        _total_y_test.append(y_test[0])
        _total_y_pred.append(y_pred[0])
        _total_y_proba.append(y_proba[0])

# scores on all
scores = calculate_scores(np.array(_total_y_test), np.array(_total_y_pred), np.array(_total_y_proba))
acc, tpr, fpr, precision, auc, pr_curve = scores
print('Accuracy', 'TPR_macro', 'FPR_macro', 'Precision_macro', 'AUC_macro', 'PR-Curve_macro', sep='\t')
print(acc, tpr, fpr, precision, auc, pr_curve, sep='\t')

# save
xgb_results_df = pd.DataFrame(results, columns=columns)
result_file = 'xgb_results_T.csv'
xgb_results_df.to_csv(result_file, index=False)

# train all
print('Trainig based on all')
model, _ = train(X, y)
ax = plot_importance(model, title='Weight', importance_type='weight', max_num_features=10)
plt.show()
ax = plot_importance(model, title='Gain', importance_type='gain', max_num_features=10)
plt.show()
ax = plot_importance(model, title='Cover', importance_type='cover', max_num_features=10)
plt.show()

# fix tree
booster = model.get_booster()
model_bytearray = booster.save_raw()[4:]
def fix(self=None):
    return model_bytearray
booster.save_raw = fix

# shap
import shap
explainer = shap.TreeExplainer(booster)
shap_values = explainer.shap_values(X)
shap.initjs()
for i in range(4):
    shap.save_html('shap_' + str(i) + '.html', shap.force_plot(explainer.expected_value[i], shap_values[i], X))
shap.summary_plot(shap_values, X)
