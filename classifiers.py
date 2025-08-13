import os
import arff
import torch

from tqdm import tqdm
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import balanced_accuracy_score, make_scorer

from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.dummy import DummyClassifier

from tabpfn import TabPFNClassifier
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore", message=".*The max_iter was reached which means the coef_ did not converge*")
warnings.filterwarnings("ignore", message=".*lbfgs failed to converge (status=1)*")
warnings.filterwarnings("ignore", message=".*Running on CPU with more than 200 samples may be slow.*")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Liblinear failed to converge, increase the number of iterations.*")


def calculate_bac_cv(classificador, X, y, n_splits=10):
    min_classe = pd.Series(y).value_counts().min()

    cv = StratifiedKFold(n_splits=min(n_splits, min_classe), shuffle=True, random_state=42)
    scorer = make_scorer(balanced_accuracy_score)

    scores = cross_val_score(classificador, X, y, cv=cv, scoring=scorer)

    mean_bac = scores.mean()
    std_bac = scores.std()

    return mean_bac, std_bac

def knn_classifier(k, weights):
    return KNeighborsClassifier(n_neighbors=k, weights = weights)

def nn_model_averaging(activation, runs=5):
    return MLPClassifier(early_stopping=True, random_state=runs, hidden_layer_sizes=100, activation=activation)

def adaboost_classifier(n_estimators):
    base = DecisionTreeClassifier(random_state=42)
    return AdaBoostClassifier(estimator=base, random_state=42, n_estimators=n_estimators)

def decisiontree_classifier(criterion, depth, splitter):
    return DecisionTreeClassifier(random_state=42,  criterion=criterion, max_depth = depth, splitter = splitter)

def j48_classifier(depth):
    return DecisionTreeClassifier(random_state=42, criterion='entropy', max_depth = depth, min_samples_split=10, min_samples_leaf=5, splitter = 'best')

def extratree_classifier(criterion, depth):
    return ExtraTreesClassifier(random_state=42, criterion=criterion, max_depth = depth)

def dummy_classifier(strategy):
    return DummyClassifier(random_state=42, strategy=strategy)

def gbm_classifier(n_estimators, depth):
    return GradientBoostingClassifier(random_state=42, n_estimators=n_estimators, max_depth = depth)

def rf_classifier(n_estimators, depth):
    return RandomForestClassifier(random_state=42, n_estimators=n_estimators, max_depth = depth)

def xgb_classifier(n_estimators, depth):
    return XGBClassifier(random_state=42, n_estimators=n_estimators, max_depth = depth)

def glmnet_classifier(penalty, C):
    if (penalty == 'elasticnet'):
        l1_ratio = 0.5
    else:
        l1_ratio = None
    return LogisticRegression(solver='saga',random_state=42, penalty=penalty, C=C, l1_ratio=l1_ratio)

def multinom_classifier(C):
    return LogisticRegression(random_state=42, C=C)

def svm_classifier(C, kernel, degree):
    return SVC(random_state=42, C=C, kernel=kernel, degree=degree)

def svm_linear(C):
    return LinearSVC(random_state=42, C=C)

def nb_gauss_classifier(**kwargs):
    return GaussianNB(**kwargs)

def lda_classifier(tol):
    return LinearDiscriminantAnalysis(tol=tol)

def load_arff_dataset(path_file_arff):
    with open(path_file_arff, 'r') as f:
        dataset = arff.load(f)
    df = pd.DataFrame(dataset['data'], columns=[att[0] for att in dataset['attributes']])
    return df

def preprocess(df, col_class):
    X = df.drop(columns=[col_class])
    y = df[col_class]

    for col in X.columns:
        if X[col].dtype == object:
            X[col] = LabelEncoder().fit_transform(X[col])
    if y.dtype == object:
        y = LabelEncoder().fit_transform(y)

    return X, y

def apply_classifier(arff_file_path, classifiers_params, source_label):

    df = load_arff_dataset(arff_file_path)

    nome_arquivo = os.path.basename(arff_file_path)
    dataset_name = os.path.splitext(nome_arquivo)[0]
    col_class = df.columns[-1]

    X, y = preprocess(df, col_class)

    results = []

    for clf_name, params_list in classifiers_params.items():

        for params in params_list:

            if clf_name == 'IBk_knn':
                clf = knn_classifier(**params)

            elif clf_name == 'nn':

                min_classe = pd.Series(y).value_counts().min()

                scores = []

                def calculate_bac_nn():
                    for seed in range(5):
                        clf = nn_model_averaging(runs=seed, **params)
                        score = calculate_bac_cv(clf, X, y)
                        scores.append(score[0])

                    mean_bac = np.mean(scores)
                    std_bac = np.std(scores)
                    return mean_bac, std_bac

                mean_bac, std_bac = calculate_bac_nn()
                results.append((source_label, dataset_name, clf_name, params, round(mean_bac,4), round(std_bac,4)))
                continue

            elif clf_name == 'avnn':

                min_classe = pd.Series(y).value_counts().min()

                cv = StratifiedKFold(n_splits=min(10, min_classe), shuffle=True, random_state=42)
                bac_scores = []

                if not isinstance(X, pd.DataFrame):
                    X = pd.DataFrame(X)

                if not isinstance(y, pd.Series):
                    y = pd.Series(y)

                def calculate_bac_avnn():
                        for train_idx, test_idx in cv.split(X, y):
                            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                            preds_prob = []
                            for seed in range(5):
                            	clf = nn_model_averaging(runs=seed, **params)
                            	clf.fit(X_train, y_train)
                            	probs = clf.predict_proba(X_test)
                            	preds_prob.append(probs)

                            avg_probs = np.mean(preds_prob, axis=0)
                            y_pred = np.argmax(avg_probs, axis=1)

                            bac = balanced_accuracy_score(y_test, y_pred)
                            bac_scores.append(bac)

                        mean_bac = np.mean(bac_scores)
                        dev_bac = np.std(bac_scores)
                        return mean_bac, dev_bac

                mean_bac, std_bac = calculate_bac_avnn()
                results.append((source_label, dataset_name, clf_name, params, round(mean_bac,4), round(std_bac,4)))
                continue

            elif clf_name == 'tabpfn':

                if not isinstance(X, pd.DataFrame):
                    X = pd.DataFrame(X)

                if not isinstance(y, pd.Series):
                    y = pd.Series(y)

                min_classe = y.value_counts().min()

                cv = StratifiedKFold(n_splits=min(10, min_classe), shuffle=True, random_state=42)
                bac_scores = []

                def tabpfn_classifier():
                    for train_idx, test_idx in cv.split(X, y):
                        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                        device = "cuda" if torch.cuda.is_available() else "cpu"

                        clf = TabPFNClassifier(device=device)
                        clf.fit(X_train, y_train)
                        y_pred = clf.predict(X_test)

                        bac = balanced_accuracy_score(y_test, y_pred)
                        bac_scores.append(bac)

                    mean_bac = np.mean(bac_scores)
                    std_bac = np.std(bac_scores)
                    return mean_bac, std_bac

                mean_bac, std_bac = tabpfn_classifier()
                results.append((source_label, dataset_name, clf_name, params, round(mean_bac,4), round(std_bac,4)))
                continue

            elif clf_name == 'adaboost':
                clf = adaboost_classifier(**params)

            elif clf_name == 'decisiontree':
                clf = decisiontree_classifier(**params)

            elif clf_name == 'extratree':
                clf = extratree_classifier(**params)

            elif clf_name == 'dummy':
                clf = dummy_classifier(**params)

            elif clf_name == 'gbm':
                clf = gbm_classifier(**params)
            elif clf_name == 'glmnet':
                clf = glmnet_classifier(**params)
            elif clf_name == 'j48':
                clf = j48_classifier(**params)
            elif clf_name == 'svm':
                clf = svm_classifier(**params)
            elif clf_name == 'svm_linear':
                clf = svm_linear(**params)
            elif clf_name == 'multinom':
                clf = multinom_classifier(**params)
            elif clf_name == 'rf':
                clf = rf_classifier(**params)
            elif clf_name == 'nb_gaus':
                clf = nb_gauss_classifier(**params)
            elif clf_name == 'xgb':
                clf = xgb_classifier(**params)
            elif clf_name == 'lda':
                clf = lda_classifier(**params)

            mean_bac, std_bac = calculate_bac_cv(clf, X, y)
            results.append((source_label, dataset_name, clf_name, params, round(mean_bac,4), round(std_bac,4)))

    return results

if __name__ == '__main__':
    directories = [
        ('./data/OpenML/', 'OpenML'),
        ('./data/UCI/', 'UCI'),
        ('./data/LC-ICPR/', 'LC-ICPR')
    ]
    #Parameters of the classifiers to be used
    ks = [1,5,10]
    weights = ['uniform', 'distance']
    estimators = [50,100,1000]
    criterion =['gini', 'entropy']
    depth = [None, 3, 50]
    split = ['best', 'random']
    strategies = ['most_frequent', 'uniform']
    penalties = ['l1', 'l2', 'elasticnet']
    Cs = [0.1, 1.0, 1000]
    kernels = ['poly', 'rbf']
    degrees = [3,5]
    tols = [0.0001, 0.001, 0.01]
    activations = ['relu', 'logistic']

    classifiers_params = {
            'IBk_knn': [{'k':k, 'weights':w} for k in ks for w in weights],
            'adaboost': [{'n_estimators':v_estimator} for v_estimator in estimators],
            'decisiontree': [{'criterion':cri , 'depth': dep, 'splitter': spl} for cri in criterion for dep in depth for spl in split],
            'extratree': [{'criterion':cri , 'depth': dep} for cri in criterion for dep in depth],
            'dummy':[{'strategy':strat} for strat in strategies],
            'j48': [{'depth': dep} for dep in depth],
            'lda':[{'tol': v_tol} for v_tol in tols],
            'nb_gaus': [{}],
            'svm_linear':[{'C': v_C} for v_C in Cs],
            'svm': [{'C': v_C, 'kernel': ker, 'degree': deg} for v_C in Cs for ker in kernels for deg in degrees],
            'multinom': [{'C': v_C} for v_C in Cs],
            'glmnet': [{'penalty':pen , 'C': v_C} for pen in penalties for v_C in Cs],
            'xgb': [{'n_estimators':v_estimator , 'depth': dep} for v_estimator in estimators for dep in depth],
            'gbm': [{'n_estimators':v_estimator , 'depth': dep} for v_estimator in estimators for dep in depth],
            'rf': [{'n_estimators':v_estimator , 'depth': dep} for v_estimator in estimators for dep in depth],
            'tabpfn':[{}],
            'nn': [{'activation':act} for act in activations],
            'avnn': [{'activation':act} for act in activations]
    }

    result_values = []

    for folder, source in directories:
        files = [f for f in os.listdir(folder) if f.endswith('.arff')]
        #for file in files:
        for file in tqdm(files, desc=f"{source} processing"):

        	path_file = os.path.join(folder, file)

        	result_values.append(apply_classifier(arff_file_path=path_file,classifiers_params=classifiers_params, source_label=source))

    print ('Source, Dataset, Classifier, Classifier_param, BAC, BAC_std')
    for result_dataset in result_values:
        for result in result_dataset:
            source, dataset, clf, param, bac, std = result
            print(f"{source}, {dataset}, {clf}, {param}, {float(bac):.4f}, {float(std):.4f}")
