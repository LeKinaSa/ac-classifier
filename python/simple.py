
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

import numpy as np
import pandas as pd
import os

import data
from data import set_working_directory


def save_submission(competition, y):
    submission = competition[['loan_id']].copy()
    submission = submission.rename(columns={'loan_id': 'Id'})
    submission['Predicted'] = y.tolist()

    os.makedirs('../data/submissions/', exist_ok=True)
    submission.to_csv('../data/submissions/simple.csv', index=False)

def model_learning_and_classification(dev, competition, estimator, param_grid={}, smote=False):
    X, y = dev.loc[:, ~dev.columns.isin(['loan_id', 'status'])], dev.loc[:, 'status']

    cv = StratifiedKFold()

    if smote:
        estimator = Pipeline([
            ('sampling', SMOTE()),
            ('classification', estimator)
        ])
        for key in list(param_grid.keys()):
            param_grid[f'classification__{key}'] = param_grid[key]
            del param_grid[key]
    
    clf = GridSearchCV(estimator, param_grid=param_grid, scoring='roc_auc', cv=cv)

    clf.fit(X, y)

    auc = clf.best_score_
    best_params = clf.best_params_

    X = competition.loc[:, ~competition.columns.isin(['loan_id', 'status'])]
    y = np.round(clf.predict_proba(X)[:, -1], 5)

    return ((auc, best_params), (competition, y))

def model_learning_and_classification_without_grid_search(dev, competition, estimator, smote=False):
    dev = dev.sort_values(by='date_loan')
    dev         =         dev.drop('date_loan', axis=1)
    competition = competition.drop('date_loan', axis=1)

    X, y = dev.iloc[:, :-1], dev.iloc[:, -1]
    split_value = int(len(X) * 0.75)
    X_train, X_test = X.iloc[:split_value], X.iloc[split_value:]
    y_train, y_test = y.iloc[:split_value], y.iloc[split_value:]

    if smote:
        X_train, y_train = SMOTE().fit_resample(X_train, y_train)
    
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    competition = competition.sort_values(by='loan_id')
    
    X = competition.iloc[:, :-1]
    y = np.round(estimator.predict_proba(X)[:, -1], 5)
    return (auc, (competition, y))

def main():
    # Data
    dev, competition = data.get_loans_data()
    selected_columns = [
        'loan_id', 'status',
        #'gender_owner',
        #'duration',
        #'card',
        'balance_deviation',
        #'balance_distribution_median',
        'balance_distribution_first_quarter',
        'high_balance', 'last_neg',
    ]
    dev = data.select(dev, selected_columns)
    competition = data.select(competition, selected_columns)
    # correlation_analysis.correlation_analysis(dev, True, 1)
    
    # Classifiers
    classifiers = {
        'RFC' : (
            RandomForestClassifier(),
            {
                'n_estimators': [50, 100, 150],
                'criterion': ['gini', 'entropy'],
                'class_weight': ['balanced', 'balanced_subsample', None],
            }
        ),
        # 'ABC' : (
        #     AdaBoostClassifier(),
        #     {
        #         'algorithm': ['SAMME', 'SAMME.R'],
        #         'base_estimator': [
        #             #DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=10),
        #             RandomForestClassifier(class_weight='balanced_subsample', criterion='gini', n_estimators=100),
        #             #SVC(probability=True),
        #             #GradientBoostingClassifier(criterion='friedman_mse', loss='deviance', max_depth=3, max_features='sqrt'),
        #             LogisticRegression(),
        #         ],
        #         'n_estimators': [25, 50, 75],
        #     }
        # ),
        # 'LGR': (
        #     LogisticRegression(),
        #     {
        #         'solver': ['newton-cg', 'sag', 'lbfgs', 'liblinear'],
        #         'class_weight': [None, 'balanced'],
        #         'max_iter': [50, 100, 150, 250, 500],
        #     }
        # ),
    }

    best_results = (None, None)
    best_auc = 0
    best_classifier = None
    for smote in [False, True]:
        for classifier in classifiers:
            print(f'Classifier: {classifier}')
            (estimator, param_grid) = classifiers[classifier]
            (scores, results) = model_learning_and_classification(dev, competition, estimator, param_grid, smote)
            
            (auc, params) = scores
            print(f'AUC score: {auc}')
            print(f'Best params: {params}')
            print(f'Smote: {smote}')

            if best_auc < auc:
                best_auc = auc
                best_classifier = classifier
                best_results = results
                best_params = params
                best_smote = smote

    best_auc = round(best_auc, 5)
    print(f'\nBest classifier: {best_classifier}  (AUC: {best_auc}) - params {best_params}, smote {best_smote}')
    (competition, prediction) = best_results
    save_submission(competition, prediction)

def main_without_grid_search():
    # Data
    dev, competition = data.get_data_with_dates()
    selected_columns = [
        'loan_id', 'duration', 'gender_owner',
        'balance_deviation', 'balance_distribution_first_quarter',
        'card', 'high_balance', 'last_neg', 'status', 'date_loan',
    ]
    dev = data.select(dev, selected_columns)
    competition = data.select(competition, selected_columns)

    # Classifiers
    classifiers = {
        'RFC_G0_F': (RandomForestClassifier(), False),
        'RFC_G0_T': (RandomForestClassifier(), True),
        'RFC_GB_F': (RandomForestClassifier(class_weight='balanced'), False),
        'RFC_GB_T': (RandomForestClassifier(class_weight='balanced'), True),
        'RFC_GS_F': (RandomForestClassifier(class_weight='balanced_subsample'), False),
        'RFC_GS_T': (RandomForestClassifier(class_weight='balanced_subsample'), True),

        'RFC_E0_F': (RandomForestClassifier(criterion='entropy'), False),
        'RFC_E0_T': (RandomForestClassifier(criterion='entropy'), True),
        'RFC_EB_F': (RandomForestClassifier(criterion='entropy', class_weight='balanced'), False),
        'RFC_EB_T': (RandomForestClassifier(criterion='entropy', class_weight='balanced'), True),
        'RFC_ES_F': (RandomForestClassifier(criterion='entropy', class_weight='balanced_subsample'), False),
        'RFC_ES_T': (RandomForestClassifier(criterion='entropy', class_weight='balanced_subsample'), True),
    }

    best_auc = 0
    for classifier in classifiers:
        print(f'Classifier: {classifier}')
        (estimator, smote) = classifiers[classifier]
        (auc, results) = model_learning_and_classification_without_grid_search(dev, competition, estimator, smote)
        print(f'AUC score: {auc}')

        if best_auc < auc:
            best_auc = auc
            best_classifier = classifier
            best_results = results
        
    best_auc = round(best_auc, 5)
    print(f'\nBest classifier: {best_classifier}  (AUC: {best_auc})')
    (competition, prediction) = best_results
    save_submission(competition, prediction)

if __name__ == '__main__':
    set_working_directory()
    #main_without_grid_search()
    main()
