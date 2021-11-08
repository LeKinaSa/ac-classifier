
from data import get_loan_account_district_data

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, AdaBoostClassifier
from sklearn.svm import SVC

def save_submission(competition, y):
    submission = competition[['loan_id']].copy()
    submission = submission.rename(columns={'loan_id': 'Id'})
    submission['Predicted'] = y.tolist()

    os.makedirs('../data/submissions/', exist_ok=True)
    submission.to_csv('../data/submissions/simple.csv', index=False)

def model_learning_and_classification(estimator, param_grid={}):
    dev, competition = get_loan_account_district_data(remove_non_numeric=True)

    to_drop = ['account_id', 'district_id', 'code', 'date_x', 'date_y', 'payments']

    dev         =         dev.drop(to_drop, axis=1)
    competition = competition.drop(to_drop, axis=1)

    X, y = dev.loc[:, ~dev.columns.isin(['loan_id', 'status'])], dev.loc[:, 'status']

    cv = StratifiedKFold()
    clf = GridSearchCV(estimator, param_grid=param_grid, scoring='roc_auc', cv=cv)

    clf.fit(X, y)

    auc = clf.best_score_
    best_params = clf.best_params_

    X = competition.loc[:, ~competition.columns.isin(['loan_id', 'status'])]
    y = np.round(clf.predict_proba(X)[:, -1], 5)

    return ((auc, best_params), (competition, y))

def main():
    classifiers = {
        'KNC' : (
            KNeighborsClassifier(),
            {
                'n_neighbors': [5, 10, 20],
                'weights': ['uniform', 'distance'],
                'algorithm': ['ball_tree', 'kd_tree', 'brute'],
                'p': [1, 2, 3],
            }
        ),
        'SVC' : (
            SVC(),
            {
                'probability': [True]
            }
        ),
        # 'StC' : (
        #     StackingClassifier(
        #         estimators=[
        #             RandomForestClassifier(n_estimators=10, random_state=42),
        #             KNeighborsClassifier()
        #         ],
        #         final_estimator=LogisticRegression()
        #     ),
        #     {}
        # ),
        'DTC' : (
            DecisionTreeClassifier(),
            {
                'criterion': ['gini', 'entropy'],
                'max_depth': [3, 5, 10, None],
                'class_weight': [None, 'balanced']
            }
        ),
        'ABC' : (
            AdaBoostClassifier(),
            {}
        )
    }

    best_results = (None, None)
    best_auc = 0
    for classifier in classifiers:
        print(f'Classifier: {classifier}')
        (estimator, param_grid) = classifiers[classifier]
        (scores, results) = model_learning_and_classification(estimator, param_grid)
        
        (auc, best_params) = scores
        print(f'AUC score: {auc}')
        print(f'Best params: {best_params}')

        if best_auc < auc:
            best_results = results
        
    (competition, prediction) = best_results
    save_submission(competition, prediction)

if __name__ == '__main__':
    main()
