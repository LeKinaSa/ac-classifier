
from data import *

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier

def save_submission(competition, y):
    submission = competition[['loan_id']].copy()
    submission = submission.rename(columns={'loan_id': 'Id'})
    submission['Predicted'] = y.tolist()

    os.makedirs('../data/submissions/', exist_ok=True)
    submission.to_csv('../data/submissions/simple.csv', index=False)

def main():
    dev, competition = get_loan_account_district_data(remove_non_numeric=True)

    to_drop = ['account_id', 'district_id', 'code', 'date_x', 'date_y', 'payments']

    dev         =         dev.drop(to_drop, axis=1)
    competition = competition.drop(to_drop, axis=1)

    X, y = dev.loc[:, ~dev.columns.isin(['loan_id', 'status'])], dev.loc[:, 'status']

    # estimators = [
    #     ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
    #     ('knc', KNeighborsClassifier())
    # ]
    # estimator = StackingClassifier(
    #     estimators=estimators, final_estimator=LogisticRegression()
    # )
    
    estimator = KNeighborsClassifier()
    param_grid = {
        'n_neighbors': [5, 10, 20],
        'weights': ['uniform', 'distance'],
        'algorithm': ['ball_tree', 'kd_tree', 'brute'],
        'p': [1, 2, 3],
    }

    # DecisionTreeClassifier param_grid
    # param_grid = {
    #     'criterion': ['gini', 'entropy'],
    #     'max_depth': [3, 5, 10, None],
    #     'class_weight': [None, 'balanced']
    # }

    cv = StratifiedKFold()
    clf = GridSearchCV(estimator, param_grid=param_grid, scoring='roc_auc', cv=cv)

    clf.fit(X, y)

    auc = clf.best_score_
    print(f'AUC score: {auc}')

    best_params = clf.best_params_
    print(f'Best params: {best_params}')

    X = competition.loc[:, ~competition.columns.isin(['loan_id', 'status'])]
    y = np.round(clf.predict_proba(X)[:, -1], 5)

    save_submission(competition, y)

if __name__ == '__main__':
    main()
