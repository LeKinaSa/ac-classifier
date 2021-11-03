
from data import *

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier

def save_submission(X, y):
    submission = X[['loan_id']].copy()
    submission = submission.rename(columns={'loan_id': 'Id'})
    submission['Predicted'] = y.tolist()

    os.makedirs('../data/submissions/', exist_ok=True)
    submission.to_csv('../data/submissions/simple.csv', index=False)

def main():
    dev, competition = get_loan_account_district_data(remove_non_numeric=True)

    # print(dev.columns)
    dev         =         dev.drop(['account_id', 'district_id'], axis=1)
    competition = competition.drop(['account_id', 'district_id'], axis=1)

    X, y = dev.loc[:, dev.columns != 'status'], dev.loc[:, 'status']

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
        'algorithm': ['ball_tree', 'kd_tree', 'brute']
    }

    cv = StratifiedKFold()
    clf = GridSearchCV(estimator, param_grid=param_grid, scoring='roc_auc', cv=cv)

    clf.fit(X, y)

    auc = clf.best_score_
    print(f'AUC score: {auc}')

    competition = competition.sort_values(by='loan_id')

    X = competition.loc[:, competition.columns != 'status']
    y = np.round(clf.predict_proba(X)[:, -1], 5)

    save_submission(X, y)

if __name__ == '__main__':
    main()
