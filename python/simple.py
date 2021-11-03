
from data import get_loan_data

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

def save_submission(X, y):
    submission = X[['loan_id']].copy()
    submission = submission.rename(columns={'loan_id': 'Id'})
    submission['Predicted'] = y.tolist()

    os.makedirs('../data/submissions/', exist_ok=True)
    submission.to_csv('../data/submissions/simple.csv', index=False)

def main():
    train, test = get_loan_data()

    X, y = train.iloc[:, :-1], train.iloc[:, -1]

    estimator = GaussianNB()
    cv = StratifiedKFold()
    clf = GridSearchCV(estimator, param_grid={}, scoring='roc_auc', cv=cv)

    clf.fit(X, y)

    auc = clf.best_score_
    print(f'AUC score: {auc}')

    test = test.sort_values(by='loan_id')

    X = test.iloc[:, :-1]
    y = np.round(clf.predict_proba(X)[:, -1], 5)

    save_submission(X, y)

if __name__ == "__main__":
    main()
