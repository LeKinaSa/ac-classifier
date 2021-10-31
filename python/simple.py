
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def main():
    train = pd.read_csv('../data/loan_train.csv', sep=';')
    test = pd.read_csv('../data/loan_test.csv', sep=';')

    train['status'] = train['status'].apply(lambda x: 1 if x == -1 else 0)

    X, y = train.iloc[:, :-1], train.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    dtc = DecisionTreeClassifier(random_state=0, max_depth=3)

    dtc.fit(X_train, y_train)
    y_pred = dtc.predict_proba(X_test)[:, 1]

    accuracy = dtc.score(X_test, y_test)
    print(f'Accuracy (%): {round(accuracy * 100, 3)}')

    auc = roc_auc_score(y_test, y_pred)
    print(f'AUC score: {auc}')

    test = test.sort_values(by='loan_id')

    X = test.iloc[:, :-1]
    y = dtc.predict(X)

    submission = X[['loan_id']].copy()
    submission = submission.rename(columns={'loan_id': 'Id'})
    submission['Predicted'] = y.tolist()

    submission.to_csv('../data/submissions/simple.csv', index=False)

if __name__ == "__main__":
    main()
