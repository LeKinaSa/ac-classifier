
import pandas as pd

def get_loan_data():
    train = pd.read_csv('../data/loan_train.csv', sep=';')
    test = pd.read_csv('../data/loan_test.csv', sep=';')

    train['status'] = train['status'].apply(lambda x: 1 if x == -1 else 0)

    return train, test