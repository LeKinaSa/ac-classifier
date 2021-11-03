
import pandas as pd

def get_loan_data():
    train = pd.read_csv('../data/loan_train.csv', sep=';')
    test = pd.read_csv('../data/loan_test.csv', sep=';')

    train['status'] = train['status'].apply(lambda x: 1 if x == -1 else 0)

    return train, test

def get_loan_account_district_data():
    loan_train = pd.read_csv('../data/loan_train.csv', sep=';')
    loan_test  = pd.read_csv('../data/loan_test.csv',  sep=';')

    loan_train['status'] = loan_train['status'].apply(lambda x: 1 if x == -1 else 0)

    account = pd.read_csv('../data/account.csv', sep=';')

    district = pd.read_csv('../data/district.csv', sep=';')

    account_district = pd.merge(left=account, right=district, left_on='district_id', right_on='code ')

    train = pd.merge(left=loan_train, right=account_district, left_on='account_id', right_on='account_id')
    test  = pd.merge(left=loan_test , right=account_district, left_on='account_id', right_on='account_id')

    return train, test

if __name__ == "__main__":
    print(get_loan_account_district_data()[0].head(5))
    print(get_loan_account_district_data()[0].columns)