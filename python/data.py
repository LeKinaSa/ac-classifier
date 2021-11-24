
import pandas as pd
import os.path

def get_loan_data():
    dev = pd.read_csv('../data/loan_train.csv', sep=';')
    competition = pd.read_csv('../data/loan_test.csv', sep=';')

    dev['status'] = dev['status'].apply(lambda x: 1 if x == -1 else 0)

    return dev, competition

def clean_district_data(df):
    df = df.rename(columns={
        'code ': 'code',
        'name ': 'name',
        'no. of inhabitants': 'population',
        'no. of municipalities with inhabitants < 499 ': 'muni_under499',
        'no. of municipalities with inhabitants 500-1999': 'muni_500_1999',
        'no. of municipalities with inhabitants 2000-9999 ': 'muni_2000_9999',
        'no. of municipalities with inhabitants >10000 ' : 'muni_over10000',
        'no. of cities ' : 'n_cities',
        'ratio of urban inhabitants ': 'ratio_urban',
        'average salary ': 'avg_salary',
        'unemploymant rate \'95 ' : 'unemployment_95',
        'unemploymant rate \'96 ' : 'unemployment_96',
        'no. of enterpreneurs per 1000 inhabitants ' : 'enterpreneurs_per_1000',
        'no. of commited crimes \'95 ' : 'crimes_95',
        'no. of commited crimes \'96 ': 'crimes_96',
    })

    df['crimes_95'] = pd.to_numeric(df['crimes_95'], errors='coerce')
    df['crimes_95'].fillna(df['crimes_96'], inplace=True)

    df['unemployment_95'] = pd.to_numeric(df['unemployment_95'], errors='coerce')
    df['unemployment_95'].fillna(df['unemployment_96'], inplace=True)

    df['crimes_95_per_1000'] = df['crimes_95'] / df['population'] * 1000
    df['crimes_96_per_1000'] = df['crimes_96'] / df['population'] * 1000

    df = df.drop(['crimes_95', 'crimes_96'], axis=1)
    return df

def get_loan_account_district_data(remove_non_numeric=False):
    # Available Columns
    #   loan_id
    #   account_id
    #   date_x
    #   amount
    #   duration
    #   payments
    #   status
    #   district_id
    #   frequency
    #   date_y
    #   code
    #   name
    #   region
    #   population
    #   muni_under499
    #   muni_500_1999
    #   muni_2000_9999
    #   muni_over10000
    #   n_cities
    #   ratio_urban
    #   avg_salary
    #   unemployment_95
    #   unemployment_96
    #   enterpreneurs_per_1000
    #   crimes_95_per_1000
    #   crimes_96_per_1000
    loan_dev, loan_competition = get_loan_data()

    account = pd.read_csv('../data/account.csv', sep=';')

    trans_dev = pd.read_csv('../data/trans_train.csv', sep=';')
    trans_dev = trans_dev.groupby('account_id')['trans_id'].count().rename('trans_count').reset_index()

    account = pd.merge(left=account, right=trans_dev, on='account_id')

    district = pd.read_csv('../data/district.csv', sep=';', na_values='?')

    district = clean_district_data(district)

    account_district = pd.merge(left=account, right=district, left_on='district_id', right_on='code')

    dev = pd.merge(left=loan_dev, right=account_district, left_on='account_id', right_on='account_id')
    competition = pd.merge(left=loan_competition, right=account_district, left_on='account_id', right_on='account_id')

    if remove_non_numeric:
        dev = dev.select_dtypes(['number']).copy()
        competition = competition.select_dtypes(['number']).copy()

    return dev, competition

def get_card_data():
    # Available Columns
    #   card_id
    #   disp_id
    #   type
    #   issued
    dev = pd.read_csv('../data/card_train.csv', sep=';')
    competition = pd.read_csv('../data/card_test.csv', sep=';')
    return dev, competition

def get_transactions_data():
    # Available Columns
    #   trans_id
    #   account_id
    #   date
    #   type
    #   operation
    #   amount
    #   balance
    #   k_symbol
    #   bank
    #   account
    dev = pd.read_csv('../data/trans_train.csv', sep=';', dtype={
        'trans_id': int,
        'account_id': int,
        'date': int,
        'type': 'category',
        'operation': 'category',
        'amount': float,
        'balance': float,
        'k_symbol': 'category',
        'bank': 'category',
        'account': 'Int64'
    })
    competition = pd.read_csv('../data/trans_test.csv', sep=';')
    return dev, competition

def get_client_data():
    # Available Columns
    #   client_id
    #   birthday
    #   sex
    #   district_id
    client = pd.read_csv('../data/client.csv', sep=';')
    client['birthday'] = client['birth_number'].apply(lambda x: get_birthday_from_birth_number(x))
    client['gender'] = client['birth_number'].apply(lambda x: get_gender_from_birth_number(x))
    client = client.drop('birth_number', axis=1)
    return client

def get_birthday_from_birth_number(birth_number):
    year              = birth_number // 10000
    month_with_gender = (birth_number % 10000) // 100
    day               = birth_number % 100
    month = month_with_gender % 50
    return '-'.join(['19' + str(year).zfill(2), str(month).zfill(2), str(day).zfill(2)])

def get_gender_from_birth_number(birth_number):
    month_with_gender = (birth_number % 10000) // 100
    return 'Female' if month_with_gender > 50 else 'Male'

def get_disposition_data():
    # Available Columns
    #   disp_id
    #   client_id
    #   account_id
    #   type
    disposition = pd.read_csv('../data/disp.csv', sep=';', dtype={
        'distp_id': int,
        'client_id': int,
        'account_id': int,
        'type': 'category',
    })
    return disposition

def main():
    #with pd.option_context('display.max_columns', None):
    #    print(get_loan_account_district_data(remove_non_numeric=True)[0].iloc[[0]])

    # clients = get_client_data()
    # print(clients.nunique())
    # print(clients.dtypes)

    transactions = get_disposition_data()
    print(transactions.head())

if __name__ == '__main__':
    main()