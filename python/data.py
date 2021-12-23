
import os
import statistics
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

### Base Tables ###

def get_loan_data(): # Loan (loan_dev, loan_comp)
    dev = pd.read_csv('../data/loan_train.csv', sep=';')
    competition = pd.read_csv('../data/loan_test.csv', sep=';')

    dev['status'] = dev['status'].apply(lambda x: 1 if x == -1 else 0)

    return dev, competition

def get_district_data(): # District (district)
    district = pd.read_csv('../data/district.csv', sep=';', na_values='?')

    district = district.rename(columns={
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
        'no. of enterpreneurs per 1000 inhabitants ' : 'entrepreneurs_per_1000',
        'no. of commited crimes \'95 ' : 'crimes_95',
        'no. of commited crimes \'96 ': 'crimes_96',
    })

    district['crimes_95'] = pd.to_numeric(district['crimes_95'], errors='coerce')
    district['crimes_95'].fillna(district['crimes_96'], inplace=True)

    district['unemployment_95'] = pd.to_numeric(district['unemployment_95'], errors='coerce')
    district['unemployment_95'].fillna(district['unemployment_96'], inplace=True)

    district['crimes_95_per_1000'] = district['crimes_95'] / district['population'] * 1000
    district['crimes_96_per_1000'] = district['crimes_96'] / district['population'] * 1000

    district['unemployment_growth'] = (
        (district['unemployment_96'] - district['unemployment_95']) /
        district['unemployment_95']
    )
    district['crime_growth'] = (
        (district['crimes_96_per_1000'] - district['crimes_95_per_1000']) /
        district['crimes_95_per_1000']
    )

    district = district.drop(['crimes_95', 'crimes_96'], axis=1)
    return district

def get_account_data(): # Account (account)
    return pd.read_csv('../data/account.csv', sep=';')

def get_card_data(): # Card (card_dev + card_comp)
    dev = pd.read_csv('../data/card_train.csv', sep=';')
    competition = pd.read_csv('../data/card_test.csv', sep=';')
    return dev.append(competition)

def get_transactions_data(): # Transactions (trans_dev + trans_comp)
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
    return dev.append(competition)

def get_client_data(): # Client (client)
    client = pd.read_csv('../data/client.csv', sep=';')
    client['birthday'] = client['birth_number'].apply(get_birthday_from_birth_number)
    client['gender'] = client['birth_number'].apply(get_gender_from_birth_number)
    client = client.drop('birth_number', axis=1)
    return client

def get_disposition_data(): # Disposition (disp)
    disposition = pd.read_csv('../data/disp.csv', sep=';', dtype={
        'disp_id': int,
        'client_id': int,
        'account_id': int,
        'type': 'category',
    })
    return disposition

def get_mean_transaction_data(): # Transactions (mean transaction)
    transactions = get_transactions_data()
    transactions = transactions.drop(['trans_id', 'date', 'balance', 'account', 'bank', 'k_symbol', 'operation'], axis=1)
    
    t = transactions.groupby('account_id')['amount'].mean().rename('avg_abs_amount').reset_index()

    transactions = modify_transactions_by_type(transactions)
    transactions['sign'] = transactions['amount'].apply(lambda x: 1 if x >= 0 else 0)
    t_credit = transactions.groupby('account_id')['sign'].mean().rename('credit_ratio').reset_index()

    transactions = transactions.groupby('account_id')['amount'].mean().rename('avg_amount').reset_index()

    t = pd.merge(left=t, right=t_credit, on='account_id')
    return pd.merge(left=transactions, right=t, on='account_id')

def get_average_daily_balance_data(only_loans=True): # Transactions (average daily balance)
    loan_dev, loan_comp = get_loan_data()
    loans = loan_dev.append(loan_comp)
    loans = loans.drop(['loan_id', 'amount', 'duration', 'status'], axis=1)
    transactions = get_transactions_data()
    transactions = transactions.drop(['trans_id', 'account', 'bank', 'k_symbol', 'operation', 'amount', 'type'], axis=1)

    transactions['date'] = pd.to_datetime(transactions['date'].apply(get_readable_date))
    loans['date'] = pd.to_datetime(loans['date'].apply(get_readable_date))

    df = pd.DataFrame()

    for group in transactions.groupby('account_id'):
        line = {}
        id = group[0]
        group_df = group[1]

        loan = loans.loc[loans['account_id'] == id]
        if only_loans:
            if len(loan) == 0:
                    continue
            loan_date = loan.iloc[0]['date']
            loan_payments = loan.iloc[0]['payments']

        line['account_id'] = id

        days = []

        for row1, row2 in zip(group_df.iterrows(), group_df.iloc[1:].iterrows()):
            row1 = row1[1]
            row2 = row2[1]

            interval = (row2.date - row1.date).days

            days += [row1.balance] * interval
            last_transaction_date, last_transaction_balance = row2.date, row2.balance

        if only_loans:
            interval = (loan_date - last_transaction_date).days
            days += [last_transaction_balance] * interval

        line['avg_balance'] = group[1]['balance'].mean()

        if len(days) == 0:
            line['negative_balance'] = None
            line['high_balance'] = None
            line['last_neg'] = None
            line['last_high'] = None
            line['avg_daily_balance'] = None
            line['balance_distribution_first_quarter'] = None
            line['balance_distribution_median']        = None
            line['balance_distribution_third_quarter'] = None
            line['balance_deviation'] = None
        else:
            line['negative_balance'] = len(list(filter(lambda x: x < 0, days))) / len(days)

            if only_loans:
                line['high_balance'] = len(list(filter(lambda x: x < loan_payments, days))) / len(days)
            
            for index in range(len(days) - 1, -1, -1):
                if days[index] < 0:
                    line['last_neg'] = (len(days) - index) / len(days)
                    break
            else:
                line['last_neg'] = 1
            for index in range(len(days) - 1, -1, -1):
                if only_loans and days[index] > loan_payments:
                    line['last_high'] = (len(days) - index) / len(days)
                    break
            else:
                line['last_high'] = 1
            
            line['avg_daily_balance'] = statistics.mean(days)

            first_quarter, median, third_quarter = statistics.quantiles(days)

            line['balance_distribution_first_quarter'] = first_quarter
            line['balance_distribution_median']        = median
            line['balance_distribution_third_quarter'] = third_quarter

            line['balance_deviation'] = statistics.stdev(days)

        df = df.append(pd.Series(line), ignore_index=True)
    
    return df

def get_number_of_transactions_data(): # Transactions (number of transactions)
    return (get_transactions_data().groupby('account_id')['trans_id']
        .count().rename('n_transactions').reset_index())

def get_improved_transaction_data(only_loans=True): # Transactions (improved)
    transactions_mean = get_mean_transaction_data()
    transactions_daily = get_average_daily_balance_data(only_loans)
    n_transactions = get_number_of_transactions_data()

    transactions_info = pd.merge(left=transactions_mean, right=n_transactions, on='account_id')
    return pd.merge(left=transactions_daily, right=transactions_info, on='account_id')

### Merged Tables ###

def merge_all():
    dev, comp = get_loan_data()
    dev.rename(columns={'date': 'date_loan'}, inplace=True)
    comp.rename(columns={'date': 'date_loan'}, inplace=True)

    account = get_account_data()
    account.drop('district_id', axis=1, inplace=True)
    account.rename(columns={'date': 'date_account'}, inplace=True)

    clients = get_client_data()
    disposition = get_disposition_data()
    disposition_owners = disposition.loc[disposition['type'] == 'OWNER']
    disposition_owners = disposition_owners.drop(['type'], axis=1)
    owners = pd.merge(left=disposition_owners, right=clients, on='client_id', how='left')

    tables = pd.merge(left=owners, right=account, on='account_id', how='left')

    district = get_district_data()
    tables = pd.merge(left=tables, right=district, left_on='district_id', right_on='code', how='left')

    card = get_card_data()
    tables = pd.merge(left=tables, right=card, on='disp_id', how='left')

    transactions = get_improved_transaction_data()
    tables = pd.merge(left=tables, right=transactions, on='account_id')
    
    dev  = pd.merge(left=dev , right=tables, on='account_id')
    comp = pd.merge(left=comp, right=tables, on='account_id')

    return dev, comp

### Save and Read Data ###

def save_raw_loans_data():
    d, c = merge_all()
    os.makedirs('../data/processed/', exist_ok=True)
    d.to_csv('../data/processed/dev.csv', index=False)
    c.to_csv('../data/processed/comp.csv', index=False)

def get_raw_loans_data():
    dev_file  = '../data/processed/dev.csv'
    comp_file = '../data/processed/comp.csv'

    if not os.path.exists(dev_file) or not os.path.exists(comp_file):
        save_raw_loans_data()
    
    d = pd.read_csv(dev_file)
    c = pd.read_csv(comp_file)
    
    return d, c

### Data Transformation ###

def get_birthday_from_birth_number(birth_number):
    year              = birth_number // 10000
    month_with_gender = (birth_number % 10000) // 100
    day               = birth_number % 100
    month = month_with_gender % 50
    return int(str(year).zfill(2) + str(month).zfill(2) + str(day).zfill(2))

def get_readable_date(date):
    year = str(date)[0:2]
    month = str(date)[2:4]
    day = str(date)[4:6]
    return '-'.join(['19' + str(year).zfill(2), str(month).zfill(2), str(day).zfill(2)])

def get_gender_from_birth_number(birth_number):
    month_with_gender = (birth_number % 10000) // 100
    return 'Female' if month_with_gender > 50 else 'Male'

def modify_transactions_by_type(transactions):
    transactions.loc[transactions['type'].isin(['withdrawal', 'withdrawal in cash']), 'amount'] = \
        transactions.loc[transactions['type'].isin(['withdrawal', 'withdrawal in cash']), 'amount'].apply(lambda x: -x)
    return transactions

def normalize(df, columns, normalizer):
    for column in columns:
        df[column] = df[column].divide(df[normalizer])
    return df

def get_ages(df, dates, loan_date):
    for date in dates:
        df[date] = df.apply(
            lambda row: row[date] if pd.isnull(row[date]) else int((row[loan_date] - row[date]).days / 365.2425 * 12),
            axis=1
        )
    return df

def normalize_district(df):
    # Convert ratio_urban from percentage to ratio
    df['ratio_urban'] = df['ratio_urban'] / 100

    # Obtain total municipalities
    municipalities = ['muni_under499', 'muni_500_1999', 'muni_2000_9999', 'muni_over10000']
    df['total_muni'] = 0
    for column in municipalities:
        df['total_muni'] = df['total_muni'] + df[column]
    
    # Convert municipality information to ratios
    municipalities.append('n_cities')
    for column in municipalities:
        df[column] = df[column] / df['total_muni']

    return df.drop('total_muni', axis=1)

def convert_gender_to_int(x):
    return 1 if x == 'Female' else 0

def process_loan_data(d):
    # Drop ids and disctrict codes (don't drop 'loan_id')
    d = d.drop([
        'account_id', 'district_id', 'code', 'name', 'region', 'disp_id', 'card_id', 'client_id'
    ], axis=1)

    # The transaction values need to be normalized with the monthly payments
    d = normalize(d, [
        'avg_amount', 'avg_abs_amount', 'avg_balance',
        'avg_daily_balance', 'balance_deviation', 
        'balance_distribution_first_quarter', 'balance_distribution_median', 
        'balance_distribution_third_quarter'
    ], 'payments')

    # Use the payment to normalize salaries
    d = normalize(d, ['avg_salary'], 'payments')

    # Transform card type information through one hot encoding
    encoder = OneHotEncoder()
    one_hot = encoder.fit_transform(d['type'].values.reshape(-1, 1)).toarray()
    
    cols = list(encoder.categories_[0])
    cols[3] = 'none'
    cols = list(map(lambda x: 'card_' + x, cols))

    card_df = pd.DataFrame(one_hot, columns=cols, dtype=int)
    d = d.join(card_df)
    d = d.drop('type', axis=1)
    
    # Convert dates to ages
    d['issued'] = d['issued'].astype('Int64')
    for date in ['date_account', 'issued', 'birthday', 'date_loan']:
        d[date] = pd.to_datetime(d[date].apply(
            lambda x: get_readable_date(x) if type(x) is int else x))

    d = get_ages(d, ['date_account', 'issued', 'birthday'], 'date_loan')
    d = d.rename(columns={
        'date_account' : 'age_account_months',
        'issued' : 'age_card_months',
        'birthday' : 'age_owner',
    })
    d = d.drop('date_loan', axis=1)

    # Convert owner age to years
    d['age_owner'] = (d['age_owner'] / 12).astype(int)

    # Normalize district data
    d = normalize_district(d)
    
    # Convert owner's gender to integer
    d['gender'] = d['gender'].apply(convert_gender_to_int)

    return d

def get_clustering_data():
    clustering_file  = '../data/processed/clustering.csv'

    if os.path.exists(clustering_file):
        return pd.read_csv(clustering_file)
    
    acc = get_account_data()
    trans = get_improved_transaction_data(only_loans=False)
    dist = get_district_data()
    client = get_client_data()
    disp = get_disposition_data()
    loan_dev, loan_comp = get_loan_data()
    loan = loan_dev.append(loan_comp, ignore_index=True)
    loan.rename(columns={'date': 'date_loan'}, inplace=True)

    client.drop('district_id', axis=1, inplace=True)

    # Merge data frames
    df = pd.merge(client, disp, on='client_id')
    df = pd.merge(df, acc, on='account_id')
    df = pd.merge(df, dist, left_on='district_id', right_on='code')
    df = pd.merge(df, trans, on='account_id')
    df = pd.merge(df, loan, on='account_id', how='left')

    # Drop unnecessary columns
    df.drop(
        ['client_id', 'district_id', 'disp_id', 'account_id', 'code', 
        'name'],
        axis=1,
        inplace=True
    )

    for date in ['birthday', 'date']:
        df[date] = pd.to_datetime(df[date].apply(get_readable_date))
    
    df['age'] = df.apply(
        lambda row: int((row['date'] - row['birthday']).days / 365.2425),
        axis=1
    )
    df.drop(['birthday', 'date'], axis=1, inplace=True)

    df['gender'] = df['gender'].apply(convert_gender_to_int)
    
    os.makedirs('../data/processed/', exist_ok=True)
    df.to_csv('../data/processed/clustering.csv', index=False)

    return df

def get_loans_data():
    d, c = get_raw_loans_data()
    return process_loan_data(d), process_loan_data(c)

def select(d, columns):
    new = pd.DataFrame()
    for c in columns:
        new[c] = d[c]
    return new

def set_working_directory():
    cwd = os.getcwd()
    ubuntu_split = cwd.split('/')
    windows_split = cwd.split('\\')
    if len(ubuntu_split) == 1:
        # Windows OS
        if windows_split[-1] != 'python':
            os.chdir(cwd + '\python')
    else:
        # Ubuntu OS
        if ubuntu_split[-1] != 'python':
            os.chdir(cwd + '/python')

def main():
    client = get_client_data()
    account = get_account_data()
    disp = get_disposition_data()
    dev, comp = get_loan_data()
    loan = dev.append(comp, ignore_index=True)
    loan.rename(columns={'date': 'date_loan'}, inplace=True)

    a = pd.merge(disp, account, on='account_id')
    a = pd.merge(a, client, on='client_id')
    a = pd.merge(a, loan, on='account_id')

    duplicated_mask = a.duplicated(['birthday', 'gender', 'district_id_x', 
        'district_id_y', 'type', 'frequency', 'date', 'amount', 'duration', 
        'date_loan', 'status'], keep=False)

    duplicated = a[duplicated_mask].sort_values(by=['birthday', 'gender', 'date'])

    print(len(duplicated))
    # d, c = get_loans_data()
    # print('Loans Development:', d.shape)
    # print('Loans Competition:', c.shape)
    # clients = get_clients_data()
    # print('Clients:', clients.shape)
    # (d, c) = (d.drop(useless, axis=1), c.drop(useless, axis=1))
    # d.to_csv('../data/processed/dev_ready.csv', index=False)
    # c.to_csv('../data/processed/comp_ready.csv', index=False)
    # partial = balance(d)
    # partial.to_csv('../data/processed/dev_partial_ready.csv', index=False)

if __name__ == '__main__':
    set_working_directory()
    main()
