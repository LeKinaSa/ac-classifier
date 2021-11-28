
import os
import statistics
import pandas as pd

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
        'no. of enterpreneurs per 1000 inhabitants ' : 'enterpreneurs_per_1000',
        'no. of commited crimes \'95 ' : 'crimes_95',
        'no. of commited crimes \'96 ': 'crimes_96',
    })

    district['crimes_95'] = pd.to_numeric(district['crimes_95'], errors='coerce')
    district['crimes_95'].fillna(district['crimes_96'], inplace=True)

    district['unemployment_95'] = pd.to_numeric(district['unemployment_95'], errors='coerce')
    district['unemployment_95'].fillna(district['unemployment_96'], inplace=True)

    district['crimes_95_per_1000'] = district['crimes_95'] / district['population'] * 1000
    district['crimes_96_per_1000'] = district['crimes_96'] / district['population'] * 1000

    district = district.drop(['crimes_95', 'crimes_96'], axis=1)
    return district

def get_account_data(): # Account (account)
    account = pd.read_csv('../data/account.csv', sep=';')
    return account

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

def get_client_data(): # Client (client)
    client = pd.read_csv('../data/client.csv', sep=';')
    client['birthday'] = client['birth_number'].apply(lambda x: get_birthday_from_birth_number(x))
    client['gender'] = client['birth_number'].apply(lambda x: get_gender_from_birth_number(x))
    client = client.drop('birth_number', axis=1)
    return client

def get_disposition_data(): # Disposition (disp)
    disposition = pd.read_csv('../data/disp.csv', sep=';', dtype={
        'distp_id': int,
        'client_id': int,
        'account_id': int,
        'type': 'category',
    })
    return disposition

def modify_transactions_by_type(transactions):
    transactions.loc[transactions['type'].isin(['withdrawal', 'withdrawal in cash']), 'amount'] = \
        transactions.loc[transactions['type'].isin(['withdrawal', 'withdrawal in cash']), 'amount'].apply(lambda x: -x)
    return transactions

def get_mean_transaction_data(): # Transactions (mean transaction)
    transactions = get_transactions_data()
    transactions = transactions.drop(['trans_id', 'date', 'balance', 'account', 'bank', 'k_symbol', 'operation'], axis=1)
    
    t = transactions.groupby('account_id')['amount'].mean().rename('avg_abs_amount').reset_index()

    transactions = modify_transactions_by_type(transactions)
    transactions = transactions.groupby('account_id')['amount'].mean().rename('avg_amount').reset_index()

    return pd.merge(left=transactions, right=t, on='account_id')

def get_average_daily_balance_data(): # Transactions (average daily balance)
    loan_dev, loan_comp = get_loan_data()
    loans = loan_dev.append(loan_comp)
    loans = loans.drop(['loan_id', 'amount', 'duration', 'payments', 'status'], axis=1)
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
        if len(loan) == 0:
            continue
        loan_date = loan.iloc[0]['date']

        line['account_id'] = id

        days = []

        for row1, row2 in zip(group_df.iterrows(), group_df.iloc[1:].iterrows()):
            row1 = row1[1]
            row2 = row2[1]

            interval = (row2.date - row1.date).days

            days += [row1.balance] * interval
            (last_transaction_date, last_transaction_balance) = (row2.date, row2.balance)

        interval = (loan_date - last_transaction_date).days
        days += [last_transaction_balance] * interval

        line['avg_balance'] = group[1]['balance'].mean()

        if len(days) == 0:
            line['avg_daily_balance'] = None
            line['balance_distribution_first_quarter'] = None
            line['balance_distribution_median']        = None
            line['balance_distribution_third_quarter'] = None
            line['balance_deviation'] = None
        else:
            line['avg_daily_balance'] = statistics.mean(days)

            first_quarter, median, third_quarter = statistics.quantiles(days)

            line['balance_distribution_first_quarter'] = first_quarter
            line['balance_distribution_median']        = median
            line['balance_distribution_third_quarter'] = third_quarter

            line['balance_deviation'] = statistics.stdev(days)

        df = df.append(pd.Series(line), ignore_index=True)
    
    return df

def get_number_of_transactions_data(): # Transactions (number of transactions)
    transactions = get_transactions_data()
    transactions = transactions.groupby('account_id')['trans_id'].count().rename('n_transactions').reset_index()
    return transactions

def get_improved_transaction_data(): # Transactions (improved)
    transactions_mean = get_mean_transaction_data()
    transactions_daily = get_average_daily_balance_data()
    n_transactions = get_number_of_transactions_data()

    transactions_info = pd.merge(left=transactions_mean, right=n_transactions, on='account_id')
    return pd.merge(left=transactions_daily, right=transactions_info, on='account_id')

### Merged Tables ###

def get_loan_account_district_data(remove_non_numeric=False): # Loan, Account, District
    loan_dev, loan_competition = get_loan_data()

    account = get_account_data()
    district = get_district_data()

    account_district = pd.merge(left=account, right=district, left_on='district_id', right_on='code', how='left')

    dev = pd.merge(left=loan_dev, right=account_district, left_on='account_id', right_on='account_id', how='left')
    competition = pd.merge(left=loan_competition, right=account_district, left_on='account_id', right_on='account_id', how='left')

    if remove_non_numeric:
        dev = dev.select_dtypes(['number']).copy()
        competition = competition.select_dtypes(['number']).copy()

    return dev, competition

def get_account_owner_data(): # Client, Disposition(owner)
    clients = get_client_data()
    disposition = get_disposition_data()
    disposition_owners = disposition.loc[disposition['type'] == 'OWNER']
    disposition_owners = disposition_owners.drop(['type'], axis=1)
    owners = pd.merge(left=disposition_owners, right=clients, on='client_id', how='left')
    return owners

def get_loan_client_owner_data(): # Loan, Account, Client, Disposition(owner)
    loan_dev, loan_comp = get_loan_data()
    account = get_account_data()
    owners = get_account_owner_data()

    loan_dev  =  loan_dev.rename(columns={'date': 'date_loan'})
    loan_comp = loan_comp.rename(columns={'date': 'date_loan'})
    account   =   account.rename(columns={'date': 'date_account', 'district_id': 'district_id_account'})
    owners    =    owners.rename(columns={'district_id': 'district_id_owner'})

    account_owner = pd.merge(left=owners, right=account, on='account_id', how='left')
    loan_owner_dev  = pd.merge(left=loan_dev , right=account_owner, on='account_id', how='left')
    loan_owner_comp = pd.merge(left=loan_comp, right=account_owner, on='account_id', how='left')
    return (loan_owner_dev, loan_owner_comp)

def get_loan_client_owner_district_data(): # Loan, Account, Client, Disposition(owner), District(account), District(owner)
    district = get_district_data()
    account_district = district.rename(columns={
        'code' : 'code_account',
        'name' : 'name_account',
        'region' : 'region_account',
        'population' : 'population_account',
        'muni_under499' : 'muni_under499_account',
        'muni_500_1999' : 'muni_500_1999_account',
        'muni_2000_9999' : 'muni_2000_9999_account',
        'muni_over10000' : 'muni_over10000_account',
        'n_cities' : 'n_cities_account',
        'ratio_urban' : 'ratio_urban_account',
        'avg_salary' : 'avg_salary_account',
        'unemployment_95' : 'unemployment_95_account',
        'unemployment_96' : 'unemployment_96_account',
        'enterpreneurs_per_1000' : 'enterpreneurs_per_1000_account',
        'crimes_95_per_1000' : 'crimes_95_per_1000_account',
        'crimes_96_per_1000' : 'crimes_96_per_1000_account',
    })
    client_district  = district.rename(columns={
        'code' : 'code_owner',
        'name' : 'name_owner',
        'region' : 'region_owner',
        'population' : 'population_owner',
        'muni_under499' : 'muni_under499_owner',
        'muni_500_1999' : 'muni_500_1999_owner',
        'muni_2000_9999' : 'muni_2000_9999_owner',
        'muni_over10000' : 'muni_over10000_owner',
        'n_cities' : 'n_cities_owner',
        'ratio_urban' : 'ratio_urban_owner',
        'avg_salary' : 'avg_salary_owner',
        'unemployment_95' : 'unemployment_95_owner',
        'unemployment_96' : 'unemployment_96_owner',
        'enterpreneurs_per_1000' : 'enterpreneurs_per_1000_owner',
        'crimes_95_per_1000' : 'crimes_95_per_1000_owner',
        'crimes_96_per_1000' : 'crimes_96_per_1000_owner',
    })
    loan_owner_dev, loan_owner_comp = get_loan_client_owner_data()

    dev = pd.merge(left=loan_owner_dev, right=account_district, left_on='district_id_account', right_on='code_account', how='left')
    dev = pd.merge(left=dev, right=client_district, left_on='district_id_owner', right_on='code_owner', how='left')

    comp = pd.merge(left=loan_owner_comp, right=account_district, left_on='district_id_account', right_on='code_account', how='left')
    comp = pd.merge(left=comp, right=client_district, left_on='district_id_owner', right_on='code_owner', how='left')

    return (dev, comp)

def get_loan_client_owner_district_and_card_data(): # Loan, Account, Client, Disposition(owner), District(account), District(owner), Card(owner)
    loan_owner_district_dev, loan_owner_district_comp = get_loan_client_owner_district_data()
    card = get_card_data()

    dev  = pd.merge(left=loan_owner_district_dev , right=card , on='disp_id', how='left')
    comp = pd.merge(left=loan_owner_district_comp, right=card, on='disp_id', how='left')
    
    return (dev, comp)

def get_account_disponent_data(): # Client, Disposition(disponent)
    clients = get_client_data()
    disposition = get_disposition_data()
    disposition_disponents = disposition.loc[disposition['type'] == 'DISPONENT']
    disposition_disponents = disposition_disponents.drop(['type'], axis=1)
    disponents = pd.merge(left=disposition_disponents, right=clients, on='client_id', how='left')
    return disponents

def get_account_disponent_district_data(): # Client, Disposition(disponent), District(disponent)
    disponent = get_account_disponent_data()
    district = get_district_data()
    disponent_district = pd.merge(left=disponent, right=district, left_on='district_id', right_on='code', how='left')
    return disponent_district

def get_loan_client_data(): # Loan, Account, Client, Disposition(owner), District(account), District(owner), Card(owner), Disposition(disponent)
    loan_dev, loan_comp = get_loan_client_owner_district_and_card_data()
    disponent = get_account_disponent_district_data()
    
    disponent = disponent.rename(columns={
        'code' : 'code_disponent',
        'name' : 'name_disponent',
        'region' : 'region_disponent',
        'population' : 'population_disponent',
        'muni_under499' : 'muni_under499_disponent',
        'muni_500_1999' : 'muni_500_1999_disponent',
        'muni_2000_9999' : 'muni_2000_9999_disponent',
        'muni_over10000' : 'muni_over10000_disponent',
        'n_cities' : 'n_cities_disponent',
        'ratio_urban' : 'ratio_urban_disponent',
        'avg_salary' : 'avg_salary_disponent',
        'unemployment_95' : 'unemployment_95_disponent',
        'unemployment_96' : 'unemployment_96_disponent',
        'enterpreneurs_per_1000' : 'enterpreneurs_per_1000_disponent',
        'crimes_95_per_1000' : 'crimes_95_per_1000_disponent',
        'crimes_96_per_1000' : 'crimes_96_per_1000_disponent',        
        'client_id' : 'client_id_disponent',
        'birthday' : 'birthday_disponent',
        'gender' : 'gender_disponent',
        'district_id' : 'district_id_disponent',
    })
    loan_dev = loan_dev.rename(columns={
        'client_id' : 'client_id_owner',
        'birthday' : 'birthday_owner',
        'gender' : 'gender_owner',
        'district_id' : 'district_id_owner',
    })
    loan_comp = loan_comp.rename(columns={
        'client_id' : 'client_id_owner',
        'birthday' : 'birthday_owner',
        'gender' : 'gender_owner',
        'district_id' : 'district_id_owner',
    })

    dev = pd.merge(left=loan_dev, right=disponent, on=['disp_id', 'account_id'], how='left')
    comp = pd.merge(left=loan_comp, right=disponent, on=['disp_id', 'account_id'], how='left')
    
    return (dev, comp)

def get_all_data():
    (dev, comp) = get_loan_client_data()
    transactions = get_improved_transaction_data()
    
    dev  = pd.merge(left=dev , right=transactions, on='account_id')
    comp = pd.merge(left=comp, right=transactions, on='account_id')

    return (dev, comp)

def save_all_data():
    d, c = get_all_data()
    os.makedirs('../data/processed/', exist_ok=True)
    d.to_csv('../data/processed/dev.csv', index=False)
    c.to_csv('../data/processed/comp.csv', index=False)

### Using the Data Saved ###

def get_processed_data():
    dev_file  = '../data/processed/dev.csv'
    comp_file = '../data/processed/comp.csv'

    if not os.path.exists(dev_file) or not os.path.exists(comp_file):
        save_all_data()
    
    d = pd.read_csv(dev_file)
    c = pd.read_csv(comp_file)
    
    return (d, c)

def normalize(df, columns, normalizer):
    for column in columns:
        df[column] = df[column].divide(df[normalizer])
    return df

def normalize_dict(df, dict):
    for k, v in dict.items():
        df[k] = df[k].divide(df[v])
    return df

def get_ages(df, creation_dates, loan_date):
    for creation_date in creation_dates:
        df[creation_date] = df[creation_date].rsub(df[loan_date])
        df[creation_date] = df[creation_date].floordiv(10000)
    return df

def normalize_district(df, muni_under_499, muni_500_1999, muni_2000_9999, muni_over10000, n_cities):
    # Obtain total municipalities
    df['total_muni'] = 0
    for column in [muni_under_499, muni_500_1999, muni_2000_9999, muni_over10000]:
        df['total_muni'] = df['total_muni'].add(df[column])
    
    # Normalize municipalities
    for column in [muni_under_499, muni_500_1999, muni_2000_9999, muni_over10000, n_cities]:
        df[column] = df[column].divide(df['total_muni'])

    return df.drop('total_muni', axis=1)

def process_data(d):
    ### Here are some ideas of what could be done

    # Drop ids and disctrict codes (don't drop 'loan_id')
    d = d.drop([
        'account_id', 'client_id_owner', 'client_id_disponent',
        'district_id_account', 'district_id_owner', 'district_id_disponent',
        'code_account', 'code_owner', 'code_disponent', 'disp_id', 'card_id'
    ], axis=1)
    
    # The loan amount is redundant (since we have the number and value of payments: duration, payments)
    d = d.drop('amount', axis=1)

    # The transaction values need to be normalized with the payment
    d = normalize(d, [
        'avg_amount', 'avg_balance',
        'avg_daily_balance', 'balance_deviation', 'balance_distribution_first_quarter',
        'balance_distribution_median', 'balance_distribution_third_quarter'
    ], 'payments')

    # Use the payment to normalize salaries
    d = normalize(d, ['avg_salary_account', 'avg_salary_owner', 'avg_salary_disponent'], 'payments')
    
    # Since the payment was used to normalize all the value related variables, it is no longer needed
    d = d.drop('payments', axis=1)

    # Normalize 96 values based on 95 and called it evolution
    d = normalize_dict(d, {
        'unemployment_96_account'      : 'unemployment_95_account',
        'unemployment_96_owner'        : 'unemployment_95_owner',
        'unemployment_96_disponent'    : 'unemployment_95_disponent',
        'crimes_96_per_1000_account'   : 'crimes_95_per_1000_account',
        'crimes_96_per_1000_owner'     : 'crimes_95_per_1000_owner',
        'crimes_96_per_1000_disponent' : 'crimes_95_per_1000_disponent',
    })
    d = d.rename(columns={
        'unemployment_96_account'      : 'unemployment_evolution_account',
        'unemployment_96_owner'        : 'unemployment_evolution_owner',
        'unemployment_96_disponent'    : 'unemployment_evolution_disponent',
        'crimes_96_per_1000_account'   : 'crimes_evolution_account',
        'crimes_96_per_1000_owner'     : 'crimes_evolution_owner',
        'crimes_96_per_1000_disponent' : 'crimes_evolution_disponent',
    })

    # Theory: the fact that the account doesn't have a card is information
    d['type'].fillna('None', inplace=True)

    # Theory: dates are not important, but maybe ages are
    d = get_ages(d, ['date_account', 'issued', 'birthday_owner', 'birthday_disponent'], 'date_loan')
    d = d.rename(columns={
        'date_account' : 'age_account',
        'issued' : 'age_card',
        'birthday_owner' : 'age_owner',
        'birthday_disponent' : 'age_disponent'
    })
    
    # Since the date_loan was used to normalize the dates, it is no longer needed
    d = d.drop('date_loan', axis=1)
    
    # Theory: ages are not normalized so maybe they can be a problem (?)
    d = d.drop(['age_account', 'age_card', 'age_owner', 'age_disponent'], axis=1)
    
    # Theory: number of transactions is not normalized so maybe it can be a problem (?)
    d = d.drop('n_transactions', axis=1)

    # Theory: The disponent doesn't affect the status of the loan
    d = d.drop('gender_disponent', axis=1)

    # Theory: The gender doesn't affect the status of the loan
    # d = d.drop('gender_owner', axis=1)

    # Theory: Only 1 district will affect the loan
    d = d.drop([
        'name_disponent', 'region_disponent', 'population_disponent',
        'muni_under499_disponent', 'muni_500_1999_disponent',
        'muni_2000_9999_disponent', 'muni_over10000_disponent',
        'n_cities_disponent', 'ratio_urban_disponent',
        'avg_salary_disponent', 'unemployment_95_disponent',
        'unemployment_evolution_disponent', 'enterpreneurs_per_1000_disponent',
        'crimes_95_per_1000_disponent', 'crimes_evolution_disponent'
    ], axis=1)
    d = d.drop([
        'name_owner', 'region_owner', 'population_owner',
        'muni_under499_owner', 'muni_500_1999_owner',
        'muni_2000_9999_owner', 'muni_over10000_owner', 'n_cities_owner',
        'ratio_urban_owner', 'avg_salary_owner', 'unemployment_95_owner',
        'unemployment_evolution_owner', 'enterpreneurs_per_1000_owner',
        'crimes_95_per_1000_owner', 'crimes_evolution_owner'
    ], axis=1)

    # Normalize district
    d = normalize_district(d, 'muni_under499_account', 'muni_500_1999_account',
            'muni_2000_9999_account', 'muni_over10000_account', 'n_cities_account')

    # Population is a big number with no normalization, is it a problem?
    d = d.drop('population_account', axis=1)

    return d

def get_data():
    (d, c) = get_processed_data()
    return (process_data(d), process_data(c))

def main():
    # save_all_data()
    d, c = get_data()
    # d = d.drop(['avg_amount', 'avg_balance',
    #     'avg_daily_balance', 'balance_deviation', 'balance_distribution_first_quarter',
    #     'balance_distribution_median', 'balance_distribution_third_quarter',

    #     'duration', 'status', 'frequency', 'gender_owner', 'ratio_urban_account',
    #     'avg_salary_account', 'enterpreneurs_per_1000_account',
    #     'unemployment_95_account', 'unemployment_evolution_account', 
    #     'crimes_95_per_1000_account', 'crimes_evolution_account', 'type'
    # ], axis=1)

    # print(d.head(2).transpose())
    print(d.dtypes)
    print(len(d.dtypes))
    # print(len(c.dtypes))


if __name__ == '__main__':
    main()

# Other ideas
#  Time passed with balance < 0
