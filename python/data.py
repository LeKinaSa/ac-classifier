
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
            (last_transaction_date, last_transaction_balance) = (row2.date, row2.balance)

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
            line['high_balance'] = len(list(filter(lambda x: x < loan_payments, days))) / len(days)
            
            for index in range(len(days) - 1, -1, -1):
                if days[index] < 0:
                    line['last_neg'] = (len(days) - index) / len(days)
                    break
            else:
                line['last_neg'] = 1
            for index in range(len(days) - 1, -1, -1):
                if days[index] > loan_payments:
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

def merge_loan_account_district(remove_non_numeric=False): # Loan, Account, District
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

def merge_client_dispowner(): # Client, Disposition(owner)
    clients = get_client_data()
    disposition = get_disposition_data()
    disposition_owners = disposition.loc[disposition['type'] == 'OWNER']
    disposition_owners = disposition_owners.drop(['type'], axis=1)
    owners = pd.merge(left=disposition_owners, right=clients, on='client_id', how='left')
    return owners

def merge_loan_account_client_dispowner(): # Loan, Account, Client, Disposition(owner)
    loan_dev, loan_comp = get_loan_data()
    account = get_account_data()
    owners = merge_client_dispowner()

    loan_dev  =  loan_dev.rename(columns={'date': 'date_loan'})
    loan_comp = loan_comp.rename(columns={'date': 'date_loan'})
    account   =   account.rename(columns={'date': 'date_account', 'district_id': 'district_id_account'})
    owners    =    owners.rename(columns={'district_id': 'district_id_owner'})

    account_owner = pd.merge(left=owners, right=account, on='account_id', how='left')
    loan_owner_dev  = pd.merge(left=loan_dev , right=account_owner, on='account_id', how='left')
    loan_owner_comp = pd.merge(left=loan_comp, right=account_owner, on='account_id', how='left')
    return (loan_owner_dev, loan_owner_comp)

def merge_loan_account_client_dispowner_districtaccount_districtowner(): # Loan, Account, Client, Disposition(owner), District(account), District(owner)
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
        'entrepreneurs_per_1000' : 'entrepreneurs_per_1000_account',
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
        'entrepreneurs_per_1000' : 'entrepreneurs_per_1000_owner',
        'crimes_95_per_1000' : 'crimes_95_per_1000_owner',
        'crimes_96_per_1000' : 'crimes_96_per_1000_owner',
    })
    loan_owner_dev, loan_owner_comp = merge_loan_account_client_dispowner()

    dev = pd.merge(left=loan_owner_dev, right=account_district, left_on='district_id_account', right_on='code_account', how='left')
    dev = pd.merge(left=dev, right=client_district, left_on='district_id_owner', right_on='code_owner', how='left')

    comp = pd.merge(left=loan_owner_comp, right=account_district, left_on='district_id_account', right_on='code_account', how='left')
    comp = pd.merge(left=comp, right=client_district, left_on='district_id_owner', right_on='code_owner', how='left')

    return (dev, comp)

def merge_loan_account_client_dispowner_districtaccount_districtowner_card(): # Loan, Account, Client, Disposition(owner), District(account), District(owner), Card(owner)
    loan_owner_district_dev, loan_owner_district_comp = merge_loan_account_client_dispowner_districtaccount_districtowner()
    card = get_card_data()

    dev  = pd.merge(left=loan_owner_district_dev , right=card , on='disp_id', how='left')
    comp = pd.merge(left=loan_owner_district_comp, right=card, on='disp_id', how='left')
    
    return (dev, comp)

def merge_client_dispdisponent_district(): # Client, Disposition(disponent), District(disponent)
    clients = get_client_data()
    disposition = get_disposition_data()
    disposition_disponents = disposition.loc[disposition['type'] == 'DISPONENT']
    disposition_disponents = disposition_disponents.drop(['type'], axis=1)
    disponents = pd.merge(left=disposition_disponents, right=clients, on='client_id', how='left')
    district = get_district_data()
    disponent_district = pd.merge(left=disponents, right=district, left_on='district_id', right_on='code', how='left')
    return disponent_district

def merge_loan_account_client_dispowner_districtaccount_districtowner_card_dispdisponent(): # Loan, Account, Client, Disposition(owner), District(account), District(owner), Card(owner), Disposition(disponent)
    loan_dev, loan_comp = merge_loan_account_client_dispowner_districtaccount_districtowner_card()
    disponent = merge_client_dispdisponent_district()
    
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
        'entrepreneurs_per_1000' : 'entrepreneurs_per_1000_disponent',
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

def merge_all():
    (dev, comp) = merge_loan_account_client_dispowner_districtaccount_districtowner_card_dispdisponent()
    transactions = get_improved_transaction_data()
    
    dev  = pd.merge(left=dev , right=transactions, on='account_id')
    comp = pd.merge(left=comp, right=transactions, on='account_id')

    return (dev, comp)

def save_raw_data():
    d, c = merge_all()
    os.makedirs('../data/processed/', exist_ok=True)
    d.to_csv('../data/processed/dev.csv', index=False)
    c.to_csv('../data/processed/comp.csv', index=False)

### Using the Data Saved ###

def get_raw_data():
    dev_file  = '../data/processed/dev.csv'
    comp_file = '../data/processed/comp.csv'

    if not os.path.exists(dev_file) or not os.path.exists(comp_file):
        save_raw_data()
    
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

def normalize_district(df, info):
    # Normalize Urban Ratio
    df[f'ratio_urban_{info}'] = df[f'ratio_urban_{info}'].apply(lambda x: x/100)

    # Obtain total municipalities
    municipalities = [f'muni_under499_{info}', f'muni_500_1999_{info}', f'muni_2000_9999_{info}', f'muni_over10000_{info}']
    df['total_muni'] = 0
    for column in municipalities:
        df['total_muni'] = df['total_muni'].add(df[column])
    
    # Normalize municipalities
    municipalities.append(f'n_cities_{info}')
    for column in municipalities:
        df[column] = df[column].divide(df['total_muni'])

    return df.drop('total_muni', axis=1)

def normalize_region(df, info):
    dev, _ = get_raw_data()
    # The percentages of loans paid per region is calculated based on the dev dataset
    
    name = f'name_{info}'
    region = f'region_{info}'
    region_non_paid = f'region_non_paid_partial_{info}'

    region_total     = dev.groupby(region).size().rename('total').reset_index()
    region_by_status = dev.groupby([region,'status']).size().rename('non_paid').reset_index()
    region_by_status = region_by_status.loc[region_by_status['status'] == 1].drop('status', axis=1)
    region_partials  = pd.merge(left=region_total, right=region_by_status, on=region)
    region_partials[region_non_paid] = region_partials['non_paid'].divide(region_partials['total'])
    region_partials  = region_partials.drop(['non_paid', 'total'], axis=1)
    
    df = pd.merge(left=df, right=region_partials, on=region)
    df = df.drop([name, region], axis=1)
    return df

def convert_gender_to_int(x):
    return 1 if x == 'Female' else 0

def convert_card_to_int(card):
    if card == 'junior' or card == 'classic' or card == 'gold':
        return 1
    return 0

def drop_district_info(d, info):
    d = d.drop([
        'name_' + info, 'region_' + info, 'population_' + info,
        'muni_under499_' + info, 'muni_500_1999_' + info,
        'muni_2000_9999_' + info, 'muni_over10000_' + info,
        'n_cities_' + info, 'ratio_urban_' + info,
        'avg_salary_' + info, 'unemployment_95_' + info,
        'unemployment_evolution_' + info, 'entrepreneurs_per_1000_' + info,
        'crimes_95_per_1000_' + info, 'crimes_evolution_' + info
    ], axis = 1)
    return d

def process_data(d, drop_loan_date=True):
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
    # Normalize card information
    d['card'] = d['type'].apply(convert_card_to_int)
    d = d.drop('type', axis=1)

    # Theory: dates are not important, but maybe ages are
    d = get_ages(d, ['date_account', 'issued', 'birthday_owner', 'birthday_disponent'], 'date_loan')
    d = d.rename(columns={
        'date_account' : 'age_account',
        'issued' : 'age_card',
        'birthday_owner' : 'age_owner',
        'birthday_disponent' : 'age_disponent'
    })
    
    # Since the date_loan was used to normalize the dates, it is no longer needed
    if drop_loan_date:
        d = d.drop('date_loan', axis=1)
        pass
    
    # Theory: ages are not normalized so maybe they can be a problem (?)
    #d = d.drop(['age_account', 'age_card', 'age_owner', 'age_disponent'], axis=1)
    
    # Theory: number of transactions is not normalized so maybe it can be a problem (?)
    #d = d.drop('n_transactions', axis=1)

    # Theory: The disponent gender doesn't affect the status of the loan, maybe the disponent does
    #d = d.drop('gender_disponent', axis=1)

    # Normalize the owner's gender
    d['gender_owner'] = d['gender_owner'].apply(convert_gender_to_int)

    # Theory: Only 1 district will affect the loan
    #d = drop_district_info(d, 'disponent')
    #d = drop_district_info(d, 'owner')

    # Normalize district
    d = normalize_district(d, 'account')

    # Population is a big number with no normalization, is it a problem?
    #d = d.drop('population_account', axis=1)

    # Normalize Region (from text to float)
    d = normalize_region(d, 'account')

    return d

def get_data(remove_loan_date=True):
    (d, c) = get_raw_data()
    return (process_data(d, remove_loan_date), process_data(c, remove_loan_date))

def select(d, columns):
    new = pd.DataFrame()
    for c in columns:
        new[c] = d[c]
    return new

def main():
    d, _ = get_data()
    # print(d.dtypes)
    print(len(d.dtypes))

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

if __name__ == '__main__':
    set_working_directory()
    main()
