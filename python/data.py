
import pandas as pd

def get_loan_data():
    dev = pd.read_csv('../data/loan_train.csv', sep=';')
    competition = pd.read_csv('../data/loan_test.csv', sep=';')

    dev['status'] = dev['status'].apply(lambda x: 1 if x == -1 else 0)

    return dev, competition

def clean_district_data(district):
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
    district = pd.read_csv('../data/district.csv', sep=';', na_values='?')

    district = clean_district_data(district)

    account_district = pd.merge(left=account, right=district, left_on='district_id', right_on='code')

    dev = pd.merge(left=loan_dev, right=account_district, left_on='account_id', right_on='account_id')
    competition  = pd.merge(left=loan_competition , right=account_district, left_on='account_id', right_on='account_id')

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
    # TODO: sys:1: DtypeWarning: Columns (8) have mixed types.Specify dtype option on import or set low_memory=False.
    
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
    dev = pd.read_csv('../data/trans_train.csv', sep=';')
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
    client['sex'] = client['birth_number'].apply(lambda x: get_sex_from_birth_number(x))
    client = client.drop('birth_number', axis=1)
    return client

def get_birthday_from_birth_number(birth_number):
    year           =  birth_number // 10000
    month_with_sex = (birth_number %  10000) // 100
    day            =  birth_number % 100
    month = month_with_sex % 50
    return year * 10000 + month * 100 + day

def get_sex_from_birth_number(birth_number):
    month_with_sex = (birth_number %  10000) // 100
    return month_with_sex > 50

def get_disposition_data():
    # Available Columns
    #   disp_id
    #   client_id
    #   account_id
    #   type
    disposition = pd.read_csv('../data/disp.csv', sep=';')
    return disposition

def main():
    #with pd.option_context('display.max_columns', None):
    #    print(get_loan_account_district_data(remove_non_numeric=True)[0].iloc[[0]])

    clients = get_client_data()
    print(clients.nunique())
    print(clients.dtypes)

if __name__ == '__main__':
    main()