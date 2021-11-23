
import pandas as pd

### Base Tables ###

def get_loan_data(): # Loan (loan_dev, loan_comp)
    # Available Columns
    #   loan_id
    #   account_id
    #   date
    #   amount
    #   duration
    #   payments
    #   status
    dev = pd.read_csv('../data/loan_train.csv', sep=';')
    competition = pd.read_csv('../data/loan_test.csv', sep=';')

    dev['status'] = dev['status'].apply(lambda x: 1 if x == -1 else 0)

    return dev, competition

def get_district_data(): # District (district)
    # Available columns
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
    # Available Columns
    #   account_id
    #   district_id
    #   frequency
    #   date
    account = pd.read_csv('../data/account.csv', sep=';')
    return account

def get_card_data(): # Card (card_dev, card_comp)
    # Available Columns
    #   card_id
    #   disp_id
    #   type
    #   issued
    dev = pd.read_csv('../data/card_train.csv', sep=';')
    competition = pd.read_csv('../data/card_test.csv', sep=';')
    return dev, competition

def get_transactions_data(): # Transactions (trans_dev, trans_comp)
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

def get_birthday_from_birth_number(birth_number):
    year              = birth_number // 10000
    month_with_gender = (birth_number % 10000) // 100
    day               = birth_number % 100
    month = month_with_gender % 50
    return '-'.join(['19' + str(year).zfill(2), str(month).zfill(2), str(day).zfill(2)])

def get_gender_from_birth_number(birth_number):
    month_with_gender = (birth_number % 10000) // 100
    return 'Female' if month_with_gender > 50 else 'Male'

def get_client_data(): # Client (client)
    # Available Columns
    #   client_id
    #   birthday
    #   gender
    #   district_id
    client = pd.read_csv('../data/client.csv', sep=';')
    client['birthday'] = client['birth_number'].apply(lambda x: get_birthday_from_birth_number(x))
    client['gender'] = client['birth_number'].apply(lambda x: get_gender_from_birth_number(x))
    client = client.drop('birth_number', axis=1)
    return client

def get_disposition_data(): # Disposition (disp)
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

### Merged Tables ###

def get_loan_account_district_data(remove_non_numeric=False): # Loan, Account, District
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

    account = get_account_data()
    district = get_district_data()

    account_district = pd.merge(left=account, right=district, left_on='district_id', right_on='code')

    dev = pd.merge(left=loan_dev, right=account_district, left_on='account_id', right_on='account_id')
    competition = pd.merge(left=loan_competition, right=account_district, left_on='account_id', right_on='account_id')

    if remove_non_numeric:
        dev = dev.select_dtypes(['number']).copy()
        competition = competition.select_dtypes(['number']).copy()

    return dev, competition

def get_account_owner_data(): # Client, Disposition(owner)
    # Available Columns
    #   disp_id
    #   client_id
    #   account_id
    #   district_id
    #   birthday
    #   gender

    clients = get_client_data()
    disposition = get_disposition_data()
    disposition_owners = disposition.loc[disposition['type'] == 'OWNER']
    disposition_owners = disposition_owners.drop(['type'], axis=1)
    owners = pd.merge(left=disposition_owners, right=clients, on='client_id')
    return owners

def get_loan_client_owner_data(): # Loan, Account, Clients, Disposition(owner)
    # Available Columns
    #   loan_id
    #   account_id
    #   amount
    #   duration
    #   payments
    #   status
    #   date_loan
    #   disp_id
    #   client_id
    #   birthday
    #   gender
    #   district_id_owner
    #   frequency
    #   date_account
    #   district_id_account

    loan_dev, loan_comp = get_loan_data()
    account = get_account_data()
    owners = get_account_owners_data()

    loan_dev  =  loan_dev.rename(columns={'date': 'date_loan'})
    loan_comp = loan_comp.rename(columns={'date': 'date_loan'})
    account   =   account.rename(columns={'date': 'date_account', 'district_id': 'district_id_account'})
    owners    =    owners.rename(columns={'district_id': 'district_id_owner'})

    account_owner = pd.merge(left=owners, right=account, on='account_id')
    loan_owner_dev  = pd.merge(left=loan_dev , right=account_owner, on='account_id')
    loan_owner_comp = pd.merge(left=loan_comp, right=account_owner, on='account_id')
    return (loan_owner_dev, loan_owner_comp)

def get_loan_client_owner_district_data(): # Loan, Account, Clients, Disposition(owner), District(account), District(owner)
    # Available Columns
    #   loan_id
    #   account_id
    #   date_loan
    #   amount
    #   duration
    #   payments
    #   status
    #   disp_id
    #   client_id
    #   district_id_owner
    #   birthday
    #   gender
    #   district_id_account
    #   frequency
    #   date_account
    #   code_account
    #   name_account
    #   region_account
    #   population_account
    #   muni_under499_account
    #   muni_500_1999_account
    #   muni_2000_9999_account
    #   muni_over10000_account
    #   n_cities_account
    #   ratio_urban_account
    #   avg_salary_account
    #   unemployment_95_account
    #   unemployment_96_account
    #   enterpreneurs_per_1000_account
    #   crimes_95_per_1000_account
    #   crimes_96_per_1000_account
    #   code_owner
    #   name_owner
    #   region_owner
    #   population_owner
    #   muni_under499_owner
    #   muni_500_1999_owner
    #   muni_2000_9999_owner
    #   muni_over10000_owner
    #   n_cities_owner
    #   ratio_urban_owner
    #   avg_salary_owner
    #   unemployment_95_owner
    #   unemployment_96_owner
    #   enterpreneurs_per_1000_owner
    #   crimes_95_per_1000_owner
    #   crimes_96_per_1000_owner

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

    dev = pd.merge(left=loan_owner_dev, right=account_district, left_on='district_id_account', right_on='code_account')
    dev = pd.merge(left=dev, right=client_district, left_on='district_id_owner', right_on='code_owner')

    comp = pd.merge(left=loan_owner_comp, right=account_district, left_on='district_id_account', right_on='code_account')
    comp = pd.merge(left=comp, right=client_district, left_on='district_id_owner', right_on='code_owner')

    return (dev, comp)

def get_loan_client_owner_district_and_card_data(): # Loan, Account, Clients, Disposition(owner), District(account), District(owner), Card(owner)
    # Available Columns
    #   loan_id
    #   account_id
    #   date_loan
    #   amount
    #   duration
    #   payments
    #   status
    #   disp_id
    #   client_id
    #   district_id_owner
    #   birthday
    #   gender
    #   district_id_account
    #   frequency
    #   date_account
    #   code_account
    #   name_account
    #   region_account
    #   population_account
    #   muni_under499_account
    #   muni_500_1999_account
    #   muni_2000_9999_account
    #   muni_over10000_account
    #   n_cities_account
    #   ratio_urban_account
    #   avg_salary_account
    #   unemployment_95_account
    #   unemployment_96_account
    #   enterpreneurs_per_1000_account
    #   crimes_95_per_1000_account
    #   crimes_96_per_1000_account
    #   code_owner
    #   name_owner
    #   region_owner
    #   population_owner
    #   muni_under499_owner
    #   muni_500_1999_owner
    #   muni_2000_9999_owner
    #   muni_over10000_owner
    #   n_cities_owner
    #   ratio_urban_owner
    #   avg_salary_owner
    #   unemployment_95_owner
    #   unemployment_96_owner
    #   enterpreneurs_per_1000_owner
    #   crimes_95_per_1000_owner
    #   crimes_96_per_1000_owner
    #   card_id
    #   type
    #   issued

    loan_owner_district_dev, loan_owner_district_comp = get_loan_client_owner_district_data()
    card_dev, card_comp = get_card_data()

    dev  = pd.merge(left=loan_owner_district_dev , right=card_dev , on='disp_id')
    comp = pd.merge(left=loan_owner_district_comp, right=card_comp, on='disp_id')
    
    return (dev, comp)

def get_account_disponent_data(): # Client, Disposition(disponent)
    # Available Columns
    #   disp_id
    #   client_id
    #   account_id
    #   district_id
    #   birthday
    #   gender

    clients = get_client_data()
    disposition = get_disposition_data()
    disposition_disponents = disposition.loc[disposition['type'] == 'DISPONENT']
    disposition_disponents = disposition_disponents.drop(['type'], axis=1)
    disponents = pd.merge(left=disposition_disponents, right=clients, on='client_id')
    return disponents

def main():
    print(get_disposition_data().head())

if __name__ == '__main__':
    main()
