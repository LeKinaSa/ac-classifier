
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
        'no. of municipalities with inhabitants < 499  ': 'muni_under499',
        'no. of municipalities with inhabitants 500-1999': 'muni_500_1999',
        'no. of municipalities with inhabitants 2000-9999 ': 'muni_2000_9999',
        'no. of municipalities with inhabitants >10000 ' : 'muni_over10000',
        'no. of cities ' : 'n_cities',
        'ratio of urban inhabitants ': 'ratio_urban',
        'average salary ': 'avg_salary',
        'unemploymant rate \'95 ' : 'unemployment_rate_95',
        'unemploymant rate \'96 ' : 'unemployment_rate_96',
        'no. of enterpreneurs per 1000 inhabitants ' : 'enterpreneurs_per_1000',
        'no. of commited crimes \'95 ' : 'crimes_95',
        'no. of commited crimes \'96 ': 'crimes_96',
    })

    district['crimes_95'] = district['crimes_95'].astype(int)

    print(district.dtypes)

    district['crimes_95_per_1000'] = district['crimes_95'] / district['population'] * 1000
    district['crimes_96_per_1000'] = district['crimes_96'] / district['population'] * 1000

    district = district.drop(['crimes_95', 'crimes_96'], axis=1)

    return district

def get_loan_account_district_data(remove_non_numeric=False):
    loan_dev, loan_competition = get_loan_data()

    account = pd.read_csv('../data/account.csv', sep=';')
    district = pd.read_csv('../data/district.csv', sep=';')

    district = clean_district_data(district)

    account_district = pd.merge(left=account, right=district, left_on='district_id', right_on='code ')

    dev = pd.merge(left=loan_dev, right=account_district, left_on='account_id', right_on='account_id')
    competition  = pd.merge(left=loan_competition , right=account_district, left_on='account_id', right_on='account_id')

    if remove_non_numeric:
        dev = dev.select_dtypes(['number']).copy()
        competition = competition.select_dtypes(['number']).copy()

    return dev, competition

def main():
    with pd.option_context('display.max_columns', None):
        print(get_loan_account_district_data(remove_non_numeric=True)[0].iloc[[0]])

if __name__ == '__main__':
    main()