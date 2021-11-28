from correlation_analysis import correlation_analysis
from data import get_processed_data
import pandas as pd

def normalize(df, columns, normalizer):
    for column in columns:
        df[column + "_norm"] = df[column].divide(df[normalizer])
    return df

def normalize_dict(df, dict):
    for k, v in dict.items():
        df[k + "_norm"] = df[k].divide(df[v])
    return df

def get_ages(df, creation_dates, loan_date):
    for creation_date in creation_dates:
        df[creation_date] = df[creation_date].rsub(df[loan_date])
        df[creation_date] = df[creation_date].floordiv(10000)
    return df

def normalize_district(df, muni_under_499, muni_500_1999, muni_2000_9999, muni_over10000, n_cities, s):
    # Obtain total municipalities
    df['total_muni_' + s] = 0
    for column in [muni_under_499, muni_500_1999, muni_2000_9999, muni_over10000]:
        df['total_muni_' + s] = df['total_muni_' + s].add(df[column])
    
    # Normalize municipalities
    for column in [muni_under_499, muni_500_1999, muni_2000_9999, muni_over10000, n_cities]:
        df[column + "_norm"] = df[column].divide(df['total_muni_' + s])

    return df

def process_data(d):
    d = d.drop([
        'loan_id',
        'account_id', 'client_id_owner', 'client_id_disponent',
        'district_id_account', 'district_id_owner', 'district_id_disponent',
        'code_account', 'code_owner', 'code_disponent', 'disp_id', 'card_id'
    ], axis=1)

    d = normalize(d, [
        'avg_amount', 'avg_balance',
        'avg_daily_balance', 'balance_deviation', 'balance_distribution_first_quarter',
        'balance_distribution_median', 'balance_distribution_third_quarter',

        'avg_salary_account', 'avg_salary_owner', 'avg_salary_disponent'
    ], 'payments')

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

    d['type'].fillna('None', inplace=True)

    d = get_ages(d, ['date_account', 'issued', 'birthday_owner', 'birthday_disponent'], 'date_loan')
    d = d.rename(columns={
        'date_account' : 'age_account',
        'issued' : 'age_card',
        'birthday_owner' : 'age_owner',
        'birthday_disponent' : 'age_disponent'
    })
    
    d = d.drop('date_loan', axis=1)
    
    d = d.drop([
        'gender_disponent', 'age_disponent',
        'name_disponent', 'region_disponent', 'population_disponent',
        'muni_under499_disponent', 'muni_500_1999_disponent',
        'muni_2000_9999_disponent', 'muni_over10000_disponent',
        'n_cities_disponent', 'ratio_urban_disponent',
        'avg_salary_disponent', 'unemployment_95_disponent',
        'unemployment_evolution_disponent', 'enterpreneurs_per_1000_disponent',
        'crimes_95_per_1000_disponent', 'crimes_evolution_disponent',
        'unemployment_96_disponent_norm', 'crimes_96_per_1000_disponent_norm',
        'avg_salary_disponent_norm',
    ], axis=1)

    d = normalize_district(d, 'muni_under499_owner', 'muni_500_1999_owner',
            'muni_2000_9999_owner', 'muni_over10000_owner', 'n_cities_owner', 'owner')
    d = normalize_district(d, 'muni_under499_account', 'muni_500_1999_account',
            'muni_2000_9999_account', 'muni_over10000_account', 'n_cities_account', 'account')
 
    return d

def shown(d, columns):
    new = pd.DataFrame()
    new['status'] = d['status']
    for c in columns:
        new[c] = d[c]
    return new

d, _ = get_processed_data()
d = process_data(d)

columns = [
    [
        'name_account',
        'region_account',
        'population_account',
        'muni_under499_account',
        'muni_500_1999_account',
        'muni_2000_9999_account',
        'muni_over10000_account',
        'n_cities_account',
        'ratio_urban_account',
        'avg_salary_account',
        'unemployment_95_account',
        'unemployment_evolution_account',
        'enterpreneurs_per_1000_account',
        'crimes_95_per_1000_account',
        'crimes_evolution_account',
    ], [
        'name_owner',
        'region_owner',
        'population_owner',
        'muni_under499_owner',
        'muni_500_1999_owner',
        'muni_2000_9999_owner',
        'muni_over10000_owner',
        'n_cities_owner',
        'ratio_urban_owner',
        'avg_salary_owner',
        'unemployment_95_owner',
        'unemployment_evolution_owner',
        'enterpreneurs_per_1000_owner',
        'crimes_95_per_1000_owner',
        'crimes_evolution_owner',
    ], [
        'unemployment_96_account_norm',
        'unemployment_96_owner_norm',
        'crimes_96_per_1000_account_norm',
        'crimes_96_per_1000_owner_norm',
        'total_muni_owner',
        'muni_under499_owner_norm',
        'muni_500_1999_owner_norm',
        'muni_2000_9999_owner_norm',
        'muni_over10000_owner_norm',
        'n_cities_owner_norm',
        'total_muni_account',
        'muni_under499_account_norm',
        'muni_500_1999_account_norm',
        'muni_2000_9999_account_norm',
        'muni_over10000_account_norm',
        'n_cities_account_norm',
    ], [
        'avg_balance',
        'avg_daily_balance',
        'balance_deviation',
        'balance_distribution_first_quarter',
        'balance_distribution_median',
        'balance_distribution_third_quarter',
        'avg_amount',
        'avg_abs_amount',
        'n_transactions',
        'avg_amount_norm',
        'avg_balance_norm',
        'avg_daily_balance_norm',
        'balance_deviation_norm',
        'balance_distribution_first_quarter_norm',
        'balance_distribution_median_norm',
        'balance_distribution_third_quarter_norm',
    ], [
        'type',
        'age_card',
        'amount',
        'duration',
        'payments',
        'age_owner',
        'gender_owner',
        'frequency',
        'age_account',
        'avg_salary_account_norm',
        'avg_salary_owner_norm',
    ]
]

for to_show in columns:
    print(len(to_show))
    c = shown(d, to_show)
    correlation_analysis(c, True)

# for part in divide_data(d):
#     correlation_analysis(part)
