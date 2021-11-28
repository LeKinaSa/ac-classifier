
import numpy as np
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
import data

def correlation_analysis(df, annot=False, title='Correlation Graph'):
    sb.set_theme(style="white")

    # Compute the correlation matrix
    corr = df.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    plt.subplots(figsize=(11, 9))
    
    # Generate a custom diverging colormap
    cmap = sb.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sb.heatmap(corr, mask=mask, cmap=cmap, annot=annot, fmt='.03f', center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}).set(title=title)
    plt.show()

def main():
    #### District
    # d = data.get_district_data()
    # correlation_analysis(d)

    #### Loan
    # d, _ = data.get_loan_data()
    # correlation_analysis(d)

    #### Loan + Transaction
    # t = data.get_improved_transaction_data()
    # d = pd.merge(left=d, right=t, on='account_id')
    # correlation_analysis(d) # This one is a little slower to show but it has very interesting results

    #### All
    # d, _ = data.get_processed_data()
    # correlation_analysis(d) # This one is too big and it is not good for analyzing

    #### All Processed (Default Processing)

    d, _ = data.get_processed_data()
    
    # Card
    d['type'].fillna('None', inplace=True)
    d['card'] = d['type'].apply(lambda x: 0 if x == 'None' else 1)

    # Normalize crimes and unemployment
    d = data.normalize_dict(d, {
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

    # Drop Ids and Disponent
    d = d.drop([
        'loan_id',
        'account_id', 'client_id_owner', 'client_id_disponent',
        'district_id_account', 'district_id_owner', 'district_id_disponent',
        'code_account', 'code_owner', 'code_disponent', 'disp_id', 'card_id',

        'birthday_disponent', 'gender_disponent', 'name_disponent',
        'region_disponent', 'population_disponent', 'muni_under499_disponent',
        'muni_500_1999_disponent', 'muni_2000_9999_disponent',
        'muni_over10000_disponent', 'n_cities_disponent',
        'ratio_urban_disponent', 'avg_salary_disponent',
        'unemployment_95_disponent', 'unemployment_evolution_disponent',
        'enterpreneurs_per_1000_disponent', 'crimes_95_per_1000_disponent',
        'crimes_evolution_disponent',

        # 'name_owner', 'region_owner', 'population_owner',
        # 'muni_under499_owner', 'muni_500_1999_owner',
        # 'muni_2000_9999_owner', 'muni_over10000_owner',
        # 'n_cities_owner', 'ratio_urban_owner', 'avg_salary_owner',
        # 'unemployment_95_owner', 'unemployment_evolution_owner',
        # 'enterpreneurs_per_1000_owner',
        # 'crimes_95_per_1000_owner', 'crimes_evolution_owner',

        # 'name_account', 'region_account', 'population_account',
        # 'muni_under499_account', 'muni_500_1999_account',
        # 'muni_2000_9999_account', 'muni_over10000_account',
        # 'n_cities_account', 'ratio_urban_account', #'avg_salary_account',
        # 'unemployment_95_account', 'unemployment_evolution_account',
        # 'enterpreneurs_per_1000_account',
        # 'crimes_95_per_1000_account', 'crimes_evolution_account',

        'issued', 'type'
    ], axis=1)

    # correlation_analysis(d)
    # correlation_analysis(d, True)

    # d['amount_div_daily_balance'] = d['amount'] / d['avg_daily_balance']
    # d['amount_div_avg_salary'] = d['amount'] / d['avg_salary_account']
    # d['payments_div_daily_balance'] = d['payments'] / d['avg_daily_balance']
    # d['payments_div_avg_salary'] = d['payments'] / d['avg_salary_account']
    # d = d.drop('avg_salary_account', axis=1)
    # d = d.drop([
        # 'date_loan', 'amount', 'duration', 'payments', 
        # 'birthday_owner', 'gender_owner', 'frequency', 'date_account',
        # 'avg_balance', 'avg_daily_balance', 'balance_deviation',
        # 'balance_distribution_first_quarter', 'balance_distribution_median',
        # 'balance_distribution_third_quarter', 'avg_amount', 'avg_abs_amount',
        # 'n_transactions'
    # ], axis=1)
    correlation_analysis(d, True)

    #### All Processed (Default Processing) - Analyzing Correlations By Loan Status
    # d, _ = data.get_data()
    # d = d.drop('loan_id', axis=1)
    # correlation_analysis(d.drop('status', axis=1), True)
    # p = d.loc[d['status'] == 0].drop('status', axis=1)
    # n = d.loc[d['status'] == 1].drop('status', axis=1)
    # correlation_analysis(p, True, 'Correlation Graph (paid loans)')
    # correlation_analysis(n, True, 'Correlation Graph (non paid loans)')

    # TODO: all data with different types of processing in the data

    # TODO: test algorithms without random division -> dividing the dev loans by date? - dont use this for submissions

if __name__ == '__main__':
    main()
