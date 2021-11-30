
import numpy as np
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
import data

# Global Variables for showing only some graphs
district            = False
loan                = False
loan_and_trans      = False
all                 = False
analyze_by_status   = False
all_with_processing = False

def correlation_analysis(df, annot=False, decimal_places=1, title='Correlation Graph'):
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
    fmt = f'.0{decimal_places}f'
    sb.heatmap(corr, mask=mask, cmap=cmap, annot=annot, fmt=fmt, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}).set(title=title)
    plt.show()

def correlation_analysis_by_status(d, c, annot=False, decimal_places=1):
    d = d.drop('loan_id', axis=1)
    c = c.drop(['loan_id', 'status'], axis=1)

    a = d.drop('status', axis=1)
    correlation_analysis(a, annot, decimal_places, 'Correlation Graph (development data)')
    
    p = d.loc[d['status'] == 0].drop('status', axis=1)
    n = d.loc[d['status'] == 1].drop('status', axis=1)
    correlation_analysis(p, annot, decimal_places, 'Correlation Graph (paid loans)')
    correlation_analysis(n, annot, decimal_places, 'Correlation Graph (non paid loans)')
    
    correlation_analysis(c, annot, decimal_places, 'Correlation Graph (competition data)')

def main():
    #### District
    if district:
        d = data.get_district_data()
        correlation_analysis(d)

    #### Loan
    if loan:
        d, _ = data.get_loan_data()
        correlation_analysis(d)

    #### Loan + Transaction
    if loan_and_trans:
        d, _ = data.get_loan_data()
        t = data.get_improved_transaction_data()
        d = pd.merge(left=d, right=t, on='account_id')
        correlation_analysis(d) # This one is a little slower to show but it has very interesting results

    #### All
    if all:
        d, _ = data.get_processed_data()
        correlation_analysis(d) # This one is too big and it is not good for analyzing

    #### All Processed (Default Processing) - Analyzing Correlations By Loan Status
    if analyze_by_status:
        d, c = data.get_data()
        correlation_analysis_by_status(d, c, True, 1)

    ### All with Some Processing
    if all_with_processing:
        d, _ = data.get_processed_data()
        
        d['type'].fillna('None', inplace=True)
        d['card'] = d['type'].apply(lambda x: 0 if x == 'None' else 1)

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

        d = d.drop([
            'loan_id',
            'account_id', 'client_id_owner', 'client_id_disponent',
            'district_id_account', 'district_id_owner', 'district_id_disponent',
            'code_account', 'code_owner', 'code_disponent', 'disp_id', 'card_id',

            'birthday_disponent', 'gender_disponent',
            'issued', 'type'
        ], axis=1)

        d['avg_salary'] = d['avg_salary_account']
        d = data.drop_district_info(d, 'disponent')
        d = data.drop_district_info(d, 'owner')
        d = data.drop_district_info(d, 'account')

        correlation_analysis(d, True)

        d['amount_div_daily_balance'] = d['amount'] / d['avg_daily_balance']
        d['amount_div_avg_salary'] = d['amount'] / d['avg_salary']
        d['payments_div_daily_balance'] = d['payments'] / d['avg_daily_balance']
        d['payments_div_avg_salary'] = d['payments'] / d['avg_salary']
        d = d.drop([
            'date_loan', 'amount', 'duration', 'payments', 'avg_salary', 'card',
            'birthday_owner', 'gender_owner', 'frequency', 'date_account',
            'avg_balance', 'avg_daily_balance', 'balance_deviation',
            'balance_distribution_first_quarter', 'balance_distribution_median',
            'balance_distribution_third_quarter', 'avg_amount', 'avg_abs_amount',
            'high_balance', 'negative_balance', 'last_high', 'last_neg',
            'n_transactions'
        ], axis=1)
        correlation_analysis(d, True, 3)

    # TODO: all data with different types of processing in the data

    # TODO: test algorithms without random division -> dividing the dev loans by date? - dont use this for submissions
    return

if __name__ == '__main__':
    main()
