import pandas as pd
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import data
from data import set_working_directory
from sklearn.preprocessing import KBinsDiscretizer

################################################################################
##########        Global Variables for showing only some graphs       ##########
################################################################################
# General
status_pie_chart   = False
general_statistics = False
# First Contact with the Data
district_scatter_plots            = False
loan_amounts                      = False
card                              = False
percentage_paid_loans             = False
date                              = False
card_graphs                       = False
transactions_graphs               = False
owners_graphs                     = False
district_client_vs_account        = False
owners_age                        = False
salary_daily_balance              = False
salary_daily_balance_norm         = False
munis_per_district                = False
transactions_amount_and_deviation = False
dev_vs_competition                = True
# Correlation Graphs
district             = False
loan                 = False
loan_and_trans       = False
all_corr             = True
analyze_by_status    = False
all_with_processing  = False
parts                = False
# All Possible Scatter and Count Plots -> keep False
all_possible_scatter = False
all_possible_count   = False
################################################################################

def remove_dups(lst):
    return sorted(set(lst), key=lambda x: lst.index(x))

def percentage_plot(df, group, hue):
    group_counts = df.groupby(group)[hue].value_counts()
    
    keys = group_counts.to_dict().keys()

    order = remove_dups(list(map(lambda x: x[0], keys)))
    hue_order = remove_dups(list(map(lambda x: x[1], keys)))

    df = (df.groupby(group)[hue]
            .value_counts(normalize=True).mul(100)
            .rename('percentage').reset_index())
    graph = sb.catplot(data=df, x=group, y='percentage', kind='bar', 
            hue=hue, order=order, hue_order=hue_order)
    ax = graph.facet_axis(0, 0)
    for i, p in enumerate(ax.patches):
        coords = p.get_x() + p.get_width() / 2, p.get_height() + 2

        value = group_counts.iloc[i]
        ax.annotate(
            '{0:.1f}% ({1})'.format(p.get_height(), value), coords,
            ha='center', color='black', rotation='horizontal',
        )
    return graph

def correlation_analysis(df, annot=False, decimal_places=1, title='Correlation Graph', filename=None, bigger=False):
    sb.set_theme(style="white")

    # Compute the correlation matrix
    corr = df.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    figsize = (11, 9)
    if bigger:
        figsize = (22, 18)
    plt.subplots(figsize=figsize)
    
    # Generate a custom diverging colormap
    cmap = sb.diverging_palette(260, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    fmt = f'.0{decimal_places}f'
    sb.heatmap(corr, mask=mask, cmap=cmap, annot=annot, fmt=fmt, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}).set(title=title)
    
    plt.yticks(rotation=45, ha='right')
    plt.xticks(rotation=45, ha='right')
    
    if filename != None:
        plt.savefig(filename)
    plt.show()

def correlation_analysis_by_status(d, c, annot=False, decimal_places=1):
    d = d.drop('loan_id', axis=1)
    c = c.drop(['loan_id', 'status'], axis=1)

    a = d.drop('status', axis=1)
    correlation_analysis(a, annot, decimal_places, 'Correlation Graph (development data)')
    
    p = d.loc[d['status'] == 0].drop('status', axis=1)
    n = d.loc[d['status'] == 1].drop('status', axis=1)
    correlation_analysis(p, annot, decimal_places, 'Correlation Graph (paid loans)', '../img/paid.png')
    correlation_analysis(n, annot, decimal_places, 'Correlation Graph (non paid loans)', '../img/non_paid.png')
    
    correlation_analysis(c, annot, decimal_places, 'Correlation Graph (competition data)')

def scatter_plot(d, x, y, hue='status'):
    sb.scatterplot(data=d, x=x, y=y, hue=hue)
    plt.show()

def count_plot(d, x, hue='status'):
    sb.histplot(data=d, x=x, hue=hue, multiple='fill')
    plt.show()

def violin_plot(d, x, y):
    sb.violinplot(data=d, x=x, y=y)
    plt.show()

def box_plot(d, x, y):
    sb.boxplot(data=d, x=x, y=y)
    plt.show()

################################################################################

def main(): 
    #### Status Pie Chart
    if status_pie_chart:
        d, _ = data.get_loans_data()
        d = d.groupby('status').size()
        plt.pie(d)
        plt.show()

    #### General Statistics
    if general_statistics:
        dev, comp = data.get_raw_data()
        dev['year'] = dev['date_loan'] // 10000
        comp['year'] = comp['date_loan'] // 10000

        print('--------- DEVELOPMENT ---------')
        print(dev[['amount', 'duration']].describe())
        print(dev.groupby('year')[['amount', 'duration', 'payments']].mean())
        print('\n--------- COMPETITION ---------')
        print(comp[['amount', 'duration']].describe())
        print(comp.groupby('year')[['amount', 'duration', 'payments']].mean())

    #### District Graphs
    if district_scatter_plots:
        dev, _ = data.merge_loan_account_district()
        dev['date_x'] = pd.to_datetime(dev['date_x'].apply(data.get_birthday_from_birth_number))

        dev = data.balance(dev)

        sb.scatterplot(data=dev, x='unemployment_95', y='amount', hue='status')
        plt.show()

        sb.scatterplot(data=dev, x='entrepreneurs_per_1000', y='amount', hue='status')
        plt.show()

        sb.scatterplot(data=dev, x='population', y='amount', hue='status')
        plt.show()

        sb.scatterplot(data=dev, x='crimes_95_per_1000', y='amount', hue='status')
        plt.show()

        sb.scatterplot(data=dev, x='ratio_urban', y='amount', hue='status')
        plt.show()

        sb.histplot(dev, x="amount")
        plt.show()

    #### Loan Amounts, and Distribution of Amounts per Status
    if loan_amounts:
        # Loan Amounts
        dev, _ = data.get_raw_loans_data()
        g = sb.histplot(data=dev, x='amount')
        g.set(
            title='Distribution of Loan Amounts',
            ylabel='Count',
            xlabel='Amount',
        )
        plt.show()

        g = sb.scatterplot(data=dev, x='payments', y='amount', hue='duration')
        g.set(
            xlabel='Payments',
            ylabel='Amount',
        )
        plt.show()

        # Loan Amounts per Status
        g = sb.boxplot(data=dev, x='status', y='amount')
        g.set_xticklabels(['Paid', 'Unpaid'])
        g.set(
            title='Loan Amounts According to Status',
            ylabel='Status',
            xlabel='Amount',
        )
        plt.show()

        # Loan Amounts per District
        sb.boxplot(x='district_id_account', y='amount', data=dev)
        plt.show()

        # Loan Amounts per Region
        sb.boxplot(x='region_account', y='amount', data=dev)
        plt.show()

    # Percentage of Loans Paid per Region and Frequency
    if percentage_paid_loans:
        dev, _ = data.get_raw_loans_data()
        dev['paid'] = dev['status'].apply(lambda x: True if x == 0 else False)

        # Paid Loans per Region
        gb = dev.groupby('region_account')['paid'].value_counts(normalize=True).mul(100).rename('paid_percent').reset_index()
        g = sb.histplot(data=gb, x='region_account', hue='paid', weights='paid_percent', 
                discrete=True, multiple='stack', shrink=0.8, hue_order=[False, True],
                palette=['firebrick', 'forestgreen'])
        g.set(
            title='Percentage of Paid Loans per Region',
            ylabel='Percentage',
            xlabel='Region',
        )
        g.get_legend().set_title('Paid')
        plt.show()

        # Paid Loans per Frequency
        gb = dev.groupby('frequency')['paid'].value_counts(normalize=True).mul(100).rename('paid_percent').reset_index()
        g = sb.histplot(data=gb, x='frequency', hue='paid', weights='paid_percent', 
                discrete=True, multiple='stack', shrink=0.8, hue_order=[False, True],
                palette=['firebrick', 'forestgreen'])
        g.set(
            title='Percentage of Paid Loans per Frequency',
            ylabel='Percentage',
            xlabel='Frequency',
        )
        plt.show()

    #### Date Analysis 
    if date:
        dev, comp = data.get_loan_data()

        sb.kdeplot(x='date', hue='status', data=dev)
        plt.show()

        dev['date_proper'] = pd.to_datetime(dev['date'].apply(lambda x: data.get_readable_date(x)))
        comp['date_proper'] = pd.to_datetime(comp['date'].apply(lambda x: data.get_readable_date(x)))

        plt.hist(data=dev, x='date_proper', alpha=0.5, edgecolor='k', label='Development', bins=10)
        plt.hist(data=comp, x='date_proper', alpha=0.5, edgecolor='k', label='Competition', bins=5)
        plt.title('Loan Date Distribution')
        plt.xlabel('Loan Date')
        plt.ylabel('Count')
        plt.legend()
        plt.show()

    #### Card
    if card_graphs:
        card = data.get_card_data()

        sb.countplot(data=card, x='type', order=['junior', 'classic', 'gold'])
        plt.show()

        sb.countplot(data=card, x='disp_id')
        plt.show()

    #### Transactions
    if transactions_graphs:
        trans = data.get_transactions_data()

        sb.countplot(data=trans, x='type')
        plt.show()

        sb.countplot(data=trans, x='operation')
        plt.show()

        print('Total transactions:', len(trans))
        print('Transactions without a bank:', trans['bank'].isna().sum())
        sb.countplot(data=trans, x='bank')
        plt.show()

        dev, comp = data.get_loan_data()
        sb.countplot(data=dev, x='account_id') # Check the number of loans per account
        plt.show()

    #### Owners
    if owners_graphs:
        owners = data.merge_client_dispowner()
        loan_owner_client_dev, loan_owner_client_comp = data.merge_loan_account_client_dispowner()

        # sb.countplot(data=owners, x='account_id').set(title='Number of Owners per Account')
        # plt.show()

        g = sb.countplot(data=owners, x='gender', palette=['violet', 'deepskyblue'])
        g.bar_label(g.containers[0])
        g.set(
            title='Gender Distribution',
            xlabel='Gender',
            ylabel='Count',
        )
        plt.show()

        for df in [loan_owner_client_dev, loan_owner_client_comp]:
            for row in ['date_loan', 'birthday']:
                df[row] = pd.to_datetime(df[row].apply(data.get_readable_date))

            df['age_loan'] = df.apply(
                lambda row: int((row['date_loan'] - row['birthday']).days / 365.2425),
                axis=1
            )
        
        both = loan_owner_client_dev.append(loan_owner_client_comp, ignore_index=True)
        both['dev'] = ~both['status'].isnull()

        print(both['age_loan'].describe())
        
        g = sb.histplot(data=both, x='age_loan', hue='dev', bins=np.arange(10, 64, 4), alpha=0.4, palette=['tab:blue', 'tab:red'])
        g.set(
            title='Age At Loan Time',
            xlabel='Age',
            ylabel='Count',
        )
        plt.show()

        discretizer = KBinsDiscretizer(n_bins=8, encode='ordinal', strategy='uniform')
        loan_owner_client_dev['age_bin'] = discretizer.fit_transform(
            loan_owner_client_dev['age_loan'].values.reshape(-1, 1))
        
        bin_edges = discretizer.bin_edges_[0].tolist()
        labels = []

        for f, s in zip(bin_edges, bin_edges[1:]):
            labels.append(str(round(f)) + '-' + str(round(s)))

        fig, axes = plt.subplots(1, 2)
        fig.suptitle('Percentage of Paid Loans by Age')
        
        paid = (loan_owner_client_dev.groupby('age_bin')['status'].value_counts(normalize=True).mul(100)
            .rename('paid_percent').reset_index())

        g = sb.countplot(data=loan_owner_client_dev, x='age_bin', ax=axes[0])
        g.set(xlabel='Age Bin', ylabel='Count')
        g.set_xticklabels(labels)
        g.bar_label(g.containers[0])

        g = sb.barplot(data=paid, x='age_bin', y='paid_percent', hue='status', 
            ax=axes[1], palette=['forestgreen', 'firebrick'])
        g.set(xlabel='Age Bin', ylabel='Paid Loans (%)')
        g.set_xticklabels(labels)
        g.get_legend().remove()
        for container in g.containers:
            g.bar_label(container, fmt='%.1f')
        
        plt.show()

        sb.countplot(data=loan_owner_client_dev , x='gender', palette=['violet', 'deepskyblue'])
        plt.show()

        sb.countplot(data=loan_owner_client_comp, x='gender', palette=['violet', 'deepskyblue'])
        plt.show()

        g = sb.countplot(data=loan_owner_client_dev, x='status', hue='gender', palette=['violet', 'deepskyblue'])
        for container in g.containers:
            g.bar_label(container)
        plt.show()

        loan_owner_client_dev['same_district'] = loan_owner_client_dev['district_id_account'] == loan_owner_client_dev['district_id_owner']
        print('District Owner == District Account:', loan_owner_client_dev['same_district'].nunique())

        loan_owner_districts, _ = data.merge_loan_account_client_dispowner_districtaccount_districtowner()
        loan_owner_districts['same_region'] = loan_owner_districts['region_account'] == loan_owner_districts['region_owner']
        print('Region Owner == Region Account:', loan_owner_districts['same_region'].nunique())

    #### Comparison between district owner/disponent and district account
    def check_districts(row):    
        account   = row['name_account'  ]
        owner     = row['name_owner'    ]
        disponent = row['name_disponent']

        if disponent is not str: # NaN
            return account != owner
        if owner != disponent:
            print('owner != disponent')
        return account != owner and account != disponent

    if district_client_vs_account:
        clients = data.get_raw_clients_data()
        clients = data.select(clients, ['name_owner', 'name_disponent', 'name_account'])
        clients['inconsistent'] = clients.apply(check_districts, axis=1)
        print('Clients with the same district as the account:   ', len(clients[clients['inconsistent'] == False]))
        print('Clients with different district from the account:', len(clients[clients['inconsistent'] == True ]))
        #sb.countplot(data=clients, x='inconsistent').set(title='Clients and account have the same district', xlabel='Different Districts')
        #plt.show()
        ax = sb.countplot(data=clients, x='inconsistent', palette=['forestgreen', 'firebrick'])
        for p in ax.patches:
            ax.annotate(str(p.get_height()), (p.get_x() + p.get_width()*0.47, p.get_height() + 0.5))
        plt.title('Clients and account have the same district')
        plt.xlabel('Different Districts')
        plt.show()

    #### Owners Age
    if owners_age:
        d, c = data.get_raw_loans_data()
        loans = pd.concat([d, c], verify_integrity=True, ignore_index=True)
        print('Integrity Verification:', len(d) + len(c), len(loans))

        # Birthday -> Age
        loans = data.get_ages(loans, ['birthday_owner'], 'date_loan')
        loans = loans.rename(columns={'birthday_owner' : 'age_owner'})
        
        loans = data.select(loans, ['age_owner', 'gender_owner', 'name_owner', 'name_disponent', 'name_account'])
        loans['inconsistent'] = loans.apply(check_districts, axis=1)
        
        false_count = len(loans[loans['inconsistent'] == False])
        true_count  = len(loans[loans['inconsistent'] == True ])
        print('Loans from clients with the same district as the account:   ', false_count)
        print('Loans from clients with different district from the account:', true_count )
        
        g = sb.violinplot(data=loans, x='inconsistent', y='age_owner', hue='gender_owner', \
                          split=True, scale='count', scale_hue=False, linewidth=0.4, cut=0)
        g.set_xticklabels([f'False ({false_count})', f'True ({true_count})'])
        plt.title('Clients and account have the same district')
        plt.xlabel('Different Districts')
        plt.ylabel('Owner\'s Age')
        plt.legend(title='Owner\'s Gender', loc='upper right')
        plt.show()

    # Salary and Daily Balance
    if salary_daily_balance:
        dev, _ = data.get_raw_loans_data()
        sb.scatterplot(data=dev, x='avg_daily_balance', y='avg_salary_account', hue='status').set(title='Salary and Balance Comparison', xlabel='Average Daily Balance', ylabel='Average Salary')
        plt.xscale('log')
        plt.yscale('log')
        plt.show()

    # Salary and Daily Balance Normalized
    if salary_daily_balance_norm:
        dev, _ = data.get_loans_data()
        sb.scatterplot(data=dev, x='avg_daily_balance', y='avg_salary_account', hue='status').set(title='Salary and Balance Comparison', xlabel='Average Daily Balance', ylabel='Average Salary')
        plt.xscale('log')
        plt.yscale('log')
        plt.show()

    # Districts
    if munis_per_district:
        district_data = data.get_district_data()
        district_data = data.normalize_district(district_data, '')
        d = district_data.groupby('region').mean().reset_index()
        d = data.select(d, ['region', 'muni_under499', 'muni_500_1999', 'muni_2000_9999', 'muni_over10000'])
        sb.set()
        d.set_index('region')\
            .reindex(d.set_index('region').sum().index, axis=1)\
            .plot(kind='bar', stacked=True,
                figsize=(11,8)).set(title='Region')
        plt.show()

    #### Transactions Amount and Deviation
    if transactions_amount_and_deviation:
        dev, _ = data.get_raw_loans_data()
        sb.scatterplot(data=dev, x='avg_amount', y='balance_deviation', hue='status').set(title='Transaction Amount and Balance Deviation Comparison', xlabel='Transaction Amount', ylabel='Balance Deviation')
        plt.xscale('log')
        plt.yscale('log')
        plt.show()

    #### Comparison between development and competition datasets
    if dev_vs_competition:
        d, c = data.get_raw_loans_data()
        d['data'] = 'Development'
        c['data'] = 'Competition'
        loans = d.append(c, ignore_index=True)

        g = sb.boxplot(data=loans, x='data', y='amount')
        g.set(xlabel='Dataset', ylabel='Loan Amount', title='Loan Amount Distribution per Dataset')
        plt.show()

    #### District (Correlation)
    if district:
        d = data.get_district_data()
        correlation_analysis(d)

    #### Loan (Correlation)
    if loan:
        d, _ = data.get_loan_data()
        correlation_analysis(d)

    #### Loan + Transaction (Correlation)
    if loan_and_trans:
        d, _ = data.get_loan_data()
        t = data.get_improved_transaction_data()
        d = pd.merge(left=d, right=t, on='account_id')
        correlation_analysis(d) # This one is a little slower to show but it has very interesting results

    #### All (Correlation)
    if all_corr:
        d, _ = data.get_loans_data()
        d = d.dropna(axis=1, how='any')
        to_drop = [
            'frequency', 'disponent', 'gender_disponent', 'loan_id', 'date_loan',
            'region_non_paid_partial_owner', 'region_non_paid_partial_account',
        ]
        d = d.drop(to_drop, axis=1)
        columns = list(d.columns)
        (columns[0], columns[1]) = (columns[1], columns[0])
        d = data.select(d, columns)
        correlation_analysis(d) # This one is too big and it is not good for analyzing

    #### All Processed (Default Processing) - Analyzing Correlations By Loan Status
    if analyze_by_status:
        d, c = data.get_loans_data()
        selected = [
            'loan_id', 'status',
            'duration',
            'gender_owner',
            'balance_deviation', 'balance_distribution_first_quarter',
            'card', 'high_balance', 'last_neg',
        ]
        (d, c) = (data.select(d, selected), data.select(c, selected))
        correlation_analysis_by_status(d, c, True, 1)

    #### All with Some Processing (Correlation)
    if all_with_processing:
        d, _ = data.get_raw_loans_data()
        
        d['type'].fillna('None', inplace=True)
        d['card'] = d['type'].apply(lambda x: 0 if x == 'None' else 1)

        d = data.get_evolution_values(d)

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
    
    #### All in Parts (Correlation)
    if parts:
        d, _ = data.get_loans_data()
        d = d.drop(['loan_id'], axis=1)
        columns = list(d.drop('status', axis=1).columns)
        for i in range(0, len(columns), 5):
            to_show = columns[i:i+5]
            s = data.select(d, to_show)
            correlation_analysis(s, True)

    #### All possible scatter plots
    if all_possible_scatter:
        d, _ = data.get_raw_loans_data()
        d, _ = data.get_loans_data()
        columns = d.drop(['loan_id', 'status'], axis=1).columns
        n = len(list(combinations(columns, 2)))
        answer = input(f'Show all possible scatter plots ({n}) [y/N]? ').lower()
        if answer == 'y' or answer == 'yes':
            for x, y in combinations(columns, 2):
                scatter_plot(d, x, y)
    
    #### All possible count plots
    if all_possible_count:
        d, _ = data.get_raw_loans_data()
        d, _ = data.get_loans_data()
        columns = d.drop(['loan_id', 'status'], axis=1).columns
        n = len(columns)
        answer = input(f'Show all possible count plots ({n}) [y/N]? ').lower()
        if answer == 'y' or answer == 'yes':
            for x in columns:
                count_plot(d, x)

    return

if __name__ == '__main__':
    set_working_directory()
    main()
