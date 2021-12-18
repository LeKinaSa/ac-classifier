import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import data
from correlation_analysis import scatter_plot

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

        value = group_counts.iloc[i] # TODO
        ax.annotate(
            '{0:.1f}% ({1})'.format(p.get_height(), value), coords,
            ha='center', color='black', rotation='horizontal',
        )
    return graph

dev, comp = data.get_loan_account_district_data()

print(dev.nunique())
print(dev.dtypes)

print(dev.date_x.head())

g = sb.histplot(data=dev, x='amount')
g.set(
    title='Distribution of Loan Amounts',
    ylabel='Count',
    xlabel='Amount',
)
#plt.bar_label(g.containers[0]) # TODO
plt.show()

g = sb.boxplot(data=dev, x='status', y='amount')
g.set_xticklabels(['Paid', 'Unpaid'])
g.set(
    title='Loan Amounts According to Status',
    ylabel='Status',
    xlabel='Amount',
)
plt.show()

# sb.histplot(data=dev, x='status')
# plt.show()

# sb.boxplot(x='district_id', y='amount', data=dev)
# plt.show()

sb.boxplot(x='region', y='amount', data=dev)
plt.show()

df = dev.copy()
df['paid'] = df['status'].apply(lambda x: True if x == 0 else False)

gb = df.groupby('region')['paid'].value_counts(normalize=True).mul(100).rename('paid_percent').reset_index()
g = sb.histplot(data=gb, x='region', hue='paid', weights='paid_percent', 
        discrete=True, multiple='stack', shrink=0.8, hue_order=[False, True],
        palette=['firebrick', 'forestgreen'])
g.set(
    title='Percentage of Paid Loans per Region',
    ylabel='Percentage',
    xlabel='Region',
)
g.get_legend().set_title('Paid')
#plt.bar_label(g.containers[0]) # TODO
#sb.displot(data=dev, x='region', hue='status', multiple='dodge')
plt.show()

gb = df.groupby('frequency')['paid'].value_counts(normalize=True).mul(100).rename('paid_percent').reset_index()
g = sb.histplot(data=gb, x='frequency', hue='paid', weights='paid_percent', 
        discrete=True, multiple='stack', shrink=0.8, hue_order=[False, True],
        palette=['firebrick', 'forestgreen'])
g.set(
    title='Percentage of Paid Loans per Frequency',
    ylabel='Percentage',
    xlabel='Frequency',
)
#sb.displot(data=dev, x='frequency', hue='status', multiple='dodge')
plt.show()

sb.displot(data=dev, x='amount', hue='status', hue_order=[0, 1], multiple='dodge')
plt.show()

# sb.kdeplot(x='date_x', hue='status', data=dev)
# plt.show()

#dev['frequency_percent'] = dev.groupby('status')['frequency'].apply(lambda x: 100*x/x.sum())
print(len(dev[dev['frequency'] == 'weekly issuance']))
print(dev.groupby(['frequency', 'status']).sum())

df = dev.groupby('frequency')['status'].value_counts(normalize=True).mul(100).rename('percentage').reset_index()
sb.displot(data=df, x='frequency')
plt.show()

percentage_plot(dev, 'frequency', 'status')
plt.show()

g = percentage_plot(dev, 'region', 'status')
g.set(
    title='Percentage of Paid Loans per Region',
    ylabel='Percentage',
    xlabel='Region',
)
plt.show()

sb.displot(data=dev, x='region', hue='status', hue_order=[0,1], multiple='fill')
plt.show()

dev['date_proper'] = pd.to_datetime(dev['date_x'].apply(lambda x: data.get_readable_date(x)))
comp['date_proper'] = pd.to_datetime(comp['date_x'].apply(lambda x: data.get_readable_date(x)))

# sb.kdeplot(x='date_x', hue='status', data=comp) # TODO: see
plt.hist(data=dev, x='date_proper', alpha=0.5, edgecolor='k', label='Development', bins=10)
plt.hist(data=comp, x='date_proper', alpha=0.5, edgecolor='k', label='Competition', bins=5)
plt.title('Loan Date Distribution')
plt.xlabel('Loan Date')
plt.ylabel('Count')
plt.legend()
plt.show()

### Card ###

# card_dev, _ = data.get_card_data()

# sb.countplot(data=card_dev, x='type', order=['junior', 'classic', 'gold'])
# plt.show()

# sb.countplot(data=card_dev, x='disp_id')
# plt.show()

### Transactions ###

trans = data.get_transactions_data()

sb.countplot(data=trans, x='type')
plt.show()

sb.countplot(data=trans, x='operation')
plt.show()

print('Total transactions:', len(trans))
print('Transactions without a bank:', trans['bank'].isna().sum())
sb.countplot(data=trans, x='bank')
plt.show()

sb.countplot(data=dev, x='account_id') # Check the number of loans per account
plt.show()

## Owners ###
owners = data.get_account_owner_data()
loan_owner_client_dev, loan_owner_client_comp = data.get_loan_client_owner_data()

sb.countplot(data=owners, x='account_id').set(title='Number of Owners per Account')
plt.show()

sb.countplot(data=owners, x='gender')
plt.show()

sb.countplot(data=loan_owner_client_dev , x='gender')
plt.show()

sb.countplot(data=loan_owner_client_comp, x='gender')
plt.show()

sb.countplot(data=loan_owner_client_dev, x='status', hue='gender')
plt.show()

loan_owner, _ = data.get_loan_client_owner_data()
loan_owner['same_district'] = loan_owner['district_id_account'] == loan_owner['district_id_owner']
print(loan_owner['same_district'].nunique())

loan_owner_districts, _ = data.get_loan_client_owner_district_data()
loan_owner_districts['same_region'] = loan_owner_districts['region_account'] == loan_owner_districts['region_owner']
print(loan_owner_districts['same_region'].nunique())

# Salary and Daily Balance
dev, _ = data.get_processed_data()
sb.scatterplot(data=dev, x='avg_daily_balance', y='avg_salary_account', hue='status').set(title='Salary and Balance Comparison', xlabel='Average Daily Balance', ylabel='Average Salary')
plt.xscale('log')
plt.yscale('log')
plt.show()

# Salary and Daily Balance Normalized
dev, _ = data.get_data()
sb.scatterplot(data=dev, x='avg_daily_balance', y='avg_salary_account', hue='status').set(title='Salary and Balance Comparison', xlabel='Average Daily Balance', ylabel='Average Salary')
plt.xscale('log')
plt.yscale('log')
plt.show()

# Districts
district = data.get_district_data()
district = data.normalize_district(district, 'muni_under499', 'muni_500_1999', 'muni_2000_9999', 'muni_over10000', 'n_cities')
d = district.groupby('region').mean().reset_index()
d = data.select(d, ['region', 'muni_under499', 'muni_500_1999', 'muni_2000_9999', 'muni_over10000'])
sb.set()
d.set_index('region')\
    .reindex(d.set_index('region').sum().index, axis=1)\
    .plot(kind='bar', stacked=True,
        figsize=(11,8)).set(title='Region')
plt.show()

# Transactions
dev, _ = data.get_processed_data()
sb.scatterplot(data=dev, x='avg_amount', y='balance_deviation', hue='status').set(title='Transaction Amount and Balance Deviation Comparison', xlabel='Transaction Amount', ylabel='Balance Deviation')
plt.xscale('log')
plt.yscale('log')
plt.show()
