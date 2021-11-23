import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import data

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

sb.histplot(data=dev, x='amount')
plt.show()

sb.boxplot(data=dev, x='amount')
plt.show()

sb.histplot(data=dev, x='status')
plt.show()

sb.boxplot(x='district_id', y='amount', data=dev)
plt.show()

sb.boxplot(x='region', y='amount', data=dev)
plt.show()

sb.displot(data=dev, x='region', hue='status', multiple='dodge')
plt.show()

sb.displot(data=dev, x='frequency', hue='status', multiple='dodge')
plt.show()

sb.displot(data=dev, x='amount', hue='status', hue_order=[0, 1], multiple='dodge')
plt.show()

sb.kdeplot(x='date_x', hue='status', data=dev)
plt.show()

#dev['frequency_percent'] = dev.groupby('status')['frequency'].apply(lambda x: 100*x/x.sum())
print(len(dev[dev['frequency'] == 'weekly issuance']))
print(dev.groupby(['frequency', 'status']).sum())

df = dev.groupby('frequency')['status'].value_counts(normalize=True).mul(100).rename('percentage').reset_index()
sb.displot(data=df, x='frequency')
plt.show()

percentage_plot(dev, 'frequency', 'status')
plt.show()

percentage_plot(dev, 'region', 'status')
plt.show()

sb.displot(data=dev, x='region', hue='status', hue_order=[0,1], multiple='fill')
plt.show()

sb.kdeplot(x='date_x', hue='status', data=comp) # TODO: see
plt.show()

### Card ###

card_dev, _ = data.get_card_data()

sb.countplot(data=card_dev, x='type', order=['junior', 'classic', 'gold'])
plt.show()

sb.countplot(data=card_dev, x='disp_id')
plt.show()

### Transactions ###

trans_dev, _ = data.get_transactions_data()

sb.countplot(data=trans_dev, x='type')
plt.show()

sb.countplot(data=trans_dev, x='operation')
plt.show()

print('Total transactions:', len(trans_dev))
print('Transactions without a bank:', trans_dev['bank'].isna().sum())
sb.countplot(data=trans_dev, x='bank')
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
