import pandas as pd
import seaborn
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
    graph = seaborn.catplot(data=df, x=group, y='percentage', kind='bar', 
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

def percentage_plot(df, group, hue):
    group_counts = df.groupby(group)[hue].value_counts()
    
    keys = group_counts.to_dict().keys()

    order = remove_dups(list(map(lambda x: x[0], keys)))
    hue_order = remove_dups(list(map(lambda x: x[1], keys)))

    df = (df.groupby(group)[hue]
            .value_counts(normalize=True).mul(100)
            .rename('percentage').reset_index())
    graph = seaborn.catplot(data=df, x=group, y='percentage', kind='bar', 
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

dev_m = dev.copy()
dev_m['date_x'] = pd.to_datetime(dev_m['date_x'].apply(data.get_birthday_from_birth_number))

d_maj = dev_m[dev_m['status'] == 0]
d_min = dev_m[dev_m['status'] == 1]

d_under = d_maj.sample(len(d_min.index), random_state=0)
dev_m = pd.concat([d_under, d_min])

seaborn.scatterplot(data=dev_m, x='unemployment_95', y='amount', hue='status')
plt.show()

seaborn.scatterplot(data=dev_m, x='enterpreneurs_per_1000', y='amount', hue='status')
plt.show()

seaborn.scatterplot(data=dev_m, x='population', y='amount', hue='status')
plt.show()

seaborn.scatterplot(data=dev_m, x='crimes_95_per_1000', y='amount', hue='status')
plt.show()

seaborn.scatterplot(data=dev_m, x='ratio_urban', y='amount', hue='status')
plt.show()

seaborn.histplot(dev, x="amount")
plt.show()

seaborn.histplot(dev, x="status")
plt.show()

seaborn.boxplot(x="district_id", y="amount", data=dev)
plt.show()

seaborn.boxplot(x="region", y="amount", data=dev)
plt.show()

seaborn.displot(dev, x="region", hue="status", multiple="dodge")
plt.show()

seaborn.displot(dev, x="frequency", hue="status", multiple="dodge")
plt.show()

seaborn.displot(dev, x="amount", hue="status", hue_order=[0, 1], multiple="dodge")
plt.show()

seaborn.kdeplot(x="date_x", hue="status", data=dev)
plt.show()

dev['frequency_percent'] = dev.groupby('status')['frequency'].apply(lambda x: 100*x/x.sum())
print(len(dev[dev['frequency'] == 'weekly issuance']))
print(dev.groupby(['frequency', 'status']).sum())

df = dev.groupby('frequency')['status'].value_counts(normalize=True).mul(100).rename('percentage').reset_index()
seaborn.displot(data=df, x='frequency')
plt.show()

percentage_plot(dev, 'frequency', 'status')
plt.show()

dev['frequency_percent'] = dev.groupby('status')['frequency'].apply(lambda x: 100*x/x.sum())
print(len(dev[dev['frequency'] == 'weekly issuance']))
print(dev.groupby(['frequency', 'status']).sum())

df = dev.groupby('frequency')['status'].value_counts(normalize=True).mul(100).rename('percentage').reset_index()
seaborn.displot(data=df, x='frequency')
plt.show()

percentage_plot(dev, 'frequency', 'status')
plt.show()

percentage_plot(dev, 'region', 'status')
plt.show()

seaborn.displot(data=dev, x='region', hue='status', hue_order=[0,1], multiple='fill')
plt.show()

seaborn.kdeplot(x="date_x", hue="status", data=comp)
plt.show()

### Card ###

card_dev, _ = data.get_card_data()

seaborn.countplot(data=card_dev, x="type", order=["junior", "classic", "gold"])
plt.show()

### Transactions ###

trans_dev, _ = data.get_transactions_data()

seaborn.countplot(data=trans_dev, x="type")
plt.show()

seaborn.countplot(data=trans_dev, x="operation")
plt.show()

print("Total transactions:", len(trans_dev))
print("Transactions without a bank:", trans_dev["bank"].isna().sum())
seaborn.countplot(data=trans_dev, x="bank")
plt.show()
