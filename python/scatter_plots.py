from seaborn.relational import scatterplot
import data
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt


def scatter_plot(d, x, y):
    sb.scatterplot(data=d, x=x, y=y, hue='status')
    #plt.get_current_fig_manager().state('zoomed')
    plt.show()

def count_plot(d, x):
    sb.histplot(data=d, x=x, hue='status', multiple='fill')
    plt.show()

#d_maj = dev[dev['status'] == 0]
#d_min = dev[dev['status'] == 1]

#d_under = d_maj.sample(len(d_min.index), random_state=0)
#dev = pd.concat([d_under, d_min])

def normalize_dict(df, dict):
    for k, v in dict.items():
        df[k] = df[k].divide(df[v])
    return df

dev, _ = data.get_processed_data()

dev['date_loan'] = pd.to_datetime(dev['date_loan'].apply(data.get_readable_date))
dev['birthday_owner'] = pd.to_datetime(dev['birthday_owner'].apply(data.get_readable_date))
dev['date_account'] = pd.to_datetime(dev['date_account'].apply(data.get_readable_date))

dev = normalize_dict(dev, {
    'unemployment_96_account'      : 'unemployment_95_account',
    'unemployment_96_owner'        : 'unemployment_95_owner',
    'unemployment_96_disponent'    : 'unemployment_95_disponent',
    'crimes_96_per_1000_account'   : 'crimes_95_per_1000_account',
    'crimes_96_per_1000_owner'     : 'crimes_95_per_1000_owner',
    'crimes_96_per_1000_disponent' : 'crimes_95_per_1000_disponent',
})
dev = dev.rename(columns={
    'unemployment_96_account'      : 'unemployment_evolution_account',
    'unemployment_96_owner'        : 'unemployment_evolution_owner',
    'unemployment_96_disponent'    : 'unemployment_evolution_disponent',
    'crimes_96_per_1000_account'   : 'crimes_evolution_account',
    'crimes_96_per_1000_owner'     : 'crimes_evolution_owner',
    'crimes_96_per_1000_disponent' : 'crimes_evolution_disponent',
})

dev = dev.drop([
    'loan_id',
    'account_id', 'client_id_owner', 'client_id_disponent',
    'district_id_account', 'district_id_owner', 'district_id_disponent',
    'code_account', 'code_owner', 'code_disponent', 'disp_id', 'card_id'
], axis=1)

dev = dev.drop([
    'birthday_disponent', 'gender_disponent', 'name_disponent',
    'region_disponent', 'population_disponent', 'muni_under499_disponent',
    'muni_500_1999_disponent', 'muni_2000_9999_disponent',
    'muni_over10000_disponent', 'n_cities_disponent',
    'ratio_urban_disponent', 'avg_salary_disponent',
    'unemployment_95_disponent', 'unemployment_evolution_disponent',
    'enterpreneurs_per_1000_disponent', 'crimes_95_per_1000_disponent',
    'crimes_evolution_disponent'
], axis=1)

def has_card(x):
    if x == 'junior' or x == 'classic' or x == 'gold':
        return 1
    return 0

dev['card'] = dev['type'].apply(has_card)
dev = dev.drop(['type', 'issued'], axis=1)

columns = list(dev.columns)
columns.remove('status')
# Already Checked and Good
# columns.remove('card')
# columns.remove('type')
# columns.remove('issued')

# Already Checked
# for c in ['date_loan', 'amount', 'duration', 'payments', 'birthday_owner', 'gender_owner', 'frequency']:
#     columns.remove(c)

# Useless
# for c in ['population_account', 'muni_under499_account', 'muni_500_1999_account', 'muni_2000_9999_account', 'muni_over10000_account', 'n_cities_account']:
#     columns.remove(c)
# for c in ['population_owner', 'muni_under499_owner', 'muni_500_1999_owner', 'muni_2000_9999_owner', 'muni_over10000_owner', 'n_cities_owner']:
#     columns.remove(c)

print(len(columns))
#print(columns)


# for i in range(len(columns)):
#     x = columns[i]
#     for j in range(i + 1, len(columns)):
#         y = columns[j]
#         scatter_plot(dev, x, y)

for c in columns:
    count_plot(dev, c)


# scatter_plot(dev, 'unemployment_95_account', 'amount')
# scatter_plot(dev, 'unemployment_evolution_account', 'amount')

# scatter_plot(dev, 'avg_daily_balance', 'payments')
# dev['avg_daily_balance_norm'] = dev['payments'].divide(dev['avg_daily_balance'])
# scatter_plot(dev, 'avg_daily_balance_norm', 'amount')

# scatter_plot(dev, 'payments', 'avg_salary_account')
# scatter_plot(dev, 'payments', 'avg_salary_owner')

# scatter_plot(dev, 'enterpreneurs_per_1000_account', 'amount')
# scatter_plot(dev, 'population_account', 'amount')
# scatter_plot(dev, 'crimes_95_per_1000_account', 'amount')
# scatter_plot(dev, 'ratio_urban_account', 'amount')
