import pandas as pd
import seaborn
import matplotlib.pyplot as plt
import data

dev, _ = data.get_loan_account_district_data()

print(dev.nunique())
print(dev.dtypes)

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
