import pandas as pd
import seaborn
import matplotlib.pyplot as plt
import data

dev = pd.read_csv('../data/loan_train.csv', sep=';')

seaborn.histplot(dev, x="amount")
plt.show()

seaborn.histplot(dev, x="status")
plt.show()

lda = data.get_loan_account_district_data()

seaborn.boxplot(x="district_id", y="amount", data=lda[0])
plt.show()