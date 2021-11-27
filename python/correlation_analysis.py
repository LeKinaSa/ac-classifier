
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
    sb.heatmap(corr, mask=mask, cmap=cmap, annot=annot, fmt='.01f', center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}).set(title=title)
    plt.show()

#### District
d = data.get_district_data()
correlation_analysis(d)

#### Loan
d, _ = data.get_loan_data()
correlation_analysis(d)

#### Loan + Transaction
t = data.get_improved_transaction_data()
d = pd.merge(left=d, right=t, on='account_id')
correlation_analysis(d) # This one is a little slower to show but it has very interesting results

#### All
# d, _ = data.get_processed_data()
# correlation_analysis(d) # This one is too big and it is not good for analyzing

#### All Processed (Default Processing)
d, _ = data.get_data()
d = d.drop('loan_id', axis=1)
correlation_analysis(d)
correlation_analysis(d, True)

#### All Processed (Default Processing) - Analyzing Correlations By Loan Status
d, _ = data.get_data()
d = d.drop('loan_id', axis=1)
correlation_analysis(d.drop('status', axis=1), True)
p = d.loc[d['status'] == 0].drop('status', axis=1)
n = d.loc[d['status'] == 1].drop('status', axis=1)
correlation_analysis(p, True, 'Correlation Graph (paid loans)')
correlation_analysis(n, True, 'Correlation Graph (non paid loans)')


# TODO: all data with different types of processing in the data

# TODO: test algorithms without random division -> dividing the dev loans by date? - dont use this for submissions
