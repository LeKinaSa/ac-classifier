
from sklearn.cluster import DBSCAN, KMeans, MiniBatchKMeans

import seaborn as sb
import matplotlib.pyplot as plt

import data
from data import set_working_directory

def main():
    d, c = data.get_data()
    d = d.append(c).drop('frequency', axis=1)
    
    clustering_groups = {
        'client': ['gender_owner', 'age_owner'],
        'account': ['card', 'disponent'],
        'district': ['ratio_urban_account', 'avg_salary_account', 'crimes_95_per_1000_account', 'unemployment_95_account', 'entrepreneurs_per_1000_account'],
        'balance': ['avg_daily_balance', 'balance_deviation', 'high_balance', 'negative_balance'],
        'all': None,
    }
    clustering_techniques = {
        #'DBSCAN': DBSCAN(eps=5, min_samples=5),
        'Kmeans': KMeans(n_clusters=5, random_state=0),
        #'MiniBatches': MiniBatchKMeans(n_clusters=8, random_state=0, batch_size=200)
    }

    for group in clustering_groups:
        to_cluster = clustering_groups[group]
        fit_clustering = d
        if to_cluster != None:
            fit_clustering = data.select(d, to_cluster)
        for technique in clustering_techniques:
            clustering_technique = clustering_techniques[technique]
            cluster = f'{group} - {technique}'
            print(cluster)
            
            # Apply Clustering
            clustering_technique.fit(fit_clustering)
            labels = clustering_technique.labels_
            d['cluster'] = labels

            # Show Results of Clustering
            #print(cluster)
    return

if __name__ == '__main__':
    set_working_directory()
    main()
