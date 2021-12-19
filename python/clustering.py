
from sklearn.cluster import DBSCAN, KMeans, MiniBatchKMeans

import data
from data import set_working_directory
import stats

def analysis(d):
    stats.scatter_plot(d, 'balance_distribution_median', 'balance_deviation', 'cluster')
    return

def main():
    d, c = data.get_data()
    d = d.append(c)
    to_cluster = [
        'gender_owner',
        #'muni_under499_account',
        #'muni_500_1999_account',
        #'muni_2000_9999_account',
        #'muni_over10000_account',
        'n_cities_account',
        'ratio_urban_account',
        'avg_salary_account',
        'unemployment_95_account',
        'unemployment_evolution_account',
        'entrepreneurs_per_1000_account',
        'crimes_95_per_1000_account',
        'crimes_evolution_account',
        'avg_balance',
        'avg_daily_balance',
        'balance_deviation',
        'balance_distribution_first_quarter',
        'balance_distribution_median',
        'balance_distribution_third_quarter',
        'high_balance',
        'last_high',
        'last_neg',
        'negative_balance',
        'avg_amount',
        'avg_abs_amount',
        'card',
        #'region_non_paid_partial_account'
    ]
    fit_clustering = data.select(d, to_cluster)
    clustering_techniques = {
        # 'DBSCAN': DBSCAN(eps=5, min_samples=5),
        'Kmeans': KMeans(n_clusters=5, random_state=0),
        #'MiniBatches': MiniBatchKMeans(n_clusters=8, random_state=0, batch_size=200)
    }

    for technique in clustering_techniques:
        clustering_technique = clustering_techniques[technique]
        clustering_technique.fit(fit_clustering)
        labels = clustering_technique.labels_
        d['cluster'] = labels
        analysis(d)
        
    return

if __name__ == '__main__':
    set_working_directory()
    main()
