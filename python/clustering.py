
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.model_selection import ParameterGrid

from sklearn.preprocessing import StandardScaler

import data
from data import set_working_directory
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def silhouette_plot(X, labels, n_clusters):
    avg = silhouette_score(X, labels)
    print(f'{n_clusters} clusters: {round(avg, 4)}')

    _, ax = plt.subplots()
    ax.set_xlim([-0.1, 1])
    ax.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    samples = silhouette_samples(X, labels)

    y_lower = 10
    for i in range(n_clusters):
        cluster_values = samples[labels == i]
        cluster_values.sort()

        cluster_size = cluster_values.shape[0]
        y_upper = y_lower + cluster_size
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            cluster_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.8,
        )

        ax.text(-0.05, y_lower + 0.5 * cluster_size, str(i))
        y_lower = y_upper + 10

    ax.set_yticks([])
    ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    ax.axvline(x=avg, color='red', linestyle='--')
    ax.set_title(f'Silhouette Plot ({n_clusters} Clusters)')
    ax.set_xlabel('Silhouette Coefficient Values')
    ax.set_ylabel('Cluster')

    plt.show()

def analysis_client(d):
    gb = d.groupby('cluster')[['age', 'avg_daily_balance', 'avg_salary']].mean().reset_index().sort_values(['age'])

    fig, axes = plt.subplots(1, 2)
    fig.suptitle('Simple Client Clustering')

    g = sb.boxplot(data=d, x='cluster', y='age', order=gb['cluster'], ax=axes[0])
    g.set(xlabel='Cluster', ylabel='Age')

    g = sb.countplot(data=d, x='cluster', hue='gender', ax=axes[1],
        palette=['dodgerblue', 'violet'], order=gb['cluster'])
    g.set(xlabel='Cluster', ylabel='Count')
    g.legend(['Male', 'Female'], loc='lower right')
    for container in g.containers:
        g.bar_label(container)

    plt.show()

def analysis_demographic(d):
    fig, axes = plt.subplots(2, 3)
    fig.suptitle('Demographic Clustering')

    g = sb.boxplot(data=d, x='cluster', y='avg_salary', ax=axes[0, 0])
    g.set(xlabel='Cluster', ylabel='Average Salary')

    g = sb.boxplot(data=d, x='cluster', y='unemployment_96', ax=axes[0, 1])
    g.set(xlabel='Cluster', ylabel='Unemployment Ratio (1996)')

    g = sb.boxplot(data=d, x='cluster', y='ratio_urban', ax=axes[0, 2])
    g.set(xlabel='Cluster', ylabel='Ratio of Urban Inhabitants')

    g = sb.boxplot(data=d, x='cluster', y='crimes_96_per_1000', ax=axes[1, 0])
    g.set(xlabel='Cluster', ylabel='Crimes per 1000 Inhabitants (1996)')

    g = sb.countplot(data=d, x='cluster', hue='gender', ax=axes[1, 1],
        palette=['dodgerblue', 'violet'])
    g.set(xlabel='Cluster', ylabel='Count')
    g.legend(['Male', 'Female'])
    for container in g.containers:
        g.bar_label(container)

    g = sb.boxplot(data=d, x='cluster', y='age', ax=axes[1, 2])
    g.set(xlabel='Cluster', ylabel='Age')

    plt.show()
    
    g = sb.scatterplot(data=d, x='age', y='avg_salary', hue='cluster', alpha=0.3)
    g.set(xlabel='Age', ylabel='Average Salary', title='Demographic Clustering Scatterplot')
    plt.show()

def analysis_financial(d):
    fig, axes = plt.subplots(2, 3)
    fig.suptitle('Financial Clustering')

    g = sb.boxplot(data=d, x='cluster', y='avg_daily_balance', ax=axes[0, 0])
    g.set(xlabel='Cluster', ylabel='Average Daily Account Balance')

    g = sb.boxplot(data=d, x='cluster', y='balance_deviation', ax=axes[0, 1])
    g.set(xlabel='Cluster', ylabel='Account Balance Deviation')

    g = sb.boxplot(data=d, x='cluster', y='avg_abs_amount', ax=axes[0, 2])
    g.set(xlabel='Cluster', ylabel='Average Absolute Transaction Amount')

    g = sb.boxplot(data=d, x='cluster', y='n_transactions', ax=axes[1, 0])
    g.set(xlabel='Cluster', ylabel='Number of Transactions')

    g = sb.countplot(data=d, x='cluster', hue='gender', ax=axes[1, 1],
        palette=['dodgerblue', 'violet'])
    g.set(xlabel='Cluster', ylabel='Count')
    g.legend(['Male', 'Female'])
    for container in g.containers:
        g.bar_label(container)

    g = sb.boxplot(data=d, x='cluster', y='age', ax=axes[1, 2])
    g.set(xlabel='Cluster', ylabel='Age')

    plt.show()

    g = sb.scatterplot(data=d, x='avg_daily_balance', y='balance_deviation', hue='cluster', alpha=0.3)
    g.set(
        xlabel='Average Daily Account Balance', 
        ylabel='Account Balance Deviation', 
        title='Financial Clustering Scatterplot'
    )
    plt.show()

def analysis_demographic_financial(d):
    fig, axes = plt.subplots(2, 3)
    fig.suptitle('Demographic & Financial Clustering')

    g = sb.boxplot(data=d, x='cluster', y='avg_salary', ax=axes[0, 0])
    g.set(xlabel='Cluster', ylabel='Average Salary')

    g = sb.boxplot(data=d, x='cluster', y='population', ax=axes[0, 1])
    g.set(xlabel='Cluster', ylabel='Population')
    g.set_yscale('log')

    g = sb.boxplot(data=d, x='cluster', y='unemployment_96', ax=axes[0, 2])
    g.set(xlabel='Cluster', ylabel='Unemployment Ratio (1996)')

    g = sb.boxplot(data=d, x='cluster', y='avg_daily_balance', ax=axes[1, 0])
    g.set(xlabel='Cluster', ylabel='Average Daily Account Balance')

    g = sb.boxplot(data=d, x='cluster', y='balance_deviation', ax=axes[1, 1])
    g.set(xlabel='Cluster', ylabel='Account Balance Deviation')

    g = sb.boxplot(data=d, x='cluster', y='avg_abs_amount', ax=axes[1, 2])
    g.set(xlabel='Cluster', ylabel='Average Absolute Transaction Amount')

    plt.show()

def analysis_loan(d):
    _, axes = plt.subplots(2, 3)

    g = sb.boxplot(data=d, x='cluster', y='avg_daily_balance', ax=axes[0, 0])
    g.set(xlabel='Cluster', ylabel='Average Daily Balance')

    g = sb.boxplot(data=d, x='cluster', y='balance_deviation', ax=axes[0, 1])
    g.set(xlabel='Cluster', ylabel='Balance Deviation')

    paid = (d.groupby('cluster')['status'].value_counts(normalize=True).mul(100)
        .rename('paid_percent').reset_index())

    g = sb.barplot(data=paid, x='cluster', y='paid_percent', hue='status', 
        ax=axes[0, 2], palette=['forestgreen', 'firebrick'])
    g.set(xlabel='Cluster', ylabel='Paid Loans (%)')
    g.get_legend().remove()
    for container in g.containers:
        g.bar_label(container, fmt='%.1f')

    g = sb.boxplot(data=d, x='cluster', y='payments', ax=axes[1, 0])
    g.set(xlabel='Cluster', ylabel='Payments')

    g = sb.boxplot(data=d, x='cluster', y='amount', ax=axes[1, 1])
    g.set(xlabel='Cluster', ylabel='Amount')

    g = sb.boxplot(data=d, x='cluster', y='duration', ax=axes[1, 2])
    g.set(xlabel='Cluster', ylabel='Duration')

    plt.show()

def select_best(X):
    best_score = 0
    best_n_clusters = None
    best_algorithm = None
    best_params = None

    def try_algorithm(algo, name, params):
        nonlocal best_score, best_n_clusters, best_algorithm, best_params

        algo.set_params(**params)
        algo.fit(X)
        labels = algo.labels_

        try:
            score = silhouette_score(X, labels)
        except ValueError:
            return

        if score > best_score:
            best_score = score
            best_n_clusters = len(np.unique(labels))
            best_algorithm = name
            best_params = params

    params = {
        'KMedoids': {
            'metric': ['euclidean', 'manhattan', 'chebyshev']
        },
        'KMeans': {},
        'Agglomerative': {},
        'DBSCAN': {
            'eps': [0.25, 0.5],
            'min_samples': [5, 10, 20],
        }
    }
    
    for n_clusters in range(2, 7):
        algorithms = {
            'KMedoids': KMedoids(n_clusters=n_clusters, init='k-medoids++', random_state=0),
            'KMeans': KMeans(n_clusters=n_clusters, random_state=0),
            'Agglomerative': AgglomerativeClustering(n_clusters=n_clusters)
        }

        for name, algo in algorithms.items():
            params_list = list(ParameterGrid(params[name]))

            for p in params_list:
                try_algorithm(algo, name, p)

    name = 'DBSCAN'
    params_list = list(ParameterGrid(params[name]))

    for p in params_list:
        try_algorithm(DBSCAN(), name, p)

    print(f'Best: {best_algorithm} with {best_n_clusters} clusters '
        f'(score: {round(best_score, 4)})')
    print(f'Best Params: {best_params}')
    return best_n_clusters, best_algorithm

def build_feature_matrix(d, features):
    return StandardScaler().fit_transform(data.select(d, features))

def perform_clustering_task(algo, X, d, analysis_func):
    algo.fit(X)
    d['cluster'] = algo.labels_
    n_clusters = d['cluster'].nunique()
    silhouette_plot(X, d['cluster'], n_clusters)
    analysis_func(d)

def main():
    sb.set_theme(palette='bright')
    d = data.get_clustering_data()
    # Drop disponents
    d = d[d['type'] != 'DISPONENT']

    # Begin clustering
    print('BASIC CLIENT CLUSTERING')
    features = ['gender', 'age']
    X = build_feature_matrix(d, features)

    # BEST: KMeans, 4 Clusters
    # select_best(X)
    perform_clustering_task(KMeans(n_clusters=4, random_state=0), X, d, analysis_client)

    print('DEMOGRAPHIC CLUSTERING')
    features = ['gender', 'age', 'avg_salary', 'crimes_96_per_1000',
        'unemployment_96', 'entrepreneurs_per_1000', 'ratio_urban']
    X = build_feature_matrix(d, features)

    # BEST: Agglomerative, 2 Clusters
    # select_best(X)
    perform_clustering_task(AgglomerativeClustering(n_clusters=2), X, d, analysis_demographic)

    d = d[~d['avg_daily_balance'].isnull()]
    d = d[d['n_transactions'] > 15]

    print('FINANCIAL CLUSTERING')
    features = ['avg_daily_balance', 'balance_deviation', 'avg_abs_amount']
    X = build_feature_matrix(d, features)
    
    # BEST: KMeans, 2 Clusters
    # select_best(X)
    perform_clustering_task(KMeans(n_clusters=2, random_state=0), X, d, 
        analysis_financial)

    print('DEMOGRAPHIC & FINANCIAL CLUSTERING')
    features = ['avg_salary', 'population', 'avg_daily_balance', 'balance_deviation', 'avg_abs_amount']
    X = build_feature_matrix(d, features)

    # BEST: Agglomerative, 2 Clusters
    # Used KMeans, 3 Clusters instead
    # select_best(X)
    perform_clustering_task(KMeans(n_clusters=3, random_state=0), X, d, analysis_demographic_financial)

    print('LOAN CLUSTERING')
    d = data.get_clustering_data()
    d = d[d['type'] != 'DISPONENT']
    d = d[~d['status'].isnull()]

    d['balance_median_n'] = d['balance_distribution_median'] / d['payments']
    
    features = ['amount', 'payments', 'duration']
    X = build_feature_matrix(d, features)
    
    # BEST: KMeans, 4 Clusters
    # select_best(X)
    perform_clustering_task(KMeans(n_clusters=4, random_state=0), X, d, analysis_loan)

if __name__ == '__main__':
    set_working_directory()
    main()
