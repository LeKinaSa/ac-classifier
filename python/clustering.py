
from sklearn.cluster import DBSCAN, KMeans, MiniBatchKMeans

import data
from data import set_working_directory

def main():
    d, _ = data.get_data()
    d = d.drop('frequency', axis=1)
    clustering_techniques = {
        'DBSCAN': DBSCAN(eps=3, min_samples=2),
        'Kmeans': KMeans(n_clusters=3, random_state=1),
        'MiniBatches': MiniBatchKMeans(n_clusters=2, random_state=0, batch_size=6)
    }
    for technique in clustering_techniques:
        clustering_technique = clustering_techniques[technique]
        clustering_technique.fit(d)
        print(clustering_technique.labels_)
    return

if __name__ == '__main__':
    set_working_directory()
    main()
