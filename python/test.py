
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier

from simple import model_learning_and_classification
import data

def get_columns(l):
    optional_columns = [
        ['muni_under499_account', 'muni_500_1999_account', 'muni_2000_9999_account'],
        ['unemployment_95_account', 'unemployment_evolution_account'],
        ['crimes_95_per_1000_account', 'crimes_evolution_account'],
        ['n_cities_account'],
        ['balance_deviation'],
        ['avg_salary_account'],
        ['enterpreneurs_per_1000_account']
    ]
    columns = []
    for item in l:
        for c in optional_columns[item]:
            columns.append(c)

    return columns

def main(threshold):
    dev, competition = data.get_data()

    selected_columns = [
        'loan_id', 'status', 
        'duration','gender_owner',
        #'enterpreneurs_per_1000_account',
        'card',
        'balance_distribution_first_quarter',
        'high_balance', 'last_neg',
    ]
    to_combine = [0, 1, 2, 3, 4, 5, 6]
    best_l = None
    best_auc = 0
    for size in range(len(to_combine) + 1):
        for l in combinations(to_combine, size):
            for c in get_columns(l):
                selected_columns.append(c)
            dev_data = data.select(dev, selected_columns)
            comp_data = data.select(competition, selected_columns)
            (scores, _) = model_learning_and_classification(dev_data, comp_data, RandomForestClassifier(), {}, False)
            (auc, _) = scores
            if auc > threshold:
                print(l)
                print(f'AUC score: {auc}')
            
            if best_auc < auc:
                best_auc = auc
                best_l = l
    print("Best L: ", best_l)

if __name__ == '__main__':
    main(0.76)
