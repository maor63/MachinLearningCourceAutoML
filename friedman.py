import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare
from scikit_posthocs import posthoc_nemenyi


def friedman_test(results_df, score, higher_is_better):
    # mean all k folds of each dataset + algo and take the relevant score
    folds_mean = (results_df.groupby((results_df['Algorithm Name'] != results_df['Algorithm Name'].shift()).cumsum()))\
        .mean().reset_index(drop=True)[score]

    # get models names
    algorithm_names = results_df['Algorithm Name'].drop_duplicates().values

    # reshape to a dataframe of datasets as rows and models as columns
    mean_reshape = pd.DataFrame(np.reshape(folds_mean.values, (int(folds_mean.shape[0] / len(algorithm_names)),
                                                               len(algorithm_names))), columns=algorithm_names)

    # run test
    stat, p_value = friedmanchisquare(*mean_reshape.T.values)

    # reject
    if p_value < 0.05:
        print('Rejected (different distributions)')

        # post hoc
        nemenyi_p_values = posthoc_nemenyi(mean_reshape.T.values).values
        for algo1_index in range(len(algorithm_names)):
            algo1 = algorithm_names[algo1_index]
            algo1_mean = mean_reshape[algo1].mean()
            for algo2_index in range(algo1_index + 1, len(algorithm_names)):
                algo2 = algorithm_names[algo2_index]
                algo2_mean = mean_reshape[algo2].mean()
                if algo1_index != algo2_index:
                    algos_p_val = nemenyi_p_values[algo1_index][algo2_index]
                    if algos_p_val < 0.05:
                        if (algo1_mean > algo2_mean and higher_is_better) or \
                                (algo1_mean < algo2_mean and not higher_is_better):
                            print(algo1 + ' is significantly better than ' + algo2, '(Nemenyi test with 0.05)')
                        else:
                            print(algo2 + ' is significantly better than ' + algo1, '(Nemenyi test with 0.05)')
                    else:
                        print(algo1 + ' and ' + algo2 + ' are not significant')
    else:
        print('Fail to reject (same distributions)')


results_df = pd.read_csv('./results_T_completed.csv')
friedman_test(results_df, score='Accuracy', higher_is_better=True)
