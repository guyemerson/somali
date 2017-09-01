"""
The snippets in this file can be used to inspect the results of a hyperparameter grid search 
"""

import pickle
from operator import itemgetter

### Load the data

with open('../data/gridsearch.pkl', 'rb') as f:
    res = pickle.load(f)
# See how many settings have been tried
print(len(res))
# Get the number of codes
K = next(iter(res.values())).shape[1]

# Choose which score to optimise
# 0 accuracy
# 1 precision
# 2 recall
# 3 F1
WHICH = 1

# For each code, get a summary for each setting, averaging over cross-validation folds
summary = [{s:r[:,i,WHICH].mean() for s,r in res.items()} for i in range(K)]
# For an overall summary, also average across codes 
overall_summary = {s:r[:,:,WHICH].mean() for s,r in res.items()}

# Get the best settings for different slices of the results

def top(scores, constr=()):
    """
    Sort the settings by score, but restricting to a subset
    :param scores: dict mapping settings to scores
    :param constr: list of constraints on settings, each of the form (index, value)
    """
    filtered = [(s,r) for s,r in scores.items()
                if all(s[i] == x
                       for i,x in constr)]
    return sorted(filtered, reverse=True, key=itemgetter(1))

# The indices follow those defined in gridsearch.py:
# 0 word bigrams
# 1 character n-grams
# 2 spelling normalisation
# 3 frequency threshold
# 4 tf-idf re-weighting
# 5 regularisation type
# 6 inverse regularisation strength
# 7 weighting scheme

# Examples of constraints
constr = []  # no constraints (i.e. consider all settings)
constr = [(6, 0.25)]  # only when inverse regularisation strength is 0.25 
constr = [(2, True),  # only with spelling normalisation
          (1, (4,6)),  # AND with character n-grams from n=4 to n=6
          (4, False)]  # AND without tf-idf re-weighting

# Find the best N overall settings, under the given constraints
overall_best = top(overall_summary, constr)
for x in overall_best[:20]:  # choose N here
    print(x)

# Find the best N settings for each code, under the given constraints
best = [top(summary[i], constr) for i in range(K)]
for i in range(K):
    print(i)
    for x in best[i][:5]:  # choose N here
        print(x)

# For the best N overall settings, print the score for each code 
for k in range(5):  # choose N here
    settings = overall_best[k][0]
    print(settings)
    for i in range(K):
        print(summary[i][settings])

# For specific values of a hyperparameter,
# find the best settings for the remaining hyperparameters
# Examples of 'index' and 'values':
index = 5
values = ['l1', 'l2']
index = 2
values = [True, False]
index = 1
values = [False, (4,6), (5,6), (5,7)]
# For each code, print the best setting for each of the specified values
# (also applying the given constraints)
for i in range(K):
    print(i)
    for v in values:
        b = top(summary[i], [(index, v)]+constr)
        print(b[0])
