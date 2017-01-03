import os, pickle, numpy as np

def highest(classifiers, N=10):
    """
    Find the features most associated with each class
    :param classifiers: list of LogisticRegression models
    :param N: number of features to return
    :return: top features for each classifier (as a matrix)
    """
    res = np.zeros((len(classifiers), N), dtype='int64')
    for i, c in enumerate(classifiers):
        if c is not None:
            # Get weights for each feature (squeeze to change shape from (1,K) to (K))
            coef = c.coef_.squeeze()
            # Get the indices of the highest N weights, in descending order
            top = coef.argsort()[:-N-1:-1]
            # If a weight is not positive, set the index to -1
            top[coef[top] <= 0] = -1
            # Keep this result 
            res[i] = top
        else:
            res[i] = -1
    return res

def display(dataset, classifier_suffix, code_suffix='codes', feature_suffix='features', directory='../data'):
    """
    Display the top features for each code
    :param dataset: name of the dataset to load, which the following suffixes are added to
    :param classifier_suffix: classifier file
    :param code_suffix: code file (default 'codes')
    :param feature_suffix: feature file (default 'features')
    The following files will be loaded:
    - {dataset}_{classifier_suffix}.pkl
    - {dataset}_{code_suffix}.pkl
    - {dataset}_{feature_suffix}.pkl
    :param directory: directory of data files (default ../data)
    """
    with open(os.path.join(directory, '{}_{}.pkl'.format(dataset, classifier_suffix)), 'rb') as f:
        classifiers = pickle.load(f)
    top = highest(classifiers)
    with open(os.path.join(directory, '{}_{}.pkl'.format(dataset, code_suffix)), 'rb') as f:
        codes = pickle.load(f)
    code_list = [x for x, _ in codes]
    with open(os.path.join(directory, '{}_{}.pkl'.format(dataset, feature_suffix)), 'rb') as f:
        feats = pickle.load(f)
    feat_list = [x for x, _ in feats]
    for name, best_feats in zip(code_list, top):
        print(name)
        print([feat_list[x] for x in best_feats if x >= 0])
        print()
    

if __name__ == "__main__":
    display('nutrition', 'C1')
