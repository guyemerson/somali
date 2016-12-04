import pickle, numpy as np

def highest(classifiers, N=10):
    """
    Find the features most associated with each class
    :param classifiers: list of LogisticRegression models
    :param N: number of features to return
    :return: top features for each classifier (as a matrix)
    """
    res = np.zeros((len(classifiers), N), dtype='int64')
    for i, c in enumerate(classifiers):
        # Get weights for each feature (squeeze to change shape from (1,K) to (K))
        coef = c.coef_.squeeze()
        # Get the indices of the highest N weights, in descending order
        top = coef.argsort()[:-N-1:-1]
        # If a weight is not positive, set the index to -1
        top[coef[top] <= 0] = -1
        # Keep this result 
        res[i] = top
    return res

def display(name):
    """
    Display the top features for each code
    :param name: name of the dataset
    """
    with open('../data/{}_classifiers.pkl'.format(name), 'rb') as f:
        classifiers = pickle.load(f)
    top = highest(classifiers)
    with open('../data/{}_codes.pkl'.format(name), 'rb') as f:
        codes = pickle.load(f)
    code_list = [x for x, _ in codes]
    with open('../data/{}_vocab.pkl'.format(name), 'rb') as f:
        vocab = pickle.load(f)
    vocab_list = [x for x, _ in vocab]
    for i, feats in enumerate(top):
        print(code_list[i])
        print([vocab_list[x] for x in feats if x >= 0])
    

if __name__ == "__main__":
    display('nutrition')