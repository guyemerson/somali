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
        res[i] = c.coef_.squeeze().argsort()[:-N-1:-1]
    return res

if __name__ == "__main__":
    with open('../data/malaria_classifiers.pkl', 'rb') as f:
        classifiers = pickle.load(f)
    top = highest(classifiers)
    with open('../data/malaria_codes.pkl', 'rb') as f:
        codes = pickle.load(f)
    code_list = [x for x, _ in codes]
    with open('../data/malaria_vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    vocab_list = [x for x, _ in vocab]
    for i, feats in enumerate(top):
        print(code_list[i])
        print([vocab_list[x] for x in feats])