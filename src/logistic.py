import pickle, numpy as np
from sklearn import linear_model

def train(features, codes, C=0.01):
    """
    Train logistic regression classifiers,
    independently for each code 
    :param features: input matrix
    :param codes: output matrix
    :param C: inverse of L1 regularisation strength
    :return: list of classifiers
    """
    classifiers = []
    
    N, K = codes.shape  # number of messages, number of codes
    total = codes.sum(0)  # total labelled with each code
    
    for i in range(K):
        # Define a logistic regression model for each code
        model = linear_model.LogisticRegression(
            penalty='l1',
            class_weight={1:total[i], 0:N-total[i]},
            C=C)
        
        model.fit(features, codes[:,i])
        
        classifiers.append(model)
    
    return classifiers

if __name__ == "__main__":
    with open('../data/malaria.pkl', 'rb') as f:
        features, codes = pickle.load(f)
    classifiers = train(features, codes)
    with open('../data/malaria_classifiers.pkl', 'wb') as f:
        pickle.dump(classifiers, f)