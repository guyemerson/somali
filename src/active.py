import numpy as np

from logistic import predict_prob

def entropy(p):
    """
    Calculate the entropy of probabilities of binary outcomes
    (the closer the probability is to 1/2, the higher the entropy)
    :param p: numpy array of probabilities
    :return: numpy array of entropies
    """
    q = 1-p
    return p * np.log(p) + q * np.log(q)

def score_by_uncertainty(data, classifiers):
    """
    Score datapoints by how uncertain a classifier is
    :param data: array of vectors
    :param classifiers: one or more probabilistic classifiers for a binary decisions
    :return: entropy of each datapoint
    """
    # Get the classifier's prediction probabilities
    prob = predict_prob(classifiers, data)
    print(prob.shape)
    # Convert this to uncertainty
    return entropy(prob)

# In a Bayesian setting, we define a prior distribution over possible models,
# and observing data then gives us a posterior distribution.
# Intuitively, the more often we see a particular feature with a particular code,
# the more confident we should become that this is a real correlation, rather than random noise.
# We want to choose new data points that are likely to make big changes to the posterior.
# However, calculating the posterior exactly is usually intractable.
# Here, we use a crude approximation, comparing two point estimates of the posterior,
# one with strong regularisation (underfitting), and one with weak regularisation (overfitting).
# If the overfitting model makes a confident prediction, while the underfitting model does not,
# this suggests that this data point uses features that we are marginally sure about.

def score_by_relative_uncertainty(data, over, under):
    """
    Score datapoints by the difference in uncertainty between two classifiers
    :param data: array of vectors
    :param over: one or more probabilistic classifiers for a binary decisions
    :param under: one or more probabilistic classifiers for a binary decisions
    :return: "under" entropy minus "over" entropy 
    """
    # Get the classifiers' prediction probabilities
    over_prob = predict_prob(over, data)
    under_prob = predict_prob(under, data)
    # Find the difference in uncertainty
    return entropy(under_prob) - entropy(over_prob)

# For a single classifier, we can just take datapoints with the largest scores
# For multiple classifiers, we need to combine the scores
# Below, we use reweighted range voting - http://rangevoting.org/RRV.html
# This is so that we choose some examples from each classifier

def top_N(scores, N=None, weights=None, R=2, normalise=False):
    """
    Find the most highly scored datapoints
    :param scores: numpy array
    :param N: number of datapoints to return (default, all)
    :param weights: if there are multiple scores per datapoint,
    how much weight to place on each column of scores (default equal weight)
    :param R: factor to use in reweighting
    R=1 corresponds to Jefferson/D'Hondt
    R=2 (default) corresponds to Webster/Sainte-Laguë 
    :param normalise: whether to normalise each column of scores (default no)
    :return: indices of the top N datapoints, sorted from highest to lowest
    """
    # If N is not specified, return all
    if N is None:
        N = len(scores)
    # If there is just one set of scores, return the highest
    if scores.ndim == 1:
        return scores.argsort()[:-N-1:-1]
    
    # Apply reweighted range voting
    
    # If weights are not given, weight classifiers evenly
    if weights is None:
        weights = np.ones(scores.shape[1])
    # If required, normalise each classifiers' scores to lie in exactly [0,1]
    if normalise:
        # Create a copy, so that this function does not have side effects
        scores = scores.copy()
        scores -= scores.min(0)
        scores /= scores.max(0)
    # Initialise indices
    top = []
    # Iteratively find the highest scoring datapoint
    for _ in range(N):
        # Reweight, then find the range voting winner
        # Downweight each classifier, according to the sum of its scores for the datapoints already chosen
        cur_weights = weights / (1 + R * scores[top].sum(0))
        # Find the total reweighted score
        weighted_scores = (scores * cur_weights).sum(1)
        # Ignore datapoints that have already been chosen
        weighted_scores[top] = 0
        # Record the highest 
        top.append(weighted_scores.argmax())
    
    return np.array(top)
