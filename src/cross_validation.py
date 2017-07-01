import os.path, pickle, numpy as np
from sklearn import metrics, model_selection

from logistic import train

def scores_one(classifier, X, y_true):
    """
    Accuracy, precision, recall, F1 score,
    for a classifier on some data
    """
    y_pred = classifier.predict(X)
    return np.array([metrics.accuracy_score(y_true, y_pred),
                     metrics.precision_score(y_true, y_pred),
                     metrics.recall_score(y_true, y_pred),
                     metrics.f1_score(y_true, y_pred)])

def scores(classifiers, X, y_true):
    """
    Accuracy, precision, recall, F1 score,
    for several classifiers on some data
    """
    return np.array([scores_one(cl, X, y)
                     for cl, y in zip(classifiers, y_true.transpose())])

def save_folds(X, filename, directory='../data', n_splits=10):
    """
    Randomly split a dataset into folds, and save the folds
    """
    folds = list(model_selection.KFold(n_splits=n_splits, shuffle=True).split(X))
    with open(os.path.join(directory, filename+'.pkl'), 'wb') as f:
        pickle.dump(folds, f)
    return folds

def cross_validate(msgs, y, folds, preprocessor, preproc_args=(), preproc_kwargs={}, train_args=(), train_kwargs={}):
    """
    Apply cross-validation (trains using logistic.train)
    Similar to sklearn.model_selection.cross_val_score,
    but using arrays of scores for each fold, rather than a single number.
    :param msgs: messages, as strings
    :param y: codes, as a boolean array
    :param folds: iterable of train/test indices
    :param preprocessor: function mapping from the training data to a vectoriser
    Optional:
    :param preproc_args: additional positional arguments for the preprocessor
    :param preproc_kwargs: additional keyword arguments for the preprocessor
    :param train_args: additional positional arguments for training
    :param train_kwargs: additional keyword arguments for training
    :return: array of shape [fold, code, score], where the score varies over:
        accuracy, precision, recall, f1 
    """
    all_scores = []
    for train_inds, test_inds in folds:
        # Produce a vectoriser based on the training messages
        # (e.g. including tf-idf reweighting)
        vectoriser = preprocessor((msgs[i] for i in train_inds), *preproc_args, **preproc_kwargs)
        # Convert messages to vectors
        X = vectoriser(msgs)
        # Train the classifiers
        classifiers = train(X[train_inds], y[train_inds], *train_args, **train_kwargs)
        # Evaluate the classifiers
        fold_scores = scores(classifiers, X[test_inds], y[test_inds])
        all_scores.append(fold_scores)
    return np.array(all_scores)
