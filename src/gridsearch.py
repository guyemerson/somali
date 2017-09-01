"""
This script runs a grid search across hyperparameter settings,
using 10-fold cross-validation for each setting,
and saving accuracy, precision, recall, and F1 scores.
"""

import pickle, numpy as np, os.path
from functools import partial
from itertools import product
from random import shuffle

from preprocess import document_frequency, feature_list_and_dict
from features import apply_to_parts, with_preprocessing, combine, bag_of_words, bag_of_ngrams, bag_of_variable_character_ngrams, Vectoriser
from spelling import rules
from cross_validation import save_folds, cross_validate

### Load the data

# Assume all files are .pkl files in the same directory
DATA_DIR = '../data/'
def filepath(filename):
    return os.path.join(DATA_DIR, filename+'.pkl')

message_filename = 'anc_raw_msgs'
annotation_filename = 'anc_reduced'
fold_filename = 'anc_folds'

# Messages - list of strings
with open(filepath(message_filename), 'rb') as f:
    messages = pickle.load(f)

# Annotations - numpy array of shape [#messages, #codes]
with open(filepath(annotation_filename), 'rb') as f:
    gold = pickle.load(f)

# Folds for cross-validation (should be kept consistent across hyperparameter settings)
# If folds have not been chosen, calculate them and save them
if os.path.exists(filepath(fold_filename)):
    with open(filepath(fold_filename), 'rb') as f:
        folds = pickle.load(f)
else:
    folds = save_folds(gold, fold_filename, DATA_DIR)

### Set up training

# There are two kinds of hyperparameters used here:
# 1. Those used in converting messages to feature vectors
# 2. Those used in training the classifiers
# The first kind are used in the following function 

def get_vectoriser(msgs, bigram, char, spelling, threshold, tf_idf):
    """
    For the given hyperparameter setting,
    get a function mapping messages to feature vectors 
    :param msgs: list of strings
    Hyperparameters --
    :param bigram: bool, whether to use word bigram features
    :param char: False (don't use character n-gram features),
        or a tuple (min_n, max_n)
    :param spelling: bool, whether to use spelling normalisation
    :param threshold: int, minimum frequency for a feature
    :param tf_idf: bool, whether to use tf-idf re-weighting
    """
    # Define the feature extractor
    funcs = [bag_of_words]
    if bigram:
        funcs.append(partial(bag_of_ngrams, n=2))
    if char:
        min_n, max_n = char
        funcs.append(partial(bag_of_variable_character_ngrams,
                             min_n=min_n,
                             max_n=max_n))
    extractor = combine(funcs)
    if spelling:
        extractor = with_preprocessing(extractor, rules)
    extractor = apply_to_parts(extractor, '&&&')
    
    # Get the global list of features
    bags = (extractor(m) for m in msgs)
    freq = document_frequency(bags)
    # Filter out rare features
    freq = {feat:n for feat, n in freq.items() if n >= threshold}
    # Assign indices to features
    feat_list, feat_dict = feature_list_and_dict(freq.keys())
    # Get idf array
    if tf_idf:
        idf = np.empty(len(feat_list))
        for feat, n in freq.items():
            idf[feat_dict[feat]] = 1/n
    else:
        idf = None
    
    # Create vectoriser
    return Vectoriser(extractor, feat_dict, idf)

# Choose the range of values for each hyperparameter

bigram_settings = [True, False]  # Whether to use word bigrams
char_ngram_min = [3,4,5]  # Minimum size of character n-gram
char_ngram_diff = [0,1,2]  # Size of range of character n-grams
# Whether to use character n-grams, and if so, ranges for 'n':
char_ngram_settings = [False] \
                    + [(min_n, min_n+diff)
                       for min_n, diff in product(char_ngram_min,
                                                  char_ngram_diff)]
spelling_settings = [True, False]  # Whether to use spelling normalisation
threshold_settings = [2,5]  # Frequency thresholds
tf_idf_settings = [True, False]  # Whether to use tf-idf re-weighting

# Get all combinations of these settings

pre_settings = product(bigram_settings,
                       char_ngram_settings,
                       spelling_settings,
                       threshold_settings,
                       tf_idf_settings)

# Choose the range of values for the second kind of hyperparameter
# (those used in training)

penalty_settings = ['l1', 'l2']  # Type of regularisation
C_settings = [0.1, 0.25, 1, 4]  # Inverse strength of regularisation
weight_settings = [None]  # How to weight each training point

# Get all combinations of these settings

train_settings = product(penalty_settings,
                         C_settings,
                         weight_settings)
def train_kwargs(penalty, C, weight):
    """
    Produce a keyword dictionary, from hyperparameter values
    :param penalty: type of regularisation
    :param C: inverse strength of regularisation
    :param weight: how to weight each training point
        - None for equal total weight for positive and negative instances
        - float for amount of smoothing
    """
    if weight is None:
        option = 'balanced'
    else:
        option = 'smoothed'
    return {'penalty': penalty,
            'C': C,
            'weight_option': option,
            'smoothing': weight}

# Combine both kinds of hyperparameter
# Randomise the order, so that if the grid search is stopped part way through,
# many different parts of the hyperparameter space have been explored

all_settings = list(product(pre_settings, train_settings))
shuffle(all_settings)

# Save results in a dict, mapping settings to results

grid_filename = 'gridsearch2'

# If the grid search was started previously, load the results
# Otherwise, start from an empty dict
if os.path.exists(filepath(grid_filename)):
    with open(filepath(grid_filename), 'rb') as f:
        results = pickle.load(f)
else:
    results = {}

# For each setting that hasn't already been tried,
# apply cross-validation, and save the scores
for pre_set, train_set in all_settings:
    if pre_set + train_set in results:
        continue
    print(pre_set, train_set)
    scores = cross_validate(messages, gold, folds, get_vectoriser,
                            preproc_args=pre_set,
                            train_kwargs=train_kwargs(*train_set))
    print(scores)
    results[pre_set + train_set] = scores
    with open(filepath(grid_filename), 'wb') as f:
        pickle.dump(results, f)
