import numpy as np
from collections import Counter
from abc import ABC, abstractmethod

### Functions mapping messages to bags of features

def bag_of_words(msg):
    """
    Extract a bag of words from a message, based on whitespace
    :param msg: input string
    :return: bag of features, as a Counter
    """
    return Counter(('word', w) for w in msg.split())

def bag_of_ngrams(msg, n):
    """
    Extract a bag of word ngrams from a message, with fixed n
    :param msg: input string
    :param n: size of ngram
    :return: bag of features, as a Counter
    """
    if n == 0:
        raise ValueError('n must be a positive integer')
    words = msg.split()
    if n > len(words):
        return Counter()
    else:
        return Counter(('ngram', tuple(words[i:i+n])) for i in range(len(words)-n+1))

def bag_of_character_ngrams(msg, n):
    """
    Extract a bag of character ngrams from a message (including whitespace), with fixed n
    :param msg: input string
    :param n: size of ngram
    :return: bag of features, as a Counter
    """
    if n == 0:
        raise ValueError('n must be a positive integer')
    elif n > len(msg):
        return Counter()
    else:
        return Counter(('char', msg[i:i+n]) for i in range(len(msg)-n+1))

def bag_of_variable_character_ngrams(msg, min_n, max_n):
    """
    Extract a bag of ngrams from a message (including whitespace), with variable n
    :param msg: input string
    :param min_n: minimum size of ngram (inclusive)
    :param min_n: maximum size of ngram (inclusive)
    :return: bag of features, as a Counter
    """
    if max_n < min_n:
        raise ValueError('max_n must be more than or equal to min_n')
    bag = Counter()
    for n in range(min_n, max_n+1):
        bag += bag_of_character_ngrams(msg, n)
    return bag

### Functions for combining types of feature

class Extractor(ABC):
    """
    Parent class for feature extractors defined in terms of other functions.
    Using classes rather than functions allows them to be pickled.
    """
    @abstractmethod
    def __call__(self, msg):
        """
        Convert a message to a bag of features
        :param msg: input string
        :return: dict-like bag of features
        """


class combine(Extractor):
    """
    Wrap many feature extractors in a single function
    """
    def __init__(self, functions, arg_params=None, kwarg_params=None):
        """
        Wrap many feature extractors in a single function
        :param functions: iterable of functions mapping from a string to a Counter
        - Counters should have distinct keys, to avoid collisions
        :param arg_params: iterable of additional arguments for the feature extractors
        :param kwarg_params: iterable of additional keyword arguments for the feature extractors
        :return: combined feature extractor
        """
        # If parameters for functions are not given, set empty parameters
        if arg_params is None:
            arg_params = [() for _ in functions]
        if kwarg_params is None:
            kwarg_params = [{} for _ in functions]
        # Save functions and additional arguments, to be used in __call__
        self.functions_with_params = list(zip(functions, arg_params, kwarg_params))
    
    def __call__(self, msg):
        """
        Convert a message to a bag of features
        :param msg: input string
        :return: dict-like bag of features
        """
        bag = Counter()
        # Apply each function, with the given parameters
        for func, args, kwargs in self.functions_with_params:
            bag += func(msg, *args, **kwargs)
        return bag


class apply_to_parts(Extractor):
    """
    Wrap a feature extractor, so it applies to several messages concatenated together
    """
    def __init__(self, function, sep):
        """
        Wrap a feature extractor, so it applies to several messages concatenated together
        :param function: function mapping from a string to a Counter
        :param sep: substring separating the individual messages
        :return: new feature extractor
        """
        self.function = function
        self.sep = sep
    
    def __call__(self, msg):
        """
        Convert a message to a bag of features
        :param msg: input string
        :return: dict-like bag of features
        """
        bag = Counter()
        # Apply the function to each part
        for part in msg.split(self.sep):
            bag += self.function(part)
        return bag


### Functions for producing vectors of features

def get_global_set(bags_of_features):
    """
    Find all the distinct features in many bags of features 
    :param bags_of_features: iterable of dict-like or set-like
    :return: set of features
    """
    features = set()
    for bag in bags_of_features:
        features.update(bag)
    return features

def document_frequency(bags_of_features):
    """
    Find all the distinct features in many bags of features,
    and how often each occurs (in how many bags each occurs)
    :param bags_of_features: iterable of Counters
    :return: Counter mapping features to their document frequencies
    """
    freq = Counter()
    for bag in bags_of_features:
        freq.update(bag.keys())
    return freq

def feature_list_and_dict(features):
    """
    Assign numerical indices to a global list of features
    :param features: iterable of feature names
    :return: sorted list of features, dict mapping features to their indices
    """
    feature_list = sorted(features)
    feature_dict = {feat:i for i, feat in enumerate(feature_list)}
    return feature_list, feature_dict

def vectorise_one(bag, feature_dict):
    """
    Convert a bag of features to a numpy array
    :param bag: Counter of features
    :param feature_dict: dict mapping feature names to indices
    :return: feature vector
    """
    N = len(feature_dict)
    vec = np.zeros(N)
    for feat, value in bag.items():
        if feat in feature_dict:  # Ignore features that are not in the dictionary
            vec[feature_dict[feat]] = value
    return vec

def vectorise(bags, feature_dict):
    """
    Convert bags of features to numpy arrays
    :param bags: Counters of features
    :param feature_dict: dict mapping feature names to indices
    :return: feature vectors as a matrix
    """
    N = len(feature_dict)
    vecs = np.zeros((len(bags), N))
    for i, b in enumerate(bags):
        for feat, value in b.items():
            if feat in feature_dict:  # Ignore features that are not in the dictionary
                vecs[i, feature_dict[feat]] = value
    return vecs

def get_vectors(msgs, extractor, feature_dict, weights=None):
    """
    Get feature vectors for many messages
    :param msgs: input strings
    :param extractor: feature extractor, mapping from a string to a bag of features
    :param feature_dict: dict mapping from features names to indices
    :param weights: array of weights, to be multiplied with extracted vectors
    :return: feature vectors as a matrix
    """
    bags = [extractor(m) for m in msgs]
    vectors = vectorise(bags, feature_dict)
    if weights is not None:
        vectors *= weights
    return vectors

class Vectoriser:
    """
    Class for converting messages to feature vectors
    """
    def __init__(self, extractor, feature_dict, weights=None):
        """
        :param extractor: feature extractor, mapping from a string to a bag of features
        :param feature_dict: dict mapping from features names to indices
        :param weights: array of weights, to be multiplied with extracted vectors
        """
        self.extractor = extractor
        self.feature_dict = feature_dict
        self.weights = weights
    
    def __call__(self, msgs):
        """
        Get feature vectors for one or many messages
        :param msgs: input strings
        :return: feature vectors as a matrix
        """
        # If only one message was given, convert to a list
        if isinstance(msgs, str):
            msgs = [msgs]
        return get_vectors(msgs, self.extractor, self.feature_dict, self.weights)
    

### For human readability

def bagify_one(vector, feature_list):
    """
    Convert a feature vector to a bag of features
    :param vector: numpy array
    :param feature_list: global list of feature names
    :return: bag of features
    """
    bag = Counter()
    for i in vector.nonzero():
        bag[feature_list[i]] = vector[i]
    return bag

def bagify(vectors, feature_list):
    """
    Convert feature vectors to bags of features
    :param vectors: numpy array (matrix)
    :param feature_list: global list of feature names
    :return: list of bags of features
    """
    return [bagify_one(v, feature_list) for v in vectors]