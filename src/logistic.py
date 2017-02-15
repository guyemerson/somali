import pickle, os, numpy as np
from sklearn import linear_model

def train(features, codes, penalty='l1', C=1, keywords=None, keyword_strength=1, keyword_weight=1, weight_option='balanced', smoothing=0):
    """
    Train logistic regression classifiers,
    independently for each code
    :param features: input matrix, of shape [num_messages, num_features]
    :param codes: output matrix, of shape [num_messages, num_codes]
    :param penalty: type of regularisation ('l1' or 'l2')
    :param C: inverse of regularisation strength
    :param keywords: (optional) matrix of shape [num_codes, num_features],
        with a nonzero value when a feature should be considered a keyword for that code.
        The value should be the importance of this keyword, relative to a single message
    :param keyword_strength: (default 1) value to give to the vector for each keyword feature
    :param keyword_weight: (default 1) weight to assign a keyword vector (relative to a normal message)
    :param weight_option: determine the total weight for each code across all messages. Options are:
        - 'balanced': total weight is the same for true and false
        - 'smoothed': total weight is the observed number, plus smoothing
    :param smoothing: (default 0, i.e. no smoothing) constant to add,
        to reweight frequencies of each code
    :return: list of classifiers
    """
    N, F = features.shape
    classifiers = []
    # Iterate through each code (i.e. each column of codes matrix)
    for i, code_i in enumerate(codes.transpose()):
        # If there are no training examples, return None for this code
        N_pos = code_i.sum()
        if N_pos == 0:
            classifiers.append(None)
            continue
        
        # Check if keywords were given
        if keywords is None:
            # If no keywords, leave matrices the same
            feat_mat = features
            code_vec = code_i
            N_key = 0
        else:
            # If given, add the keywords as additional messages
            indices = keywords[i].nonzero()[0]
            N_key = len(indices)
            # Treat each keyword feature as a separate message
            key_features = np.zeros((N_key, F))
            for j, j_ind in enumerate(indices):
                # Set the strength of the feature as asked for
                key_features[j, j_ind] = keyword_strength
            key_codes = np.ones(N_key, dtype='bool')
            # Extend the feature and code arrays
            feat_mat = np.concatenate((features, key_features))
            code_vec = np.concatenate((code_i, key_codes))
        
        # Weight classes as asked for
        if weight_option == 'balanced':
            class_weight = 'balanced'
        elif weight_option == 'smoothed':
            N_neg = N - N_pos
            class_weight = {True: (N_pos + smoothing) / (N_pos + N_key*keyword_weight),
                            False: (N_neg + smoothing) / N_neg}
        else:
            raise ValueError('weight option not recognised')
        
        # Weight keyword examples as asked for
        if keyword_weight == 1:
            sample_weight = None
        else:
            sample_weight = np.ones(N+N_key)
            sample_weight[N:] = keyword_weight
        
        # Initialise a logistic regression model
        model = linear_model.LogisticRegression(penalty=penalty, C=C, class_weight=class_weight)
        
        # Train the model
        model.fit(feat_mat, code_vec, sample_weight=sample_weight)
        
        classifiers.append(model)
    
    return classifiers


def train_on_file(input_name, output_suffix=None, directory='../data', **kwargs):
    """
    Train logistic regression classifiers on a file 
    :param input_name: name of input file (without .pkl file extension)
    :param output_suffix: string to append to name of output file
    (if none is given, no file is saved)
    :param directory: directory of data files (default ../data)
    :param **kwargs: additional keyward arguments will be passed to train
    :return: features, codes, classifiers
    """
    # Load features and codes
    with open(os.path.join(directory, input_name+'.pkl'), 'rb') as f:
        features, codes = pickle.load(f)
    # Train model
    classifiers = train(features, codes, **kwargs)
    # Save model
    if output_suffix:
        with open(os.path.join(directory, '{}_{}.pkl'.format(input_name, output_suffix)), 'wb') as f:
            pickle.dump(classifiers, f)
    return features, codes, classifiers


def predict(classifiers, messages):
    """
    Apply a number of classifiers to a number of messages,
    returning the most likely result for each classfier ond message
    :param classifiers: classifier or list of classifiers
    :param messages: feature vectors (as a matrix)
    :return: array of predictions
    """
    # If more than one classifier is given, apply each
    if isinstance(classifiers, list):
        # Get the predictions from each classifier, giving zeros when a classifier is None
        predictions = [c.predict(messages) if c is not None else np.zeros(len(messages)) for c in classifiers]
        # Transpose so that the shape is (n_datapoints, n_classifiers)
        return np.array(predictions, dtype='bool').transpose()
    else:
        return classifiers.predict(messages)

def predict_prob(classifiers, messages):
    """
    Apply a number of classifiers to a number of messages,
    returning the probability of predicting each code for each message
    :param classifiers: classifier or list of classifiers
    :param messages: feature vectors (as a matrix)
    :return: array of probabilities
    """
    # If more than one classifier is given, apply each
    if isinstance(classifiers, list):
        # Get the prediction probabilities from each classifier
        # c.predict_proba returns probabilities for [False, True]
        # taking [:,1] will just give us probability of True
        prob = [c.predict_proba(messages)[:,1] if c is not None else np.zeros(len(messages)) for c in classifiers]
        # Transpose so that the shape is (n_datapoints, n_classifiers)
        return np.array(prob).transpose()
    else:
        return classifiers.predict_proba(messages)[:,1]


def evaluate(pred, gold, verbose=True):
    """
    Calculate the accuracy, precision, recall, and F1 score,
    for a set of predictions, compared to a gold standard 
    :param pred: predictions of classifiers
    :param gold: gold standard annotations
    :param verbose: whether to print results (default True)
    :return: accuracy, precision, recall, F1 (each as an array)
    """
    # Check which predictions are correct
    correct = (pred == gold)
    n_correct = correct.sum(0)
    
    # Calculate different types of mistake 
    n_true_correct = (correct * gold).sum(0)
    n_false_correct = (correct * (1-gold)).sum(0)
    n_true_wrong = ((1-correct) * gold).sum(0)
    n_false_wrong = ((1-correct) * (1-gold)).sum(0)
    
    # Accuracy: proportion correct, out of all messages
    accuracy = n_correct / pred.shape[0]
    # Precision: proportion correct, out of those predicted to have a code
    precision = n_true_correct / (n_true_correct + n_false_wrong)
    # Recall: proportion correct, out of those annotated with a code
    recall = n_true_correct / (n_true_correct + n_true_wrong)
    # F1: harmonic mean of precision and recall
    f1 = 2 * precision * recall / (precision + recall)
    
    if verbose:
        # Print results
        print('Accuracy:', accuracy)
        print('Precision:', precision)
        print('Recall:', recall)
        print('F1:', f1)
    
    return accuracy, precision, recall, f1 


if __name__ == "__main__":
    'example use:'
    #features, codes, classifiers = train_on_file('delivery', 'C1', C=1)
    #predictions = predict(classifiers, features)
    #evaluate(predictions, codes)
