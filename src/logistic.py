import pickle, numpy as np
from sklearn import linear_model

def train(features, codes, penalty='l1', C=1):
    """
    Train logistic regression classifiers,
    independently for each code 
    :param features: input matrix
    :param codes: output matrix
    :param penalty: type of regularisation (l1 or l2)
    :param C: inverse of regularisation strength
    :return: list of classifiers
    """
    classifiers = []
    # Iterate throgh each code (i.e. each column of codes matrix)
    for code_i in codes.transpose():
        # Define a logistic regression model for each code
        model = linear_model.LogisticRegression(
            penalty=penalty,
            C=C)
        # (Could use class_weight to make positive instances more important...)
        
        if code_i.sum() > 1:  # Make sure there is more than one training example
            # Train the model
            model.fit(features, code_i)
            
            classifiers.append(model)
        
        else:
            classifiers.append(None)
    
    return classifiers


def train_on_file(input_name, output_name=None, **kwargs):
    """
    Train logistic regression classifiers on a file 
    :param input_name: name of input file (inside ../data/, without .pkl file extension)
    :param output_name: name of output file (inside ../data/, without .pkl file extension, defaults to input_name + '_classifiers')
    :param **kwargs: additional keyward arguments will be passed to train
    :return: features, codes, classifiers
    """
    # Default value for output_name
    if output_name is None:
        output_name = input_name + '_classifiers'
    # Load features and codes
    with open('../data/{}.pkl'.format(input_name), 'rb') as f:
        features, codes = pickle.load(f)
    # Train model
    classifiers = train(features, codes, **kwargs)
    # Save model
    with open('../data/{}.pkl'.format(output_name), 'wb') as f:
        pickle.dump(classifiers, f)
    return features, codes, classifiers


def predict(classifiers, messages):
    """
    Apply a number of classifiers to a number of messages,
    returning the most likely result for each classfier ond message
    :param classifiers: list of classifier objects
    :param messages: feature vectors (as a matrix)
    :return: array of predictions
    """
    predictions = [c.predict(messages) if c is not None else np.zeros(len(messages)) for c in classifiers]
    return np.array(predictions).transpose()

def predict_proba(classifiers, messages):
    """
    Apply a number of classifiers to a number of messages,
    returning the probability of predicting each code for each message
    :param classifiers: list of classifier objects
    :param messages: feature vectors (as a matrix)
    :return: array of probabilities
    """
    prob = [c.predict_proba(messages) for c in classifiers]
    return np.array(prob).transpose()


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
    features, codes, classifiers = train_on_file('delivery', output_name='delivery_C1', C=1)
    predictions = predict(classifiers, features)
    evaluate(predictions, codes)