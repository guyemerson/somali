import csv, pickle
import numpy as np
from sklearn import linear_model
from collections import Counter

#vocabfile = '../data/vocab.pkl'
datafile = '../data/ANC_Delivery Training Set.xlsx - Short.csv'

# Load vocabulary
#with open(vocabfile, 'rb') as f:
#    vocab = pickle.load(f)

vocab = Counter()
# Extract vocabulary from file
with open(datafile, newline='') as f:
    reader = csv.reader(f)
    headings = next(reader)[1:]
    
    for msg, *_ in reader:
        # Add to counts
        vocab.update(msg.split())

# Define feature vectors based on vocab

vocab_list = sorted(vocab.keys())
vocab_dict = {x:i for i,x in enumerate(vocab_list)}
V = len(vocab_list)

def featurise(msg):
    """
    Convert a message to a bag of words vector
    :param msg: a string
    :return: a vector with the count of each word
    """
    vec = np.zeros(V)
    for w in msg.split():
        vec[vocab_dict[w]] += 1
    
    return vec

messages = []
labels = []

# Preprocess the CSV file
with open(datafile, newline='') as f:
    reader = csv.reader(f)
    headings = next(reader)[1:]
    
    for msg, *lab in reader:
        # Convert messages to a bag of words
        messages.append(featurise(msg))
        labels.append([int(x) for x in lab])

msg_mat = np.array(messages)

# For each class, train a classifier
classifiers = []
for i, name in enumerate(headings):
    print(name)
    
    labs = np.array([x[i] for x in labels])  # 0/1 labels for this heading
    total = sum(labs)
    print("training on {} messages".format(total))
    if total == 0:  # Nothing to train on
        classifiers.append(None)
        continue
    class_weight = {1:total, 0:len(labs)-total}
    
    model = linear_model.LogisticRegression(
        penalty='l1',
        class_weight=class_weight,
        C=0.1)
    
    model.fit(msg_mat, labs)
    
    classifiers.append(model)

# Print the words most associated with each class
for i, c in enumerate(classifiers):
    print(headings[i])
    if not c:
        print("nothing to train on")
        continue
    feats = c.coef_.squeeze().argsort()
    top = [vocab_list[x] for x in feats[-5:]]
    print(top)
    print([vocab[x] for x in top])