import csv, pickle, numpy as np
from collections import Counter

from logistic import evaluate, predict
from features import bag_of_words, vectorise, apply_to_parts

name, weeks = 'wash', '12'
#name, weeks = 'delivery', '34'
#name, weeks = 'nutrition', '5'
#name, weeks = 'malaria', '67'


# Get the unlabelled data

msgs = []
with open('../data/initial_sep.csv', newline='') as f:
    reader = csv.reader(f)
    headings = next(reader)
    
    for row in reader:
        if row[3] in weeks:
            msgs.append(row)

# Vectorise the data

with open('../data/{}_features.pkl'.format(name), 'rb') as f:
    feats = pickle.load(f)
feat_list = [x for x,_ in feats]
feat_dict = {x:i for i,x in enumerate(feat_list)}
V = len(feats)

featurise = apply_to_parts(bag_of_words, '<$$$>')
feat_vecs = vectorise([featurise(x[4]) for x in msgs], feat_dict)

# Load the classifiers and codes

with open('../data/{}_C1.pkl'.format(name), 'rb') as f:
    classifiers = pickle.load(f)

with open('../data/{}_codes.pkl'.format(name), 'rb') as f:
    codes = pickle.load(f)
code_names = [x for x,_ in codes]
headings.extend(code_names)

# Make predictions

predictions = predict(classifiers, feat_vecs)
for i, m in enumerate(msgs):
    m.extend([1 if x else '' for x in predictions[i]])

# Save the data with the predictions

with open('../data/{}_predictions.csv'.format(name), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(headings)
    writer.writerows(msgs)
