import pickle, csv
from collections import Counter, namedtuple


# Convert the CSV file to a list of tuples,
# using the first line to define a namedtuple type
with open('../data/initial.csv', newline='') as f:
    reader = csv.reader(f)
    headings = next(reader)
    
    Message = namedtuple('Message', headings)
    
    data = [Message(*x) for x in reader]

# Count the tokens
count = Counter
for x in data:
    count.update(x.text.split())

# Save to file
with open('../data/vocab.pkl', 'wb') as f:
    pickle.dump(count, f)