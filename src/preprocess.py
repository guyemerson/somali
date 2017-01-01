import csv, pickle, os, numpy as np
from collections import Counter

def preprocess_long(input_file, output_file, text_col=0, ignore_cols=()):
    """
    Preprocess a csv file to feature vectors and binary codes,
    where the input data has a 0 or 1 for each code and message
    :param input_file: csv input file name
    :param output_file: pkl output file name
    :param text_col: index of column containing text
    :param ignore_cols: indices of columns to ignore
    """
    if os.path.splitext(input_file)[1] != '.csv':
        raise ValueError('Input must be a csv file')
    if os.path.splitext(output_file)[1] != '.pkl':
        raise ValueError('Output must be a pkl file')
    
    # Extract total set of features and codes
    
    with open(input_file, newline='') as f:
        # Process the file as a CSV file
        reader = csv.reader(f)
        # Find the headings (the first row of the file)
        headings = next(reader)
        # Restrict ourselves to a subset of columns (not containing text, and not ignored) 
        code_cols = sorted(set(range(len(headings))) - {text_col} - set(ignore_cols))
        K = len(code_cols)
        code_freq = np.zeros(K, dtype='int64')
        code_names = [headings[i] for i in code_cols]
        vocab = Counter()
        # Iterate through data
        for vals in reader:
            vocab.update(vals[text_col].split())
            code_freq += np.array([int(vals[i]) for i in code_cols], dtype='bool')
    
    # Save the features and codes to file
    
    output_name = os.path.splitext(output_file)[0]
    with open(output_name+'_vocab.pkl', 'wb') as f:
        pickle.dump(sorted(vocab.items()), f)
    codes = [(code_names[i], code_freq[i]) for i in range(K)]
    with open(output_name+'_codes.pkl', 'wb') as f:
        pickle.dump(codes, f)
    
    print('Codes:')
    print(*codes, sep='\n')
    
    # Convert messages to vectors
    
    vocab_list = sorted(vocab)
    vocab_dict = {w:i for i,w in enumerate(vocab_list)}
    V = len(vocab)
    
    def vectorise_features(message):
        """
        Convert a message to a feature vector
        :param message: string
        :return: numpy array
        """
        vec = np.zeros(V)
        for w in message.split():
            vec[vocab_dict[w]] += 1
        return vec
    
    # Process data
    
    feature_vecs = []
    code_vecs = []
    
    with open(input_file, newline='') as f:
        # Process the file as a CSV file
        reader = csv.reader(f)
        # Skip headings
        next(reader)
        # Process each message
        for vals in reader:
            msg = vals[text_col]
            feature_vecs.append(vectorise_features(msg))
            code_v = np.array([int(vals[i]) for i in code_cols], dtype='bool')
            code_vecs.append(code_v)
    
    # Save to file
    
    with open(output_file, 'wb') as f:
        feature_mat = np.array(feature_vecs)
        code_mat = np.array(code_vecs)
        pickle.dump((feature_mat, code_mat), f)


def preprocess_pairs(input_file, output_file, text_col=0, ignore_cols=(), uncoded=('', 'NM')):
    """
    Preprocess a csv file to feature vectors and binary codes,
    where the input data has groups of codes,
    and each message has up to two codes from each group
    :param input_file: csv input file name
    :param output_file: pkl output file name
    :param text_col: index of column containing text
    :param ignore_cols: indices of columns to ignore
    :param uncoded: strings to be interpreted as lacking a code
    """
    if os.path.splitext(input_file)[1] != '.csv':
        raise ValueError('Input must be a csv file')
    if os.path.splitext(output_file)[1] != '.pkl':
        raise ValueError('Output must be a pkl file')
    
    # Extract total set of features and codes
    
    with open(input_file, newline='') as f:
        # Process the file as a CSV file
        reader = csv.reader(f)
        
        # Find the headings (the first row of the file)
        headings = next(reader)
        # Restrict ourselves to a subset of columns (not containing text, and not ignored) 
        code_cols = sorted(set(range(len(headings))) - {text_col} - set(ignore_cols))
        # Group columns in pairs
        pair_indices = list(zip(code_cols[::2], code_cols[1::2]))
        pair_names = [headings[i][:-1].strip() for i in code_cols[::2]]
        
        # Find features and codes, and their frequencies
        vocab = Counter()
        pair_values = [Counter() for _ in pair_names]
        for vals in reader:
            # Find words in message
            msg = vals[text_col]
            vocab.update(msg.split())
            # Find codes
            for n, inds in enumerate(pair_indices):
                codes = [vals[i] for i in inds if vals[i] not in uncoded]
                pair_values[n].update(codes)
    
    # Assign an integer to each feature and code,
    # and create dicts to map to these integers
    
    vocab_list = sorted(vocab)
    vocab_dict = {w:i for i,w in enumerate(vocab_list)}
    V = len(vocab)
    
    pair_values_list = [sorted(vals) for vals in pair_values]
    pair_values_dict = []
    codes = []
    n_codes = 0
    for n, vals in enumerate(pair_values_list):
        pair_values_dict.append({c:i+n_codes for i,c in enumerate(vals)})
        n_codes += len(vals)
        for v in vals:
            codes.append(((pair_names[n], v), pair_values[n][v]))
    K = len(codes)
    
    # Save the features and codes to file
    
    output_name = os.path.splitext(output_file)[0]
    with open(output_name+'_vocab.pkl', 'wb') as f:
        pickle.dump(sorted(vocab.items()), f)
    with open(output_name+'_codes.pkl', 'wb') as f:
        pickle.dump(codes, f)
    
    print('Codes:')
    print(*codes, sep='\n')
    
    # Convert messages and codes to vectors
    
    def vectorise_features(message):
        """
        Convert a message to a feature vector
        :param message: string
        :return: numpy array
        """
        vec = np.zeros(V)
        for w in message.split():
            vec[vocab_dict[w]] += 1
        return vec
    
    def vectorise_codes(code_lists):
        """
        Convert names of codes to a vector
        :param code_lists: lists of lists of code names
        :return: numpy
        """
        vec = np.zeros(K, dtype='bool')
        for n, vals in enumerate(code_lists):
            for v in vals:
                vec[pair_values_dict[n][v]] = True
        return vec
    
    with open(input_file, newline='') as f:
        # Process the file as a CSV file
        reader = csv.reader(f)
        # Ignore headings
        next(reader)
        
        feature_vecs = []
        code_vecs = []
        
        for vals in reader:
            # Get feature vector
            msg = vals[text_col]
            feature_vecs.append(vectorise_features(msg))
            # Get code vector
            code_lists = [[vals[i] for i in inds if vals[i] not in uncoded]
                          for n, inds in enumerate(pair_indices)]
            code_vecs.append(vectorise_codes(code_lists))
        
    # Save to file
    
    with open(output_file, 'wb') as f:
        feature_mat = np.array(feature_vecs)
        code_mat = np.array(code_vecs)
        pickle.dump((feature_mat, code_mat), f)

if __name__ == "__main__":
    #preprocess_pairs('../data/malaria_original.csv', '../data/malaria.pkl', ignore_cols=[1,6])
    #preprocess_pairs('../data/wash_original.csv', '../data/wash.pkl', ignore_cols=[1,2,13,14])
    #preprocess_long('../data/nutrition_original.csv', '../data/nutrition.pkl', ignore_cols=[1,13,14])
    preprocess_long('../data/ANC_Delivery Training Set.xlsx - Short.csv', '../data/delivery.pkl')