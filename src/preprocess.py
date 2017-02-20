import csv, pickle, os, numpy as np
from collections import Counter
from warnings import warn

from features import get_global_set, feature_list_and_dict, vectorise

def save(feat_bags, code_vecs, code_names, output_file, directory='../data'):
    """
    Save features and codes to file
    :param feat_bags: list of dict-like objects (supervised learning input)
    :param code_vecs: boolean numpy matrix of codes (supervised learning output)
    - rows correspond to elements in feat_bags
    - columns correspond to elements in code_names
    :param code_names: list of names of codes
    :param output_file: name of output file (without .pkl file extension)
    - as well as saving to example.pkl, also saves to:
    - example_features.pkl (list of names of features, with frequencies)
    - example_features.txt (as above, but human-readable)
    - example_codes.pkl (list of names of codes, with frequencies)
    - example_codes.txt (as above, but human-readable)
    :param directory: directory of data files (default ../data)
    """
    N = len(feat_bags)
    K = len(code_names)
    
    if code_vecs.shape != (N,K):
        raise ValueError('Dimensions do not match')
    
    # Find the frequency of each code
    code_freq = code_vecs.sum(0)
    
    # Get the global set of features
    feat_set = get_global_set(feat_bags)
    feat_list, feat_dict = feature_list_and_dict(feat_set)
    # Convert messages to vectors
    feat_vecs = vectorise(feat_bags, feat_dict)
    # Find the document frequency of each feature
    feat_freq = (feat_vecs != 0).sum(0)
    
    # Save the features and codes to file,
    # as a list of names and frequencies
    # Convert from Numpy to Python data types
    
    feats = list(zip(feat_list, [int(x) for x in feat_freq]))
    with open(os.path.join(directory, output_file+'_features.pkl'), 'wb') as f:
        pickle.dump(feats, f)
    with open(os.path.join(directory, output_file+'_features.txt'), 'w') as f:
        for feat, count in feats:
            f.write('{}\t{}\n'.format(feat, count))
    
    codes = list(zip(code_names, [int(x) for x in code_freq]))
    with open(os.path.join(directory, output_file+'_codes.pkl'), 'wb') as f:
        pickle.dump(codes, f)
    with open(os.path.join(directory, output_file+'_codes.txt'), 'w') as f:
        for feat, count in codes:
            f.write('{}\t{}\n'.format(feat, count))
        
    
    print('Codes:')
    print(*codes, sep='\n')
    
    with open(os.path.join(directory, output_file+'.pkl'), 'wb') as f:
        pickle.dump((feat_vecs, code_vecs), f)

def preprocess_long(input_file, output_file, extractor, directory='../data', text_col=0, ignore_cols=(), convert=bool):
    """
    Preprocess a csv file to feature vectors and binary codes,
    where the input data has a 0 or 1 for each code and message
    :param input_file: input file name (without .csv file extension)
    :param output_file: output file name (without .pkl file extension)
    :param extractor: function mapping strings to bags of features
    :param directory: directory of data files (default ../data)
    :param text_col: index of column containing text
    :param ignore_cols: indices of columns to ignore
    :param convert: function to convert codes strings (e.g. bool or int)
    """
    
    # Extract features and codes
    # We can vectorise the codes immediately, but for features, we first need a global list
    
    feat_bags = []
    code_vecs = []
    
    with open(os.path.join(directory, input_file+'.csv'), newline='') as f:
        # Process the file as a CSV file
        reader = csv.reader(f)
        # Find the headings (the first row of the file)
        headings = next(reader)
        # Restrict ourselves to a subset of columns (not containing text, and not ignored) 
        code_cols = sorted(set(range(len(headings))) - {text_col} - set(ignore_cols))
        code_names = [headings[i] for i in code_cols]
        # Iterate through data
        for row in reader:
            # Get the bag of features, and the vector of codes
            feat_bags.append(extractor(row[text_col]))
            code_vecs.append(np.array([convert(row[i]) for i in code_cols], dtype='bool'))
    
    # Convert the list of code vectors to a matrix
    code_vecs = np.array(code_vecs)
    # Save the information
    save(feat_bags, code_vecs, code_names, output_file, directory)

def preprocess_pairs(input_file, output_file, extractor, directory='../data', text_col=0, ignore_cols=(), uncoded=('', 'NM')):
    """
    Preprocess a csv file to feature vectors and binary codes,
    where the input data has groups of codes,
    and each message has up to two codes from each group.
    Each group must take up exactly two columns.
    :param input_file: input file name (without .csv file extension)
    :param output_file: output file name (without .pkl file extension)
    :param extractor: function mapping strings to bags of features
    :param directory: directory of data files (default ../data)
    :param text_col: index of column containing text
    :param ignore_cols: indices of columns to ignore
    :param uncoded: strings to be interpreted as lacking a code
    """
    
    # Extract features and codes
    # We cannot vectorise these until we have a global list
    
    feat_bags = []
    code_sets = []
    
    with open(os.path.join(directory, input_file+'.csv'), newline='') as f:
        # Process the file as a CSV file
        reader = csv.reader(f)
        
        # Find the headings (the first row of the file)
        headings = next(reader)
        # Restrict ourselves to a subset of columns (not containing text, and not ignored) 
        code_cols = sorted(set(range(len(headings))) - {text_col} - set(ignore_cols))
        # Group columns in pairs
        pair_indices = list(zip(code_cols[::2], code_cols[1::2]))
        pair_names = [headings[i][:-1].strip() for i in code_cols[::2]]
        
        # Find features and codes
        for row in reader:
            # Find words in message
            feat_bags.append(extractor(row[text_col]))
            # Find codes
            row_code_set = set()
            for name, inds in zip(pair_names, pair_indices):
                # The code is recorded as a tuple (pair_name, value)
                # If a code is repeated, it is only counted once
                row_code_set |= {(name, row[i]) for i in inds if row[i] not in uncoded}
            code_sets.append(row_code_set)
    
    # Get the global set of codes, and convert to vectors
    
    codes = get_global_set(code_sets)
    K = len(codes)
    code_list, code_dict = feature_list_and_dict(codes)
    
    def vectorise_codes(set_of_codes):
        """
        Convert names of codes to a vector
        :param set_of_codes: set of code names
        :return: numpy array
        """
        vec = np.zeros(K, dtype='bool')
        for c in set_of_codes:
            vec[code_dict[c]] = True
        return vec
    
    code_vecs = np.array([vectorise_codes(x) for x in code_sets])
    
    # Save the information
    save(feat_bags, code_vecs, code_list, output_file, directory)

def preprocess_keywords(keyword_file, feature_file, output_file=None, directory='../data'):
    """
    Preprocess the keywords, converting words to feature indices
    The input file should have one line per code, with keywords separated by commas.
    Each keyword should be either a single word or a bigram.
    For readability, the line can begin with the name of the code, separated by a tab.
    e.g.: (note the tab character)
    "Emotional Causes	walwal, isla hadal"
    Note that the order of the codes should match the order used in the above functions.
    The output file will be a pickled list of lists of feature indices
    :param keyword_file: input file name (without .txt file extension)
    :param feature_file: file containing the global list of features (without .pkl file extension)
    :param output_file: output file name (without .pkl file extension)
    (default is the same as the keyword file, with a different file extension)
    :param directory: directory of data files (default ../data)
    """
    # Set output file name, if not given
    if output_file is None:
        output_file = keyword_file
    
    # Load features
    with open(os.path.join(directory, feature_file+'.pkl'), 'rb') as f:
        feats = pickle.load(f)
    feat_list = [x for x,_ in feats]  # Ignore frequency information
    feat_dict = {x:i for i,x in enumerate(feat_list)}  # Convert to a dict
    
    # Read keyword file
    with open(os.path.join(directory, keyword_file+'.txt')) as f:
        full_list = []
        for line in f:
            # Get the keywords
            parts = line.split('\t')
            keywords = [x.split() for x in parts[-1].split(',')]
            indices = []
            for k in keywords:
                # Lookup each keyword either as a single word, or as a bigram
                try:
                    if len(k) == 1:
                        indices.append(feat_dict['word', k[0]])
                    elif len(k) == 2:
                        indices.append(feat_dict['ngram', tuple(k)])
                    else:
                        raise ValueError('Keywords must be one or two words long')
                except KeyError:
                    warn("Keyword '{}' could not be found as a feature".format(' '.join(k)))
            # Add to the full list
            full_list.append(indices)
    
    # Save the keyword indices to file
    with open(os.path.join(directory, output_file+'.pkl'), 'wb') as f:
        pickle.dump(full_list, f)


if __name__ == "__main__":
    from features import bag_of_words, bag_of_ngrams, bag_of_variable_character_ngrams, combine
    
    ### Extract both single words and bigrams
    #functions = [bag_of_words, bag_of_ngrams]
    #kwargs = [{}, {'n': 2}]
    #feature_extractor = combine(functions, kwarg_params=kwargs)
    
    ### Preprocess individual files
    #preprocess_pairs('malaria_original', 'malaria', feature_extractor, ignore_cols=[1,6])
    #preprocess_pairs('wash_original', 'wash', feature_extractor, ignore_cols=[1,2,13,14])
    #preprocess_long('nutrition_original', 'nutrition', feature_extractor, ignore_cols=[1,13,14], convert=bool)
    #preprocess_long('ANC_Delivery Training Set.xlsx - Short', 'delivery', feature_extractor, convert=int)
    
    ### Preprocess keywords
    #preprocess_keywords('malaria_keywords', 'malaria_features')
    #preprocess_keywords('wash_keywords', 'wash_features')
    #preprocess_keywords('nutrition_keywords', 'nutrition_features')
    #preprocess_keywords('delivery_keywords', 'delivery_features')
