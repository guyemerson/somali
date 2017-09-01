import csv, pickle, os, numpy as np
from warnings import warn

from features import get_global_set, feature_list_and_dict, vectorise, document_frequency, Vectoriser

def save_pkl_txt(name_freq, filename, directory='../data'):
    """
    Save a list of names and frequencies, in both .pkl and .txt format
    :param name_freq: list of (name, frequency) pairs
    :param filename: name of output files (without file extension)
    :param directory: directory of data files (default ../data)
    """
    with open(os.path.join(directory, filename+'.pkl'), 'wb') as f:
        pickle.dump(name_freq, f)
    with open(os.path.join(directory, filename+'.txt'), 'w') as f:
        for name, freq in name_freq:
            f.write('{}\t{}\n'.format(name, freq))

def save(msgs, code_vecs, code_names, output_file, extractor=None, vectoriser=None, raw=False, directory='../data'):
    """
    Save features and codes to file
    :param msgs: list of strings (supervised learning input)
    :param code_vecs: boolean numpy matrix of codes (supervised learning output)
    - rows correspond to elements in feat_bags
    - columns correspond to elements in code_names
    :param code_names: list of names of codes
    :param output_file: name of output file (without .pkl file extension)
    - as well as saving to example.pkl, also saves to:
    - example_codes.pkl (list of names of codes, with frequencies)
    - example_codes.txt (as above, but human-readable)
    - and if a feature extractor is given rather than a vectoriser, also saves to:
    - example_features.pkl (list of names of features, with frequencies)
    - example_features.txt (as above, but human-readable)
    :param extractor: function mapping strings to bags of features
    :param vectoriser: function mapping lists of strings to numpy arrays
    :param raw: if true, save raw messages, rather than extracting features
    :param directory: directory of data files (default ../data)
    """
    # Check that input dimensions match
    N = len(msgs)
    K = len(code_names)
    if code_vecs.shape != (N,K):
        raise ValueError('Dimensions do not match')
    
    # Convert the messages to feature vectors
    if raw:
        feat_vecs = msgs
    elif vectoriser:
        feat_vecs = vectoriser(msgs)
    elif extractor:
        # If we just have a feature extractor, we must define indices of features
        # Extract features
        feat_bags = [extractor(m) for m in msgs]
        # Get the global set of features
        feat_set = get_global_set(feat_bags)
        feat_list, feat_dict = feature_list_and_dict(feat_set)
        # Convert messages to vectors
        feat_vecs = vectorise(feat_bags, feat_dict)
        # Find the document frequency of each feature, and save features to file
        feat_freq = (feat_vecs != 0).sum(0)
        feats = list(zip(feat_list, [int(x) for x in feat_freq]))  # Convert from Numpy to Python data types
        save_pkl_txt(feats, output_file+'_features', directory)
    else:
        raise ValueError('processing method not specified')
    
    # Find the frequency of each code
    code_freq = code_vecs.sum(0)
    
    # Save the codes to file
    codes = list(zip(code_names, [int(x) for x in code_freq]))  # Convert from Numpy to Python data types
    save_pkl_txt(codes, output_file+'_codes', directory)
    
    print('Codes:')
    print(*codes, sep='\n')
    
    # Save the input and output matrices
    with open(os.path.join(directory, output_file+'.pkl'), 'wb') as f:
        pickle.dump((feat_vecs, code_vecs), f)

def preprocess_long(input_file, output_file, extractor=None, vectoriser=None, raw=False, directory='../data', text_col=0, ignore_cols=(), convert=bool):
    """
    Preprocess a csv file to feature vectors and binary codes,
    where the input data has a 0 or 1 for each code and message
    :param input_file: input file name (without .csv file extension)
    :param output_file: output file name (without .pkl file extension)
    :param extractor: function mapping strings to bags of features
    :param vectoriser: function mapping lists of strings to numpy arrays
    :param raw: if true, save raw messages, rather than extracting features
    :param directory: directory of data files (default ../data)
    :param text_col: index of column containing text
    :param ignore_cols: indices of columns to ignore
    :param convert: function to convert code strings (e.g. bool or int)
    """
    if not raw:
        if extractor is None and vectoriser is None:
            raise TypeError('Either extractor or vectoriser must be given')
        if extractor and vectoriser:
            raise TypeError('Only one of extractor and vectoriser should be given')
    
    # Extract features and codes
    # We can vectorise the codes immediately, but for features, we first need a global list
    
    msgs = []
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
            msgs.append(row[text_col])
            code_vecs.append(np.array([convert(row[i]) for i in code_cols], dtype='bool'))
    
    # Convert the list of code vectors to a matrix
    code_vecs = np.array(code_vecs)
    
    # Save the information
    save(msgs, code_vecs, code_names, output_file, extractor, vectoriser, raw, directory)

def preprocess_pairs(input_file, output_file, extractor=None, vectoriser=None, raw=False, directory='../data', text_col=0, ignore_cols=(), uncoded=('', 'NM')):
    """
    Preprocess a csv file to feature vectors and binary codes,
    where the input data has groups of codes,
    and each message has up to two codes from each group.
    Each group must take up exactly two columns.
    :param input_file: input file name (without .csv file extension)
    :param output_file: output file name (without .pkl file extension)
    :param extractor: function mapping strings to bags of features
    :param vectoriser: function mapping lists of strings to numpy arrays
    :param raw: if true, save raw messages, rather than extracting features
    :param directory: directory of data files (default ../data)
    :param text_col: index of column containing text
    :param ignore_cols: indices of columns to ignore
    :param uncoded: strings to be interpreted as lacking a code
    """
    if not raw:
        if extractor is None and vectoriser is None:
            raise TypeError('Either extractor or vectoriser must be given')
        if extractor and vectoriser:
            raise TypeError('Only one of extractor and vectoriser should be given')
    
    # Extract features and codes
    # We cannot vectorise these until we have a global list
    
    msgs = []
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
            msgs.append(row[text_col])
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
    save(msgs, code_vecs, code_list, output_file, extractor, vectoriser, raw, directory)

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

def iter_msgs(input_files, directory='../data', text_col=0):
    """
    Iterate through messages in multiple files
    :param input_files: single filename, or list of filenames (without .csv file extension)
    :param directory: directory of data files (default ../data)
    :param text_col: index of column containing text (default 0)
    :return: iterator yielding messages
    """
    # If only one file is given, convert to a list
    if isinstance(input_files, str):
        input_files = [input_files]
    if isinstance(text_col, int):
        text_col = [text_col] * len(input_files)
    # Iterate through files
    for filename, column in zip(input_files, text_col):
        with open(os.path.join(directory, filename+'.csv'), newline='') as f:
            # Process the file as a CSV file
            reader = csv.reader(f)
            # Ignore headings
            next(reader)
            # Iterate through messages
            for row in reader:
                yield row[column]

def extract_features_and_idf_from_messages(msgs, output_file, extractor, threshold=None, directory='../data'):
    """
    Extract features from all messages, and filter by document frequency
    Creates a Vectoriser that can convert messages to feature vectors weighted by idf 
    :param msgs: iterable of messages
    :param output_file: name of output file (without .pkl file extension)
    - as well as saving to example.pkl, also saves to:
    - example_features.pkl (list of names of features, with frequencies)
    - example_features.txt (as above, but human-readable)
    :param extractor: function mapping strings to bags of features
    :param threshold: minimum document frequency to keep a feature
    :param directory: directory of data files (default ../data)
    :return: vectoriser
    """
    # Get iterator over bags of features
    bags = (extractor(m) for m in msgs)
    # Get document frequency
    freq = document_frequency(bags)
    # Filter out rare features
    if threshold is not None:
        freq = {feat:n for feat, n in freq.items() if n >= threshold}
    # Assign indices to features
    feat_list, feat_dict = feature_list_and_dict(freq.keys())
    # Get idf array
    idf = np.empty(len(feat_list))
    for feat, n in freq.items():
        idf[feat_dict[feat]] = 1/n
    # Create and save Vectoriser
    vectoriser = Vectoriser(extractor, feat_dict, idf)
    if output_file is not None:
        with open(os.path.join(directory, output_file+'.pkl'), 'wb') as f:
            pickle.dump(vectoriser, f)
        # Save list of features
        feat_freq = [(feat, freq[feat]) for feat in feat_list]
        save_pkl_txt(feat_freq, output_file+'_features', directory)
    return vectoriser

def extract_features_and_idf(input_files, output_file, extractor, threshold=None, directory='../data', text_col=0):
    """
    Extract features from all messages, and filter by document frequency
    Creates a Vectoriser that can convert messages to feature vectors weighted by idf 
    :param input_files: single filename, or list of filenames (without .csv file extension)
    :param output_file: name of output file (without .pkl file extension)
    - as well as saving to example.pkl, also saves to:
    - example_features.pkl (list of names of features, with frequencies)
    - example_features.txt (as above, but human-readable)
    :param extractor: function mapping strings to bags of features
    :param threshold: minimum document frequency to keep a feature
    :param directory: directory of data files (default ../data)
    :param text_col: index of column containing text (default 0)
    :return: vectoriser
    """
    return extract_features_and_idf_from_messages(iter_msgs(input_files, directory, text_col),
                                                  output_file, extractor, threshold, directory)


if __name__ == "__main__":
    # TODO command line
    
    ### Extract both single words and bigrams
    
    from features import bag_of_words, bag_of_ngrams, bag_of_variable_character_ngrams, combine, apply_to_parts
    functions = [bag_of_words, bag_of_ngrams, bag_of_variable_character_ngrams]
    kwargs = [{}, {'n': 2}, {'min_n': 4, 'max_n':5}]
    feature_extractor = apply_to_parts(combine(functions, kwarg_params=kwargs), '&&&')
    
    ### Define feature vectors based on a whole corpus
    
    input_files = ['malaria_original', 'wash_original', 'nutrition_original', 'ANC_Delivery Training Set.xlsx - Short']
    extract_features_and_idf(input_files, 'four_combined_word_bigram_char', feature_extractor, 3)
    
    ### Preprocess individual files with an extractor
    '''
    preprocess_pairs('malaria_original', 'malaria', feature_extractor, ignore_cols=[1,6])
    preprocess_pairs('wash_original', 'wash', feature_extractor, ignore_cols=[1,2,13,14])
    preprocess_long('nutrition_original', 'nutrition', feature_extractor, ignore_cols=[1,13,14], convert=bool)
    preprocess_long('ANC_Delivery Training Set.xlsx - Short', 'delivery', feature_extractor, convert=int)
    '''
    ### Preprocess individual files with a vectoriser
    
    with open('../data/four_combined_word_bigram_char.pkl', 'rb') as f:
        vecr = pickle.load(f)
    preprocess_pairs('anc_fully_labelled', 'delivery_new', vectoriser=vecr, ignore_cols=[0,1,2,3,5,6,13,14], text_col=4)
    #preprocess_pairs('malaria_original', 'malaria2', vectoriser=vecr, ignore_cols=[1,6])
    #preprocess_pairs('wash_original', 'wash2', vectoriser=vecr, ignore_cols=[1,2,13,14])
    #preprocess_long('nutrition_original', 'nutrition2', vectoriser=vecr, ignore_cols=[1,13,14], convert=bool)
    #preprocess_long('ANC_Delivery Training Set.xlsx - Short', 'delivery2', vectoriser=vecr, convert=int)
    
    ### Preprocess keywords
    
    #preprocess_keywords('malaria_keywords', 'four_combined_word_bigram_char_features')
    #preprocess_keywords('wash_keywords', 'four_combined_word_bigram_char_features')
    #preprocess_keywords('nutrition_keywords', 'four_combined_word_bigram_char_features')
    #preprocess_keywords('delivery_keywords', 'four_combined_word_bigram_char_features')
    
