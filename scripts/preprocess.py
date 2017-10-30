"""preprocess.py: Contains general preprocessing functions for textual data"""

import spacy
from gensim.models import Word2Vec as word2vec
import numpy as np

np.random.seed(1234)

# A dict of some additional special words
X_WORDS = {"unknown": "<unk>", "start": "<start>", "end": "<end>", "digit": "<digit>"}

# Loads the spacy model
parser = spacy.load('en_core_web_md')


def tokenizer(sentence):
    """
    Tokenizes a sentence using spacy models.
    
    :param sentence: str    
    :returns tokens: list
    """
    doc = parser(sentence)
    tokens = [word.text for word in doc] # get a list of tokenized tokens
    return tokens


def add_boundary_tags(tokens):
    """
    Adds start and end tags to list of tokens
    
    :param tokens: list: list of tokenized words
    :returns str: [<start>, w1, w2...., wn, <end>]
    """
    return [X_WORDS["start"]] + tokens + [X_WORDS["end"]]


def preprocess(documents, to_lower=True, boundary_tags=False):
    """
    Preprocesses raw text - convert into lowercase add boundary tags
    
    :param documents: list: of str
    :param to_lower: bool: whether to convert text into lowercase(default=True)
    :param boundary_tags: bool: whether to keep boundary tags or not(start, end)
    :returns processed: list: of list: of str: a list of lists of words
    """
    processed = list() 
    
    for doc in documents:
        
        # Convert into lowercase if flag is set
        if to_lower:
            doc = doc.lower()
        tokens = tokenizer(doc)
        if boundary_tags:
            tokens = add_boundary_tags(tokens)
        processed.append(tokens)
        
    return processed


def to_indices(documents, to_ix):
    """
    Converts documents into a list of indices.
    
    :param documents: list: of list: of str: a list of lists of words
    :param to_ix: dict: a word to index mapping
    :returns indices: list: of list: of int: a list of lists of word indices
    """    
    indices = list()
    
    for doc in documents:
        tokens = list()
        for word in doc:
            try:
                # Look for the word in dict
                tokens.append(to_ix[word])
            except:
                # If not found then add a special word for unknown
                tokens.append(to_ix[X_WORDS["unknown"]])
        indices.append(tokens)
        
    return indices


def w2v_word_mapping(model_path):
    """
    Returns mapping of words to indices and vice-versa.
    In addition to a numpy matrix representation of
    pre-trained word vectors with gensim.
    
    :param model_path: str: Relative path to the pre-trained gensim model    
    :returns (word_vectors: np.array: of float: A matrix representation of gensim word vectors,
              index_to_word: list: Index to word mapping,
              word_to_index: dict: Word to Index mapping)
    """
    
    # Load Word Vector Model and get a list of vocab
    wv_model = word2vec.load(model_path)
    index_to_word = list(wv_model.wv.vocab.keys())
   
    word_vectors = list()
    
    # Populate matrix of word vectors
    for word in index_to_word:
        word_vectors.append(wv_model[word])
    
    # Add a special words(unknow, start, end)
    index_to_word += X_WORDS.values()
    
    # Create a reverse mapping for words
    word_to_index = dict((word, idx) for idx, word in enumerate(index_to_word))    
    
    for word in X_WORDS:
        # A random_vector for special words
        random_vector = np.random.rand(wv_model.vector_size)
        word_vectors.append(random_vector)
    
    return np.array(word_vectors), index_to_word, word_to_index


def data_word_mapping(documents):
    """
    Returns unique words in a list of strings
    
    :param documents: list: a list of strings    
    :returns (None, index_to_word: list: Index to word mapping,
              word_to_index: dict: Word to Index mapping)
    """
    
    # If type of documents is a list of words then join them together
    if type(documents[0]) == list:
        documents = [" ".join(doc) for doc in documents]
        
    vocab = (" ".join(documents).split()) + [X_WORDS["unknown"]] # End tags will already be there
    index_to_word = np.unique(vocab)
    word_to_index = dict((word, idx) for idx, word in enumerate(index_to_word))
    
    return None, index_to_word, word_to_index


def get_word_mappings(documents=None, w2v_path=None):
    """
    Returns mapping of words to indices and vice-versa.
    If the `w2v_path` is given then this will also return 
    a numpy matrix representation of pre-trained word vectors with gensim.
    
    :param documents: list: of str: a list of documents/sentences/paragraphs
    :param w2v_path: str: Relative path for pre-trained gensim model
    
    :returns (word_vectors: np.array: of float: A matrix representation of gensim word vectors,
              index_to_word: list: Index to word mapping,
              word_to_index: dict: Word to Index mapping)
    """
    if documents:
        return data_word_mapping(documents)
    elif w2v_path:
        return w2v_word_mapping(w2v_path)
    else:
        print("Provide either a list of documents or path to a pre-trained gensim model.")
        
        
def train_test_split(dataset, test_size=0.10):
    """
    Splits the dataset into training and test sets.
    Each element of 'dataset' has to be a tuple where
    first index is input for the model and second index contains output.
    
    :param dataset: tuple: of (list, list): a tuple with one input and output sample
    :param test_size: int/float: if a float value is given than that portion of 'dataset'(default=0.10)
     will be made the test size. An integer value simply represents the count of test samples/
    :returns (train_data: tuple: of (list, list), test_data: tuple: of (list, list)) 
    """
    # Let there be some randomness
    np.random.shuffle(dataset)
    
    # If test_size is float then get number of sample for that proportion
    if type(test_size) == float:
        test_size = int(len(dataset) * test_size)

    test_data = dataset[:test_size]
    train_data = dataset[test_size:]
    
    return train_data, test_data