
# coding: utf-8

# In[76]:


import torch
from torch.autograd import Variable as Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scripts.preprocess import *
from sklearn.metrics import f1_score


# In[2]:


from gensim.models import Word2Vec as word2vec
import numpy as np

np.random.seed(1234)

# A dict of some additional special words
X_WORDS = {"unknown": "<unk>", "start": "<start>", "end": "<end>", "digit": "<digit>"}


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
        tokens = doc.split()
        if boundary_tags:
            tokens = add_boundary_tags(tokens)
        processed.append(tokens)
        
    return processed


def to_indices(document, to_ix):
    """
    Converts documents into a list of indices.
    
    :param documents: list: of list: of str: a list of lists of words
    :param to_ix: dict: a word to index mapping
    :returns indices: list: of list: of int: a list of lists of word indices
    """    
    indices = list()
        
    for word in document:
        try:
            # Look for the word in dict
            indices.append(to_ix[word])
        except:
            # If not found then add a special word for unknown
            indices.append(to_ix[X_WORDS["unknown"]])
        
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


def get_word_mappings(documents):
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


# In[3]:


TRAIN_FILE = "data/penn/train.txt"
TEST_FILE = "data/penn/test.txt"

train_data = preprocess(open(TRAIN_FILE, 'r').readlines(), boundary_tags=True)
test_data = preprocess(open(TEST_FILE, 'r').readlines(), boundary_tags=True)


# In[4]:


# Get pre-trained word vectors and indices mappings
WORD_VECTORS, INDEX_TO_WORD, WORD_TO_INDEX = get_word_mappings(documents=train_data)


# In[5]:


train_data = [(sample[:-1], sample[1:]) for sample in train_data]
test_data = [(sample[:-1], sample[1:]) for sample in test_data]


# In[6]:


TRAIN_DATA = [(to_indices(x, WORD_TO_INDEX),
               to_indices(y, WORD_TO_INDEX)) for x, y in train_data]
TEST_DATA = [(to_indices(x, WORD_TO_INDEX),
               to_indices(y, WORD_TO_INDEX)) for x, y in test_data]


# In[93]:


class LSTM(nn.Module):
    
    def __init__(self, hidden_dim, embedding_dim=None, vocab_size=None,
                 embeddings=None, dropout=0.5, has_cuda=True):
        """
        RNN Classifier performs multi class classification and Sequence Tagging.
        
        :param hidden_dim: int: Dimension of hidden layer
        :param embedding_dim: int: Word embeddings dimension        
        :param vocab_size: int: Number of unique words in the dataset
        :param embeddings: numpy.matrix: Pre-trained word words
        :param dropout: float: dropout value
        :param has_cuda: bool: Whether to run this model on gpu or not
        """        
        super(LSTM, self).__init__()
            
        self.hidden_dim = hidden_dim
        self.has_cuda = has_cuda
        num_labels = vocab_size
        
        # Setup embeddings
        if not embeddings is None:
            embedding_dim = embeddings.shape[1]
            self.word_embeddings = nn.Embedding(*embeddings.shape)
            self.word_embeddings.weight.data.copy_(torch.from_numpy(embeddings))
        elif embedding_dim and vocab_size:
            self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        else:
            print("You must provide either a pre-trained word vectors matrix as                  'embeddings' or 'embedding_dim' and 'vocab_size'")
            return None
        
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, dropout=dropout)
        self.h2o = nn.Linear(hidden_dim, num_labels)
        self.hidden = self.init_hidden()
        
    def init_hidden(self):
        """
        Initialize the hidden states for RNN
        
        :returns : tuple: of (autograd.Variable, autograd.Variable)
        """
        if self.has_cuda:
            return (Variable(torch.zeros(1, 1, self.hidden_dim).cuda()),
                    Variable(torch.zeros(1, 1, self.hidden_dim).cuda()))
        else:
            return (Variable(torch.zeros(1, 1, self.hidden_dim)),
                    Variable(torch.zeros(1, 1, self.hidden_dim)))
    
    def forward(self, tokens):
        """
        Forward-Pass for RNN, which returns the probability scores of classes. 
        
        :param tokens: autograd.Variable: a list of indices as torch tensors
        
        :returns: scores: autograd.Variable: Final score for the model
        """
        embeds = self.drop(self.word_embeddings(tokens))
        output, self.hidden = self.lstm(embeds.view(len(tokens), 1, -1), self.hidden)
        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())
        
        final_output = self.h2o(F.tanh(self.drop(output.view(len(tokens), -1))))
        scores = F.log_softmax(final_output)
       
        return scores


# In[94]:


def is_cuda():
    return True if torch.cuda.is_available() else False

CUDA = is_cuda()

if CUDA:
    torch.cuda.manual_seed(1234)
else:
    torch.manual_seed(1234)


# In[95]:


def to_Variable(sequence, has_cuda=is_cuda(), ttype=torch.LongTensor):
    """
    Convert a list of words to list of pytorch tensor variables
    
    :param tokens: list: of str: a list of words in a sentence
    :param has_cuda: bool: does this machine has cuda
    :param ttype: torch tensor type
    :returns : autograd.Variable
    """
    if has_cuda:
        tensor = ttype(sequence).cuda()
    else:
        tensor = ttype(sequence)
        
    return Variable(tensor)


def get_accuracy(x, y):
    """
    Calculates percent of similar instances among two numpy arrays
    
    :param x: np.array
    :param y: np.array
    
    :returns accuracy: float
    """
    accuracy = np.sum(x == y)/len(x)
    return accuracy


def get_metrics(x, y, num_labels):
    """
    Get F1 Score and accuracy for a predicted and target values.
    
    :param x: np.array
    :param y: np.array
    :param num_labels: number of unique labels in dataset
    :returns (total_f1_score: float, total_accuracy: float)
    """    
    total_f1_score = 0
    total_accuracy = 0
    
    for inp, out in zip(x, y):        
        f1 = f1_score(inp, list(out), labels=np.arange(num_labels), average='macro')
        
        total_f1_score += f1
        total_accuracy += get_accuracy(inp, out)        
        
    return total_f1_score/len(x), total_accuracy/len(x)


def predict(model, x):
    """
    Get the prediction as the class name from trained model.
    
    :param model: pytorch model
    :param x: str: a test document
    
    :returns tag: int: class id for the input
    """
    # Set model to evalution state to turn off dropout
    model.eval()
    x = to_Variable(x)
    yhat = model(x)
    _, tag = yhat.max(1)
    
    return tag.data.cpu().numpy()


def evaluate(model, eval_data, num_labels):
    """
    Evaluates the accuracy for the model in the global scope.
    
    :param model: PyTorch Model
    :param eval_data: tuple: as (inputs, targets)
    :param num_labels: number of unique labels in dataset
    :returns (f1_score: float, accuracy: float)
    """    
    # Turn on the evaluation state to ignore dropouts
    model.eval()
    results = [predict(model, x) for x, y in eval_data]
    f1_score, accuracy = get_metrics(np.array([y for x, y in eval_data]), results, num_labels)
    return f1_score, accuracy


# In[108]:


hparams = {'hidden_dim': 512, 'learning_rate': 0.1,
           'epochs': 10, 'dropout': 0.5, 'embedding_dim': 400,
           'vocab_size': len(INDEX_TO_WORD)}

model = LSTM(hidden_dim=hparams['hidden_dim'], embedding_dim=hparams['embedding_dim'],
             vocab_size=hparams['vocab_size'], dropout=hparams['dropout'])

loss_fn = nn.NLLLoss()
optimizer =  optim.SGD(model.parameters(), lr=hparams['learning_rate'],
                                          weight_decay=0.0001, momentum=0.9)

if CUDA:
    model = model.cuda()
    loss_fn = loss_fn.cuda()


# In[109]:


print_after = 1000
test_after = 20000

for epoch in range(hparams['epochs']):

    count = 0
    avg_loss = 0
    epoch_loss = 0
    test_f1_score = 0
    last_test_f1_score = 0

    # Randomly shuffle the dataset
    np.random.shuffle(TRAIN_DATA)
    np.random.shuffle(TEST_DATA)

    for tokens, labels in TRAIN_DATA:

        x, y = to_Variable(tokens), to_Variable(labels)        

        y_ = model(x)        
        loss = loss_fn(y_, y)

        # Initialize hidden states to zero
        model.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        loss_value = loss.data.cpu().numpy()
        avg_loss += loss_value
        epoch_loss += loss_value

        count += 1

        if count%print_after == 0:
            print("Epoch % d - Average Loss after %d samples: %f" % (epoch, count,
                                                                     avg_loss/print_after))
            avg_loss = 0

        if count%test_after == 0:
            train_f1_score, train_accuracy = evaluate(model, TRAIN_DATA[:len(TEST_DATA)],
                                                                        len(WORD_TO_INDEX))
            print("Epoch % d - Train F1 Score, Accuracy after %d samples: %f, %f"% (epoch,
                                                                                        count,
                                                                                        train_f1_score,
                                                                                        train_accuracy))

            test_f1_score, test_accuracy = evaluate(model, TEST_DATA,
                                                    len(WORD_TO_INDEX)) # So that we can use it later
            print("Epoch % d - Test F1 Score, Accuracy after %d samples: %f, %f" % (epoch,
                                                                                       count,
                                                                                       test_f1_score,
                                                                                       test_accuracy))
            model.train() # Get the model back to training state
    
    l = (epoch_loss/len(TRAIN_DATA))[0]
    print("AVERAGE EPOCH LOSS and PERPLEXITY:", (l, np.power(2, l)))


# In[107]:


# print("AVERAGE EPOCH LOSS and PERPLEXITY:", (l, np.power(2, l)))


# In[91]:



# for word in predict(model, TRAIN_DATA[10][0]):
#     print(INDEX_TO_WORD[word])


# In[92]:


# for word in TRAIN_DATA[10][1]:
#     print(INDEX_TO_WORD[word])

