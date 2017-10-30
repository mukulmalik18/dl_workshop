import torch
from torch.autograd import Variable as Variable
import torch.nn as nn
import torch.nn.functional as F

import torch
from torch.autograd import Variable as Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LSTM(nn.Module):
    
    def __init__(self, hidden_dim, index_to_tag, embedding_dim=None, vocab_size=None,
                 embeddings=None, dropout=0.5, num_layers=1, batch_size=1,
                 bidirectional=False, has_cuda=True):
        """
        LSTM Classifier performs multi class classification and Sequence Tagging.
        
        :param hidden_dim: int: Number of hidden layers in the LSTM
        :param tag_to_index: dict: Mapping of output labels with indices
        :param embedding_dim: int: Word embeddings dimension        
        :param vocab_size: int: Number of unique words in the dataset
        :param embeddings: numpy.matrix: Pre-trained word words
        :param dropout: float: dropout value
        :param num_layers: int: Number of LSTM layers
        :param batch_size: int: Number of samples in one batch
        :param bidirectional: bool: Bidirectional LSTM or not 
        :param has_cuda: bool: Whether to run this model on gpu or not
        """        
        super(LSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.has_cuda = has_cuda
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.index_to_tag = index_to_tag
        num_labels = len(index_to_tag)
        
        # If directional set directions to 2
        self.directions = 1
        if bidirectional:
            self.directions = 2
        
        # Setup embeddings
        if not embeddings is None:
            embedding_dim = embeddings.shape[1]
            self.word_embeddings = nn.Embedding(*embeddings.shape)
            self.word_embeddings.weight.data.copy_(torch.from_numpy(embeddings))
        elif embedding_dim and vocab_size:
            self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        else:
            print("You must provide either a pre-trained word vectors matrix as\
                  'embeddings' or 'embedding_dim' and 'vocab_size'")
            return None
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, dropout=dropout,
                            num_layers=num_layers, batch_first=False,
                            bidirectional=bidirectional) # Coz I like my batch first ;)
        self.h2o = nn.Linear(hidden_dim*self.directions, num_labels) # Concatenates if bi-directional
        self.hidden = self.init_hidden()
        
    def init_hidden(self):
        """
        Initialize the hidden states for LSTM
        
        :returns : tuple: of (autograd.Variable, autograd.Variable)
        """
        if self.has_cuda:
            return (Variable(torch.zeros(self.num_layers*self.directions,
                                                  self.batch_size,
                                                  self.hidden_dim).cuda()),
                    Variable(torch.zeros(self.num_layers*self.directions,
                                                  self.batch_size,
                                                  self.hidden_dim).cuda()))
        else:
            return (Variable(torch.zeros(self.num_layers*self.directions,
                                                  self.batch_size,
                                                  self.hidden_dim)),
                    Variable(torch.zeros(self.num_layers*self.directions,
                                                  self.batch_size,
                                                  self.hidden_dim)))
    
    def forward(self, tokens):
        """
        Forward-Pass for LSTM, which returns the probability scores of classes. 
        
        :param tokens: autograd.Variable: a list of indices as torch tensors
        
        :returns: label_scores: autograd.Variable: probability scores for classes
        """
        embeds = self.word_embeddings(tokens)
        output, self.hidden = self.lstm(embeds.view(len(tokens), self.batch_size, -1), self.hidden)
        final_output = self.h2o(output.view(len(tokens), -1)) # What's gonna happen with batch?
        label_probs = F.log_softmax(final_output)
        return label_probs