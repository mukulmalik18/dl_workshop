"""utils.py: Contains general PyTorch based/related helper functions."""

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable as Variable
from sklearn.metrics import f1_score as fscore

def is_cuda():
    return True if torch.cuda.is_available() else False

def to_Variable(sequence):
    """
    Convert a list of words to list of pytorch tensor variables
    
    :param tokens: list: of str: a list of words in a sentence    
    :returns : autograd.Variable
    """
    if is_cuda():
        tensor = torch.cuda.LongTensor(sequence)
    else:
        tensor = torch.LongTensor(sequence)
        
    return Variable(tensor)


def get_optimizer(model, learning_rate, name="SGD"):
    """
    Get an optimizer of your choice. Default values for
    optimizer's parameters are set based on common practices.
    
    :param model: model to be trained
    :param learning_rate: float
    :param name: str: string value for optimizer names(default="SGD",options=["SGD", "Adam", "RMSProp"])
    :returns torch.optimizer
    """
    
    optimizer_dict = {"SGD": optim.SGD(model.parameters(), lr=learning_rate,
                                          weight_decay=0.0001, momentum=0.9),
                      "Adam": optim.Adam(model.parameters(), lr=learning_rate,
                                           weight_decay=0.0001),
                      "RMSProp": optim.RMSprop(model.parameters(), lr=learning_rate,
                                              weight_decay=0.0001)}
    return optimizer_dict[name]

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
        f1 = fscore(inp, list(out), labels=np.arange(num_labels), average='weighted')
        
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