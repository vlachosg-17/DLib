import numpy as np
from numpy.core.fromnumeric import argmax

def softmax(x):
    if len(x.shape) == 1: return np.exp(x) / np.sum(np.exp(x))
    else: return np.array([softmax(x_i) for x_i in x])
    
def relu(x, der=False):
    y = np.copy(x)
    if not der:
        y[x<=0]=0
        return y
    else:
        y[x<=0]=0
        y[x>0]=1
        return y

def sigmoid(x, der=False):
    '''sigmoid funtion or derivative'''
    if not der:
        return 1 / (1 + np.exp(-x))
    else:
        return sigmoid(x) * (1-sigmoid(x))

def split(X, y, val_per):
    trX, valX = np.split(X, [int(X.shape[0]*(1-val_per))])
    trY, valY = np.split(y, [int(X.shape[0]*(1-val_per))])
    # print(X.shape, y.shape)
    # print(trX.shape, trY.shape, valX.shape, valY.shape)
    # print(int(X.shape[0]*(1-val_per)))
    return trX, trY, valX, valY

def shuffle(X, Y):
    new_raws = np.random.choice(np.arange(X.shape[0]), size = X.shape[0], replace=False)
    return X[new_raws], Y[new_raws]

def one_hot(classes):
    """
    classes: unique array of labels from the data set 
            e.g. classes = [1, 2, 3] or ["dog", "cat"] ...
    """
    u=[[1 if h==c else 0 for h in classes] for c in classes]
    return {classes[k]: u[k] for k in range(len(classes))}

def confusion_matrix(y_true, y_pred):
    levels = np.unique(y_true)
    print(levels)
    C = np.zeros(shape = [len(levels), len(levels)])
    for i, l1 in enumerate(levels):
        for j, l2 in enumerate(levels):
            C[i, j] = np.sum((y_true==l1) & (y_pred==l2))
    return C

def accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return np.sum(np.diag(cm)) / np.sum(cm)


