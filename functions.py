import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics as m

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

def one_hot_c(classes):
    """
    classes: unique array of labels from the data set 
            e.g. classes = [1, 2, 3] or ["dog", "cat"] ...
    """
    u=[[1 if h==c else 0 for h in classes] for c in classes]
    return {classes[k]: u[k] for k in range(len(classes))}

def one_hot(x):
    if len(x.shape)>1: x=x.reshape(np.max(x.shape),)
    levels = one_hot_c(np.unique(x))
    return np.array([levels[l] for l in x for level in levels.keys() if l == level])

def reverse_one_hot(x, levels):
    levels = one_hot_c(levels)
    return np.array([l[0] for l in levels.items() for t in x if all(t==l[1])])

def to_nominal(x):
    classes = np.unique(x)
    for i, c in enumerate(classes):
        x = np.where(x==c, i, x)
    return x.astype(np.int16)
    
def reverse_nominal(x, levels):
    x = x.astype(levels.dtype.str)
    for i, c in enumerate(levels):
        i = str(i)
        x = np.where(x==i, c, x)
    return x

def plot_roc(y_test, y_pred, y_prob):
    fpr, tpr, _ = m.roc_curve(y_test, y_pred)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % m.roc_auc_score(y_test, y_prob))
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def confusion_matrix(y_true, y_pred):
    """
    |        | prediction |
    +--------+------------+
    | actual |    cm      |
    """
    levels = np.unique(y_true)
    C = np.zeros(shape = [len(levels), len(levels)], dtype="int")
    for i, true_class in enumerate(levels):
        for j, pred_class in enumerate(levels):
            C[i, j] = np.sum((y_true==true_class) & (y_pred==pred_class))
    return pd.DataFrame(C, 
                        columns=[f"p_{k+1}" for k in range(len(levels))],
                        index=[f"t_{k+1}" for k in range(len(levels))])
    
def accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred).to_numpy()
    return np.sum(np.diag(cm)) / np.sum(cm)


