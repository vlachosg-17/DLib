# from . import np, pd, plt, mcs
from timeit import timeit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as mcs
from scipy.signal import convolve2d, correlate2d

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

def eye_levels(classes):
    """
    classes: unique array of labels from the data set 
            e.g. classes = [1, 2, 3] or ["dog", "cat"] ...
    """
    return {c: np.where(classes!=c, 0, 1) for c in classes}

def one_hot(x):
    if len(x.shape)>1: x=x.reshape(np.max(x.shape),)
    levels = eye_levels(np.unique(x))
    return np.array([levels[l] for l in x for level in levels.keys() if l == level])

def reverse_one_hot(x, classes):
    levels = eye_levels(classes)
    return np.array([l[0] for t in x for l in levels.items() if all(t==l[1])])

def to_nominal(x):
    classes = np.unique(x)
    for k, c in enumerate(classes):
        x = np.where(x==c, k, x)
    return x.astype(np.int32)

def to_01(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

def rot180(X):
    if X.ndim>2:
        if X.ndim==3: return np.array([rot180(X[c]) for c in range(X.shape[0])])
        if X.ndim==4: return np.array([[rot180(X[c1, c0]) for c0 in range(X.shape[1])] for c1 in range(X.shape[0])])   
    return np.flipud(np.fliplr(X))

def conv2d(X, K, xy_stride=(1,1), tp="valid"):
    """
    X: (C0, H, W)
    K: (C1, C0, Hk, Wk)
    (Hz, Wz) = (H-Hk, W-Wk) 
    """
    def ixs(start, step):
        """ Provides ranges of lists """
        return list(range(start, start + step))
    def fm_blocks(K, xy_steps, xy_stride):
        nsteps_x, nsteps_y = xy_steps
        stride_x, stride_y = xy_stride
        k_x, k_y= K.shape[-2], K.shape[-1]
        Wz=range(0, nsteps_y, stride_y)
        Hz=range(0, nsteps_x, stride_x)
        return [np.ix_(ixs(h, k_x), ixs(w, k_y)) for h in Hz for w in Wz]
    
    if K.ndim == 2: K = K[np.newaxis]
    
    if tp == "valid":
        xy_steps = (X.shape[-2]-K.shape[-2]+xy_stride[-2], X.shape[-1]-K.shape[-1]+xy_stride[-1])
        assert (xy_steps[0] % xy_stride[0] == 0) or (xy_steps[1] % xy_stride[1] == 0)
        blocks = fm_blocks(K, xy_steps, xy_stride)
        d_out = [xy_steps[0]//xy_stride[0], xy_steps[1]//xy_stride[1]]
        # X[ix, iy] = X[n:Hz+n, m:Wz+m] slice of the matrix X
        return np.array([np.sum(X[ix, iy] * K) for (ix,iy) in blocks]).reshape(d_out)
    if tp == "full":
        X_pad = np.pad(X, [(K.shape[-2]-1, K.shape[-2]-1), (K.shape[-1]-1, K.shape[-1]-1)])
        return conv2d(X_pad, K, xy_stride=xy_stride, tp="valid")

def plot_roc(y_test, y_pred, y_prob):
    fpr, tpr, _ = mcs.roc_curve(y_test, y_pred)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % mcs.roc_auc_score(y_test, y_prob))
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def confmtx(y_true, y_pred):
    """
    |        | prediction |
    +--------+------------+
    | actual |    cm      |
    """
    C = mcs.confusion_matrix(y_true, y_pred)
    return pd.DataFrame(C, 
                        columns=[f"p_{k+1}" for k in range(C.shape[1])],
                        index=[f"t_{k+1}" for k in range(C.shape[0])])

if __name__ == '__main__':
    # k1 = np.random.uniform(size=[1, 6, 6])
    # fm_input = np.random.uniform(size=[28, 28])
    # # print(f"\nChannels: {fm_input.shape[0]},\nHeight: {fm_input.shape[1]},\nWeight: {fm_input.shape[2]}")
    # print(f"\nHeight: {fm_input.shape[0]},\nWeight: {fm_input.shape[1]}")
    # fm_output1 = conv2d(fm_input, k1)
    # print(f"\nChannels: {fm_output1.shape[0]},\nHeight: {fm_output1.shape[1]},\nWeight: {fm_output1.shape[2]}")

    # fm_output2 = convolve2d(fm_input, k1[0], mode="valid")
    # print(f"\nHeight: {fm_output2.shape[0]},\nWeight: {fm_output2.shape[1]}")
    # print(all(fm_output1[0].flatten()==fm_output2.flatten()))
    # k1 = np.random.uniform(size=[100, 1, 6, 6])
    # fm_input = np.random.uniform(size=[1, 6, 6])
    # print(f"\nChannels: {fm_input.shape[0]},\nHeight: {fm_input.shape[1]},\nWeight: {fm_input.shape[2]}")
    # fm_output = conv2d(fm_input, k1, (1, 1), tp="full")
    # print(f"\nChannels: {fm_output.shape[0]},\nHeight: {fm_output.shape[1]},\nWeight: {fm_output.shape[2]}\n")
    k1 = np.array([[[1, 0, 1], [0, 1, 1], [1, 0, 1]]])
    fm_input = np.array([[180, 44, 255, 12, 0], 
                         [110, 3, 253, 12, 0], 
                         [100, 0, 255, 0, 0], 
                         [198, 0, 255, 90, 100], 
                         [0, 141, 255, 10, 67]])
    fm_output2 = correlate2d(fm_input, k1[0], mode="full")
    print(fm_output2)
    fm_output1 = conv2d(fm_input, k1[0], tp = "full")
    print(fm_output1)
    print(f"\nHeight: {fm_input.shape[0]},\nWeight: {fm_input.shape[1]}")
    print(f"\nHeight: {fm_output1.shape[0]},\nWeight: {fm_output1.shape[1]}")
    print(f"\nHeight: {fm_output2.shape[0]},\nWeight: {fm_output2.shape[0]}")
    

    # k1 = np.random.uniform(size=[6, 6])
    # fm_input = np.random.uniform(size=[6, 6])
    # print(f"\nHeight: {fm_input.shape[0]},\nWeight: {fm_input.shape[1]}")
    # fm_output = convolve2d(fm_input, k1, mode="full")
    # print(f"\nHeight: {fm_output.shape[0]},\nWeight: {fm_output.shape[1]}")
    # print(fm_input[0])
    # print(fm_output[0])

    k1 = np.random.uniform(size=[4, 4])
    k1new = np.flipud(np.fliplr(k1))
    k1.shape
    k1new.shape
    k1
    k1new
    # a=np.arange(36).reshape(2, 2, 3, 3)
    # a_rot = rot180(a)
    # print(a_rot.shape)
    # print(a_rot)