import numpy as np
from model import Model
from layer import MLP
from functions import *
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from dbs import DataBase
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--main_path",              type=str,  default="H:\My Drive\ML\My-Object-Orienated-Neural-Network")
parser.add_argument("--data_path",              type=str,  default="P:")
parser.add_argument("--train",                  type=bool, default=True)
parser.add_argument("--epochs",                 type=int,  default=500)
parser.add_argument("--lr",                     type=int,  default=0.001)
parser.add_argument("--batch_size",             type=int,  default=4)
parser.add_argument("--hidden_layer_neurons",   type=int,  default=[30])
parser.add_argument("--test_prc",               type=int,  default=0.4)
parser.add_argument("--pars_save_path",         type=str,  default="pars/drag")
parser.add_argument("--latest_checkpoint_path", type=str,  default=None)
hpars = parser.parse_args()

if "__main__" == __name__:
    ### Drag Data Set
    data, labels = DataBase("P:\data").load_drag("drag.txt")
    labs = one_hot(labels)
    X, y, testX, testY = split(data, labs, hpars.test_prc)

    # Neural Net's Architecture
    Net = Model(stored_path=hpars.latest_checkpoint_path, step=100)
    nrs = [X.shape[1]] + hpars.hidden_layer_neurons + [y.shape[1]]
    for l in range(len(nrs)-1):
        if l != len(nrs)-2:
            Net.add(MLP(dims = [nrs[l], nrs[l+1]], activation="relu"))
        else:
            Net.add(MLP(dims = [nrs[l], nrs[l+1]], activation="softmax"))
    print(Net)

    # Train the Neural Net
    if hpars.train:
        Net.train(X, y, epochs=hpars.epochs, lr=hpars.lr, batch_size=hpars.batch_size, save_pars_path=hpars.pars_save_path)
    
    y_prob = Net.prob(testX)
    y_pred = Net.predict(testX)
    y_test = reverse_one_hot(testY, classes = np.unique(labels))
    
    cm = confmtx(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    # print(y_test)
    # print(np.round(y_prob[:, 1], 3))
    print("Accuracy:", np.diag(cm.to_numpy()).sum()/cm.to_numpy().sum())
    print("AUC:", roc_auc_score(y_test, np.round(y_prob[:, 1], 3)))


    plt.plot([e for e in range(len(Net.train_errors))], Net.train_errors, label="train error")
    plt.plot([e for e in range(len(Net.valid_errors))], Net.valid_errors, label="test error")
    plt.legend(loc="upper right")
    plt.show()