import numpy as np
from model import FCNN
from functions import *
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from dbs import DataBase
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--train",               type=bool, default=False)
parser.add_argument("--epochs",               type=int, default=500)
parser.add_argument("--lr",                   type=int, default=0.001)
parser.add_argument("--batch_size",           type=int, default=4)
parser.add_argument("--hidden_layer_neurons", type=int, default=[30])
parser.add_argument("--test_prc",           type=int, default=0.4)
parser.add_argument("--pars_save_path",      type=str, default="pars/drag")
hpars = parser.parse_args()

if "__main__" == __name__:
    ### Drag Data Set
    data, labels = DataBase("data").load_drag("drag.txt")
    labs = one_hot(labels)
    X, y, testX, testY = split(data, labs, hpars.test_prc)
    nrs = [X.shape[1]] + hpars.hidden_layer_neurons + [y.shape[1]]
    if hpars.train:
        Net = FCNN(nrs, step=hpars.epochs)
        Net.train(X, y, epochs=hpars.epochs, lr=hpars.lr, batch_size=hpars.batch_size, save_pars_path=hpars.pars_save_path)
    else:
        # Parameters can be loaded to use them with out training them again
        Net = FCNN(nrs, step=hpars.epochs, stored_path=hpars.pars_save_path)
    print(Net)   

    levels = eye_levels(np.unique(labels))
    y_prob = Net.prob(testX)
    y_pred = Net.predict(testX, classes=levels)
    y_test = np.array([l[0] for l in levels.items() for t in testY if all(t==l[1])])
    
    cm = confmtx(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print(y_test)
    print(np.round(y_prob[:, 1], 3))
    print("Accuracy:", np.diag(cm.to_numpy()).sum()/cm.to_numpy().sum())
    print("AUC:", roc_auc_score(y_test, np.round(y_prob[:, 1], 3)))


    plt.plot([e for e in range(len(Net.train_errors))], Net.train_errors)
    plt.plot([e for e in range(len(Net.valid_errors))], Net.valid_errors)
    plt.show()