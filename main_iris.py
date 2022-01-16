import numpy as np
from model import FCNN
from functions import *
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from dbs import DataBase
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--train",               type=bool, default=False)
parser.add_argument("--epochs",               type=int, default=1500)
parser.add_argument("--lr",                   type=float, default=0.0003)
parser.add_argument("--batch_size",           type=int, default=4)
parser.add_argument("--hidden_layer_neurons", type=int, default=[40, 30, 10])
parser.add_argument("--test_prc",           type=int, default=0.4)
parser.add_argument("--pars_save_path",      type=str, default="pars/iris")
hpars = parser.parse_args()

if "__main__" == __name__:
    ### Iris Data Set
    data, labels = DataBase("data").load_iris("iris.data", random_seed=10000)
    print(np.unique(labels))
    print(labels.shape)
    labs = one_hot(labels)
    X, y, testX, testY = split(data, labs, hpars.test_prc)

    # Train the Neural Net
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
    print("Accuracy:", np.diag(cm.to_numpy()).sum()/cm.to_numpy().sum())
    print("AUC:", roc_auc_score(to_nominal(y_test), y_prob, multi_class="ovo"))


    plt.plot([e for e in range(len(Net.train_errors))], Net.train_errors)
    plt.plot([e for e in range(len(Net.valid_errors))], Net.valid_errors)
    plt.show()