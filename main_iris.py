import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import argparse

from mynn.model import Model
from mynn.layers import MLP, ReLu, Softmax
from utils.functions import *
from utils.dbs import DataBase

parser = argparse.ArgumentParser()
parser.add_argument("--main_path",              type=str,   default="H:\My Drive\ML\DLib")
parser.add_argument("--data_path",              type=str,   default="P:\data")
parser.add_argument("--train",                  type=bool,  default=False)
parser.add_argument("--epochs",                 type=int,   default=200)
parser.add_argument("--lr",                     type=float, default=0.003)
parser.add_argument("--batch_size",             type=int,   default=4)
# parser.add_argument("--hidden_layer_neurons",   type=int,   default=[100])
parser.add_argument("--test_prc",               type=int,   default=0.4)
parser.add_argument("--save_dir",         type=str,   default="pars\iris")
parser.add_argument("--latest_checkpoint_path", type=str,   default=None)
hpars = parser.parse_args()

if "__main__" == __name__:
    ### Iris Data Set
    data, labels = DataBase(hpars.data_path).load_iris("iris.data", lab_nom=True, random_seed=10000)
    labs = one_hot(labels)
    X, y, testX, testY = split(data, labs, hpars.test_prc)

    # Neural Net's Architecture
    layers = [
        MLP([X.shape[1], 100]), 
        ReLu(), # end of 1st hidden layer
        MLP([100, y.shape[1]]), 
        Softmax() # end 2nd hidden or output layer
        ]
    Net = Model(pipline=layers, loss="entropy", stored_path=f"{hpars.main_path}\{hpars.save_dir}")
    if hpars.train:
        Net.train(
            X=X, 
            y=y, 
            epochs=hpars.epochs, 
            lr=hpars.lr,
            batch_size=hpars.batch_size, 
            save_path=f"{hpars.main_path}\{hpars.save_dir}"
        )
    y_prob = Net.prob(testX)
    y_pred = Net.predict(testX)
    y_test = reverse_one_hot(testY, classes=np.unique(labels))

    
    cm = confmtx(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    # print(y_test)
    # print(np.round(y_prob[:, 1], 3))
    print("Accuracy:", np.diag(cm.to_numpy()).sum()/cm.to_numpy().sum())
    print("AUC:", roc_auc_score(y_test, y_prob, multi_class="ovr"))
    print("AUC:", roc_auc_score(y_test, y_prob, multi_class="ovo"))
    

    plt.plot([e for e in range(len(Net.train_errors))], Net.train_errors, label="train error")
    plt.plot([e for e in range(len(Net.valid_errors))], Net.valid_errors, label="test error")
    plt.legend(loc="upper right")
    plt.show()