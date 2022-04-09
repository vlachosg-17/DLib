import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc
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
parser.add_argument("--epochs",                 type=int,   default=1000)
parser.add_argument("--lr",                     type=float, default=0.001)
parser.add_argument("--batch_size",             type=int,   default=4)
# parser.add_argument("--hidden_layer_neurons",   type=int,   default=[100, 40])
parser.add_argument("--test_prc",               type=int,   default=0.4)
parser.add_argument("--save_dir",         type=str,   default="pars/pima")
parser.add_argument("--latest_checkpoint_path", type=str,   default=None)
hpars = parser.parse_args()

if "__main__" == __name__:
    # Features: ntp, pg, dbp, tst, h2si, bmi, dpf, age, class 
    data, labels = DataBase(hpars.data_path).load_pima("pima_indians_diabetes.txt")
    print(np.unique(labels))
    print(labels.shape)
    labs = one_hot(labels)
    X, y, testX, testY = split(data, labs, hpars.test_prc)

    # Neural Net's Architecture
    layers = [
        MLP([X.shape[1], 500]), 
        ReLu(), # end of 1st hidden layer
        MLP([500, 200]), 
        ReLu(), # end of 2nd hidden layer
        MLP([200, 100]), 
        ReLu(), # end of 3rd hidden layer
        MLP([100, 50]), 
        ReLu(), # end of 4th hidden layer
        MLP([50, y.shape[1]]), 
        Softmax() # end 5th hidden or output layer
        ]
    Net = Model(pipline=layers, loss = "square", stored_path = f"{hpars.main_path}\{hpars.save_dir}", step=hpars.epochs)
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
    print("AUC:", roc_auc_score(y_test, y_prob[:, 1]))


    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot([e for e in range(len(Net.train_errors))], Net.train_errors, label="train error")
    ax1.plot([e for e in range(len(Net.valid_errors))], Net.valid_errors, label="test error")
    ax1.legend(loc="upper right")
    fpr, tpr, threshold = roc_curve(y_test, y_prob[:, 1])
    roc_auc = auc(fpr, tpr)
    ax2.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    ax2.legend(loc = 'lower right')
    ax2.plot([0, 1], [0, 1],'r--')
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax2.set_ylabel('True Positive Rate')
    ax2.set_xlabel('False Positive Rate')
    plt.show()
