import numpy as np
from model import FCNN
from functions import *
from sklearn.metrics import roc_auc_score, roc_curve
from dbs import DataBase
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--train",               type=bool, default=False)
parser.add_argument("--epochs",               type=int, default=1600)
parser.add_argument("--lr",                   type=float, default=0.002)
parser.add_argument("--batch_size",           type=int, default=4)
parser.add_argument("--hidden_layer_neurons", type=int, default=[40, 30, 10])
parser.add_argument("--test_prc",             type=int, default=0.4)
parser.add_argument("--pars_save_path",       type=str, default="pars/pima")
hpars = parser.parse_args()

if "__main__" == __name__:
    # Features: ntp, pg, dbp, tst, h2si, bmi, dpf, age, class 
    data, labels = DataBase("data").load_pima("pima_indians_diabetes.txt")
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

    levels = one_hot_c(np.unique(labels))
    y_prob = Net.prob(testX)
    y_pred = Net.predict(testX, classes=levels)
    y_test = np.array([l[0] for l in levels.items() for t in testY if all(t==l[1])])
    

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print("Accuracy:", acc)
    print("AUC:", roc_auc_score(y_test, y_pred))

    plt.plot([e for e in range(hpars.epochs)], Net.train_errors)
    plt.plot([e for e in range(hpars.epochs)], Net.valid_errors)
    plt.show()
    
    # plot_roc(y_test, y_pred, y_prob)
    
