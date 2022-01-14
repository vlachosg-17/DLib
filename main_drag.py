import numpy as np
from matplotlib import pyplot as plt 
from model import FCNN
from functions import accuracy, one_hot, one_hot_c, split, confusion_matrix
from dbs import DataBase
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--train",               type=bool, default=True)
parser.add_argument("--epochs",               type=int, default=100)
parser.add_argument("--lr",                   type=int, default=0.001)
parser.add_argument("--batch_size",           type=int, default=4)
parser.add_argument("--hidden_layer_neurons", type=int, default=[30])
parser.add_argument("--test_perc",           type=int, default=0.4)
parser.add_argument("--pars_save_path",      type=str, default="pars/drag")
hpars = parser.parse_args()

if "__main__" == __name__:

    ### Drag Data Set
    data, labels = DataBase("data").load_drag("drag.txt")
    print(np.unique(labels))
    labs = one_hot(labels)
    X, y, testX, testY = split(data, labs, hpars.test_perc)

    epochs = 100
    learning_rate = 0.005
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
    acc = accuracy(y_test, y_prob)
    print("Confusion Matrix:")
    print(cm)
    print("Accuracy:", acc)


    plt.plot([e for e in range(hpars.epochs)], Net.train_errors)
    plt.plot([e for e in range(hpars.epochs)], Net.valid_errors)
    plt.show()
