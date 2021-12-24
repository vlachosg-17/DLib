import numpy as np
from numpy.core.defchararray import array
from model import FCNN
from functions import accuracy, one_hot, split, confusion_matrix
from dbs import DataBase
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--epochs",               type=int, default=1000)
parser.add_argument("--lr",                   type=float, default=0.001)
parser.add_argument("--batch_size",           type=int, default=4)
parser.add_argument("--hidden_layer_neurons", type=int, default=[40, 30, 10])
parser.add_argument("--test_perc",           type=int, default=0.4)
parser.add_argument("--pars_save_path",      type=str, default="pars/iris")
hpars = parser.parse_args()

if "__main__" == __name__:
    ### Iris Data Set
    data, labels = DataBase("data/iris.data").load_iris(random_seed=10000)
    levels = one_hot(np.unique(labels))
    labs= np.array([levels[l] for l in labels for level in levels.keys() if l == level])
    X, y, testX, testY = split(data, labs, hpars.test_perc)

    # Train the Neural Net
    nrs = [X.shape[1]] + hpars.hidden_layer_neurons + [y.shape[1]]
    Net = FCNN(nrs, step=hpars.epochs)
    print(Net)
    Net.train(X, y, epochs=hpars.epochs, lr=hpars.lr, batch_size=hpars.batch_size, save_pars_path=hpars.pars_save_path)

    # Parameters can be loaded to use them with out training them again
    SameNet = FCNN(nrs, step=hpars.epochs, stored_path=hpars.pars_save_path)
    print(SameNet)
    predY = SameNet.predict(testX)

    testY = np.array([l[0] for l in levels.items() for t in testY if all(t==l[1])])
    predY = np.array([l[0] for l in levels.items() for t in predY if t==np.argmax(l[1])])

    cm = confusion_matrix(testY, predY)
    acc = accuracy(testY, predY)
    print("Confusion Matrix:")
    print(cm)
    print("Accuracy:", acc)
