import numpy as np
from numpy.lib.npyio import savez_compressed
from model import FCNN
from functions import accuracy, one_hot, split, confusion_matrix
from dbs import DataBase

### Drag Data Set
data, labels = DataBase("data/drag.txt").load_drag()
labels = labels.astype("str").reshape(labels.shape[0])
levels = one_hot(np.unique(labels))
labs= np.array([levels[l] for l in labels for level in levels.keys() if l == level])
X, y, testX, testY = split(data, labs, 0.4)

epochs = 100
learning_rate = 0.001
nrs = [X.shape[1], 40, y.shape[1]]
Net = FCNN(nrs, step=epochs)
print(Net)
Net.train(X, y, epochs=epochs, lr=learning_rate, save_pars_path="pars/drag")

# Parameters can be loaded to use them with out training them again
SameNet = FCNN(nrs, step=epochs, stored_path="pars/drag")
print(SameNet)
predY = SameNet.predict(testX)

testY = np.array([l[0] for l in levels.items() for t in testY if all(t==l[1])])
predY = np.array([l[0] for l in levels.items() for t in predY if t==np.argmax(l[1])])

cm = confusion_matrix(testY, predY)
acc = accuracy(testY, predY)
print(cm)
print(acc)
