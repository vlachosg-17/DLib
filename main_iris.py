import numpy as np
from model import FCNN
from functions import accuracy, one_hot, split, confusion_matrix
from dbs import DataBase

### Iris Data Set
data, labels = DataBase("data/iris.data").load_iris()
levels = one_hot(np.unique(labels))
labs= np.array([levels[l] for l in labels for level in levels.keys() if l == level])
X, y, testX, testY = split(data, labs, 0.4)

# Train the Neural Net
epochs = 1000
learning_rate = 0.001
nrs = [X.shape[1], 40, 30, 10, y.shape[1]]
Net = FCNN(nrs, step=epochs)
print(Net)
Net.train(X, y, epochs=epochs, lr=learning_rate, save_pars_path="pars/iris")

# Parameters can be loaded to use them with out training them again
SameNet = FCNN(nrs, step=epochs, stored_path="pars/iris")
print(SameNet)
predY = SameNet.predict(testX)

testY = np.array([l[0] for l in levels.items() for t in testY if all(t==l[1])])
predY = np.array([l[0] for l in levels.items() for t in predY if t==np.argmax(l[1])])

cm = confusion_matrix(testY, predY)
acc = accuracy(testY, predY)
print(cm)
print(acc)
