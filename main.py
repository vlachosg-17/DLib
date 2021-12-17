import numpy as np
from model import FCNN
from functions import accuracy, one_hot, split, confusion_matrix
from dbs import DataBase

data, labels = DataBase("data/iris.data").load_iris()
levels = one_hot(np.unique(labels))
labs= np.array([levels[l] for l in labels for level in levels.keys() if l == level])
X, y, testX, testY = split(data, labs, 0.4)

epochs = 1000
learning_rate = 0.001
Net = FCNN(4, [X.shape[1], y.shape[1]], step=epochs)
Net.train(X, y, epochs=epochs, lr=learning_rate)

predY = Net.predict(testX)

testY = np.array([l[0] for l in levels.items() for t in testY if all(t==l[1])])
predY = np.array([l[0] for l in levels.items() for t in predY if t==np.argmax(l[1])])

cm = confusion_matrix(testY, predY)
acc = accuracy(testY, predY)
print(cm)
print(acc)