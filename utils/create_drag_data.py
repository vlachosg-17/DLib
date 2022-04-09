import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from .dbs import DataBase

def drag_data():
    train_labels = []
    train_samples = []

    for i in range(50):
        random_younger = randint(13, 64)
        train_samples.append(random_younger)
        train_labels.append(1)

        random_older = randint(64, 100)
        train_samples.append(random_older)
        train_labels.append(0)

    for i in range(1000):
        random_younger = randint(13, 64)
        train_samples.append(random_younger)
        train_labels.append(0)

        random_older = randint(64, 100)
        train_samples.append(random_older)
        train_labels.append(1)

    np.random.seed(0)
    train_samples = np.array(train_samples)
    train_labels = np.array(train_labels)
    train_samples, train_labels = shuffle(train_samples, train_labels)
    scaler = MinMaxScaler(feature_range = (0,1))
    scaler_train_samples = scaler.fit_transform(train_samples.reshape(-1,1))
    for i, x in enumerate(scaler_train_samples):
        if x==0:
            scaler_train_samples[i]=0.01
    scaler_train_samples=scaler_train_samples.astype("float32") 
    train_labels= train_labels.astype("int8")

    base=DataBase("data/drag.txt")
    base.save_drag(data=scaler_train_samples, labels=train_labels)

if __name__ == "__main__":
    drag_data()
    # print(scaler_train_samples)
    # print(scaler_train_samples.shape)
    # print(train_samples.shape, train_labels.shape)

