import numpy as np 
import functions as F
from tqdm import tqdm
from layer import MLP, Loss
from optimizer import Opt

class FCNN:
    def __init__(self, layers, dims, step=1):
        """ Fully Connected Neural Network with 2 hidden layers for multi-class classification problem"""  
        self.id = "mulit-label-classification"
        self.layers = layers
        self.Layers = [None] * self.layers
        self.dims = dims
        self.step = step
        self.Layers[0] = MLP(dims = [dims[0], 30], activation="relu")
        self.Layers[1] = MLP(dims = [self.Layers[0].dims_out, 40], activation="relu")
        self.Layers[2] = MLP(dims = [self.Layers[1].dims_out, 10], activation="relu")
        self.Layers[3] = MLP(dims = [self.Layers[2].dims_out, dims[1]], activation="softmax")

    def __repr__(self) -> str:
        return f"FCNN(from={self.dims[0]}, to={self.dims[1]}, layers={self.layers}, problem={self.id})"

    def forward(self, X):
        for Layer in self.Layers:
            X = Layer.forward(X)
        return X
    
    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)
            
    def train(self, X, y, epochs = 1, lr=0.01):
        self.trainX, self.trainY, self.validX, self.validY = F.split(X, y, 0.2)
        self.optimizer = Opt(lr)
        for t in tqdm(range(epochs)):
            self.trainX, self.trainY = F.shuffle(self.trainX, self.trainY)
            self.output = self.forward(self.trainX)
            self.lossTrain = Loss(self.trainY, self.output)
            self.lossTrain.backprop(self.Layers)
            self.optimizer.sgd(self.Layers)

            self.outV = self.forward(self.validX)
            self.lossValid = Loss(self.validY, self.outV)
            if t%self.step==0:
                print("At step:", t, "train error:", round(np.mean(self.lossTrain.errors), 5), "validation error:", round(np.mean(self.lossValid.errors), 5))
                
