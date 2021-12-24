import numpy as np
from dbs import DataBase 
import functions as F
from tqdm import tqdm
from layer import MLP, Loss
from optimizer import Opt

class FCNN:
    def __init__(self, neurons, stored_path=None, step=1):
        f""" Fully Connected Neural Network with {len(neurons) - 1} for multi-class classification problem"""  
        self.id = "mulit-label-classification"
        self.layers = len(neurons) - 1
        self.batch = None
        self.Layers = [None] * self.layers
        self.neurons = neurons
        self.step = step
        self.path = stored_path
        if self.path is not None:
            self.db_weights = DataBase(self.path+"/weights.txt")
            self.db_bias = DataBase(self.path+"/bias.txt")

        if self.path is None:
            for l in range(self.layers):
                if l != self.layers-1:
                    self.Layers[l] = MLP(dims = [neurons[l], neurons[l+1]], activation="relu")
                else:
                    self.Layers[l] = MLP(dims = [neurons[l], neurons[l+1]], activation="softmax")
        else:
            w = self.db_weights.load_par()
            b = self.db_bias.load_par()
            for l in range(self.layers):
                if l != self.layers-1:
                    self.Layers[l] = MLP(dims = [neurons[l], neurons[l+1]], activation="relu", w=w[l], b=b[l])
                else:
                    self.Layers[l] = MLP(dims = [neurons[l], neurons[l+1]], activation="softmax",  w=w[l], b=b[l])

    def __repr__(self):
        s = f"FCNN("
        for i in range(len(self.neurons)):
            if 0<= i <=len(self.neurons)-2:
                s += f"{self.neurons[i]}-->"
            else:
                s += f"{self.neurons[i]}, "
        s+= f"h_layers={self.layers}, problem={self.id})"
        return s
    
    def iter_batch(self, X):
        n_data_segments = X.shape[0]//self.batch
        segs = [[n, n+self.batch] for n in range(0, n_data_segments, self.batch)]
        for seg in segs:
            yield seg

    def forward(self, X):
        for Layer in self.Layers:
            X = Layer.forward(X)
        return X
    
    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)
            
    def train(self, X, y, epochs = 1, lr=0.01, batch_size=None, save_pars_path=None):
        self.trainX, self.trainY, self.validX, self.validY = F.split(X, y, 0.2)
        if batch_size is None: self.batch = self.trainX.shape[0]
        else: self.batch = batch_size
        self.save_path = save_pars_path
        self.optimizer = Opt(lr)
        for t in tqdm(range(epochs)):
            # Train Set
            self.trainX, self.trainY = F.shuffle(self.trainX, self.trainY)
            # --- batch iteration start here
            for s in self.iter_batch(self.trainX):
                self.output = self.forward(self.trainX[s[0]:s[1]])
                self.lossTrain = Loss(self.trainY[s[0]:s[1]], self.output)
                self.lossTrain.backprop(self.Layers)
                self.optimizer.sgd(self.Layers)
            # --- batch iteration ends here

            # Validation Set
            self.outV = self.forward(self.validX)
            self.lossValid = Loss(self.validY, self.outV)
            if t%self.step==0:
                print("At step:", t, \
                    "train error:", round(np.mean(self.lossTrain.errors), 5), \
                    "validation error:", round(np.mean(self.lossValid.errors), 5))
        
        if self.save_path is not None:
            self.db_weights = DataBase(self.save_path+"/weights.txt")
            self.db_bias = DataBase(self.save_path+"/bias.txt")
            self.db_weights.save_par([self.Layers[l].w for l in range(len(self.Layers))])
            self.db_bias.save_par([self.Layers[l].b for l in range(len(self.Layers))])
                
