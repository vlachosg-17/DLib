import numpy as np
from dbs import DataBase 
import functions as F
from tqdm import tqdm
from layer import Loss
from optimizer import Opt

class Model:
    def __init__(self, loss="entropy", stored_path=None, step=1):
        # f""" Fully Connected Neural Network with {len(neurons) - 1} for multi-class classification problem"""  
        self.id = "mulit-label-classification"
        self.batch = None
        self.Layers = []
        self.neurons = []
        self.step = step
        self.path = stored_path
        self.loss = loss
        
    
    def __repr__(self):
        self.layers = len(self.neurons) - 1
        s = f"Model("
        for i in range(len(self.neurons)):
            if 0<= i <=len(self.neurons)-2:
                s += f"{self.neurons[i]}-->"
            else:
                s += f"{self.neurons[i]}, "
        s+= f"h_layers={self.layers}, problem={self.id}, loss={self.loss})"
        return s
    
    def init_params(self, path):
        if path is None:
            # The weights and biases are already set
            self.train_errors, self.valid_errors = [], []
        else:
            self.db = DataBase(path)
            self.init_w = self.db.load_par("weigths.txt")
            self.init_b = self.db.load_par("bias.txt")
            self.train_errors = self.db.load_errors("train_errors.txt")
            self.valid_errors = self.db.load_errors("valid_errors.txt")
            for l in range(self.layers):
                self.Layers[l].w, self.Layers[l].b = self.init_w[l], self.init_b[l]  
        
    def add(self, node):
        if len(self.Layers) == 0:
            self.neurons.append(node.dims_in)
        self.neurons.append(node.dims_out)
        self.Layers.append(node)
        self.layers = len(self.neurons) - 1
        
    def iter_batch(self, X):
        n_data_segments = X.shape[0]//self.batch
        segs = [[n, n+self.batch] for n in range(0, n_data_segments, self.batch)]
        for seg in segs:
            yield seg

    def forward(self, X):
        for Layer in self.Layers:
            X = Layer.forward(X)
        return X

    def prob(self, X):
        """ Probability in which the model infrence the class """
        return self.forward(X)

    
    def predict(self, X, classes=True):
        # pred = self.forward(X)
        # pred = np.array([np.eye(1, len(p), np.argmax(p)).flatten() for p in pred], dtype=np.int32)
        # return np.array([c1 for c1, c2 in classes.items() for t in pred if all(t==c2)])
        return np.argmax(self.prob(X), axis=1)

    def train(self, X, y, epochs = 1, lr=0.01, batch_size=None, save_pars_path=None):
        self.init_params(self.path)
        self.trainX, self.trainY, self.validX, self.validY = F.split(X, y, 0.2)
        if batch_size is None: self.batch = self.trainX.shape[0]
        else: self.batch = batch_size
        self.save_path = save_pars_path
        self.optimizer = Opt(lr)
        for t in tqdm(range(epochs), ncols=70):
            self.lossT = []
            # Train Set
            self.trainX, self.trainY = F.shuffle(self.trainX, self.trainY)
            # --- batch iteration start here
            for (ls, us) in self.iter_batch(self.trainX):
                self.output = self.forward(self.trainX[ls:us])
                self.lossTrain = Loss(self.trainY[ls:us], self.output, type=self.loss)
                self.lossTrain.backprop(self.Layers)
                self.optimizer.sgd(self.Layers)
                self.lossT.append(np.mean(self.lossTrain.errors))
            # --- batch iteration ends here

            # Validation Set
            self.outV = self.forward(self.validX)
            self.lossValid = Loss(self.validY, self.outV, type=self.loss)
            
            self.valid_errors.append(np.mean(self.lossValid.errors))
            self.train_errors.append(np.mean(self.lossT))
            if t%self.step==0:
                print(" At step:", t, \
                      "train error:", round(self.train_errors[-1], 5), \
                      "validation error:", round(self.valid_errors[-1], 5))     
        
        if self.save_path is not None:
            self.db = DataBase(self.save_path, create=True)
            self.db.save_par([self.Layers[l].w for l in range(len(self.Layers))], "weigths.txt")
            self.db.save_par([self.Layers[l].b for l in range(len(self.Layers))], "bias.txt")
            self.db.save_error(self.train_errors, "train_errors.txt")
            self.db.save_error(self.valid_errors, "valid_errors.txt")
