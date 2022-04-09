
from . import np, tqdm
import os
from utils.functions import shuffle, split
from utils.dbs import DataBase 

from .layers import Loss
from .optimizer import Opt

class Model:
    def __init__(self, pipline=[], loss="entropy", stored_path=None, step=1):
        self.id = "mulit-label-classification"
        self.batch = None
        self.Layers = pipline
        self.n_layers = len(self.Layers)
        # self.neurons = []
        self.step = step
        self.path = stored_path
        self.loss_type = loss
        if stored_path is not None:
            self.init_params(load=os.path.exists(stored_path)) # For Now after I fix the loading process it must be always = True
        else:
            self.init_params(load=False)
    
    def __repr__(self):
        pass
        # Need Customization
        # self.n_layers = len(self.neurons) - 1
        # s = f"Model("
        # for i in range(len(self.neurons)):
        #     if 0<= i <=len(self.neurons)-2:
        #         s += f"{self.neurons[i]}-->"
        #     else:
        #         s += f"{self.neurons[i]}, "
        # s+= f"h_layers={self.n_layers}, problem={self.id}, loss={self.loss})"
        # return s
    
    def __call__(self, X):
        return self.forward(X)
    
    def init_params(self, load):
        if load == False:
            self.train_errors, self.valid_errors = [], []
            for l, layer in enumerate(self.Layers): 
                if layer.id == "MLP":
                    d_in, d_out = np.prod(layer.d_in), np.prod(layer.d_out)
                    init_w = np.random.uniform(-np.sqrt(1/d_in), np.sqrt(1/d_in), size=[d_in, d_out])
                    init_b = np.random.uniform(-np.sqrt(1/d_in), np.sqrt(1/d_in), size=[1, d_out])
                    self.Layers[l].weights, self.Layers[l].bias = init_w, init_b
                if layer.id == "CNL2D":
                    d_in, d_out = np.prod(layer.d_in), np.prod(layer.d_out)
                    init_w = np.random.uniform(-np.sqrt(1/d_in), np.sqrt(1/d_in), size=[d_out[0], d_in[0], layer.Hk, layer.Wk])
                    init_b = np.random.uniform(-np.sqrt(1/d_in), np.sqrt(1/d_in), size=d_out)    
                    self.Layers[l].weights, self.Layers[l].bias = init_w, init_b
        if load == True:    
            self.db = DataBase(self.path)
            self.train_errors = self.db.load_errors("train_errors.txt")
            self.valid_errors = self.db.load_errors("valid_errors.txt")
            for l, layer in enumerate(self.Layers):
                if self.Layers[l].id == "MLP":
                    if not (os.path.exists(f"{self.path}\weights {l}-layer.txt") or os.path.exists(f"{self.path}\\bais {l}-layer.txt")):
                        load=False
                        break
                    self.init_w = self.db.load_par(f"weights {l}-layer.txt")
                    self.init_b = self.db.load_par(f"bias {l}-layer.txt")
                    self.Layers[l].weights, self.Layers[l].bias = self.init_w, self.init_b
                if self.Layers[l].id == "CNL2D":
                    if not (os.path.exists(f"{self.path}\weights {l}-layer.txt") or os.path.exists(f"{self.path}\\bais {l}-layer.txt")):
                        load=False
                        break
                    self.init_w = self.db.load_par(f"weights {l}-layer.txt")
                    self.init_b = self.db.load_par(f"bias {l}-layer.txt")
                    self.Layers[l].weights = self.init_w.reshape(layer.C1, layer.C0, layer.Hk, layer.Wk), 
                    self.Layers[l].bias = self.init_b.reshape(layer.C1, layer.Hz, layer.Wz)
            if load == False:
                self.init_params(load=False)
              
    def add(self, node):
        self.Layers.append(node)
        self.n_layers = len(self.Layers)

    def rm(self):
        self.Layers.pop()
        self.n_layers = len(self.Layers)
        
    def iter_batch(self, X, batch):
        n_data_segments = X.shape[0]//batch
        segs = [[n, n+batch] for n in range(0, n_data_segments, batch)]
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

    def train(self, X, y, epochs = 1, lr=0.01, batch_size=None, save_path=None):
        self.save_path = save_path
        self.trainX, self.trainY, self.validX, self.validY = split(X, y, 0.2)
        if batch_size is None: self.batch = self.trainX.shape[0]
        else: self.batch = batch_size
        self.optimizer = Opt(lr)
        self.loss = Loss(type=self.loss_type)
        for t in tqdm(range(epochs), ncols=70):
            self.lossTrain = []
            # Train Set
            self.trainX, self.trainY = shuffle(self.trainX, self.trainY)
            # --- batch iteration start here
            for (low_b, upper_b) in self.iter_batch(self.trainX, self.batch):
                output = self.forward(self.trainX[low_b:upper_b])
                errors = self.loss(self.trainY[low_b:upper_b], output)
                self.loss.backward(self.Layers)
                self.optimizer.sgd(self.Layers)
                self.lossTrain.append(np.mean(errors))
            # --- batch iteration ends here

            # Validation Set
            output = self.forward(self.validX)
            self.lossValid = self.loss(self.validY, output, no_grad=True)
            
            self.valid_errors.append(np.mean(self.lossValid))
            self.train_errors.append(np.mean(self.lossTrain))
            if t%self.step==0:
                print(" At step:", t, \
                      "train error:", round(self.train_errors[-1], 5), \
                      "validation error:", round(self.valid_errors[-1], 5))     
        
        if self.save_path is not None:
            self.db = DataBase(self.save_path, create=True)
            for l, layer in enumerate(self.Layers):
                if self.Layers[l].id in "MLP":
                    self.db.save_par(layer.weights, f"weights {l}-layer.txt")
                    self.db.save_par(layer.bias, f"bias {l}-layer.txt")
                if self.Layers[l].id in "CNL2D":
                    w = self.Layers[l].weights
                    self.db.save_par(layer.weights.reshape(layer.C1, layer.C0*layer.Hk*layer.Wk), f"weights {l}-layer.txt")
                    self.db.save_par(layer.bias.reshape(layer.C1, layer.Hz*layer.Wz), f"bias {l}-layer.txt")
            self.db.save_error(self.train_errors, "train_errors.txt")
            self.db.save_error(self.valid_errors, "valid_errors.txt")
