import numpy as np
import functions as F
import dbs as DataBase

class MLP:
    def __init__(self, dims,**kwargs):
        if "w" in kwargs.keys():
            self.w = kwargs["w"]
        else:
            self.w = np.random.uniform(-np.sqrt(1/dims[0]),np.sqrt(1/dims[0]),size=[dims[0], dims[1]])
        if "b" in kwargs.keys():
            self.b = kwargs["b"]
        else:
            self.b = np.random.uniform(-np.sqrt(1/dims[0]),np.sqrt(1/dims[0]), size=[1, self.w.shape[1]])

        if "activation" in kwargs.keys():
            if kwargs["activation"] == "softmax":
                self.f = F.softmax
            if kwargs["activation"] == "relu":
                self.f = F.relu
            if kwargs["activation"] == "sigmoid":
                self.f = F.sigmoid
        else:
            self.f = F.relu
        self.id = "layer"
        self.dims_in = dims[0]
        self.dims_out = dims[1]
        self.params = [self.w, self.b]
        self.DwL = None
        self.loss = None
        self.x = None
        self.z = None
        self.y = None

    def __repr__(self):
        return f"Layer(in={self.dims_in}, out={self.dims_out})"
        
    def forward(self, x):
        """
        x: input of current nodes
        """
        # d: number of example in data set
        # n: number of neurons in current layer
        # m: in the next
        self.x = x # d x n
        assert self.x.shape[1] == self.dims_in
        self.z = np.dot(self.x, self.w) + self.b # (d x n) @ (n x m) .+ (1 x m) = d x m
        self.y = self.f(self.z) # f(d x m) = d x m point wise function
        return self.y
    
    def backprop(self, node):
        """
        The backpropagation method that calculates the derivatives of
        the loss function with respect to weights and bias.
        node: node ahead of the current node. The last node is presume to the 
        be the loss function, while to find the DwL of the last layer we 'backpropagate'
        from the loss function.
        """
        # n3: number of neurons in the layer ahead
        # n2: number of neurons in the current layer
        # n1: number of neurons in the layer before
        # d: number of examples in data set
        if node.id is "loss":
            if node.type == "entropy": self.DzL = node.y_hat - node.y # (d x n) - (d x n) = d x n
            else: self.DzL = 2*(node.y_hat - node.y) * self.f(self.z) # ((d x n) - (d x n)) * d x n = d x n
        else:
            self.DzL = self.f(self.z, der = True) * np.dot(node.DzL, node.w.T) # (d x n2) * (d x n3) @ (n3 x n2) = (d x n2) * (d x n2) = d x n2
        self.DwL = np.array([np.dot(self.x[d,:][np.newaxis].T, self.DzL[d,:][np.newaxis]) for d in range(self.DzL.shape[0])]) # d x [(n1 x 1) @ (1 x n2)] = d x n1 x n2 for sgd
        self.DbL = self.DzL # d x n2
        # self.gradients = [self.DwL, self.DbL]        

class Loss:
    def __init__(self, y, y_hat, type="entropy"):
        # d: number of examples
        # n: number of classes = number of neurons in last layer
        self.id = "loss"
        self.y = y # d x n
        self.y_hat = y_hat # d x n
        self.type = type
        if self.type == "square":
            self.errors = np.sum((self.y_hat-self.y) * (self.y_hat-self.y))
             
        if self.type == "entropy":
            self.errors = -np.sum(y * np.log(y_hat), axis=1) # sum((d x n) * (d x n)) = d x 1 

    def backprop(self, Layers):
        """ Backpropagate one by one from (l+1)-layer --> l-layer """
        for l in range(len(Layers)-1, -1, -1):
            if l == len(Layers)-1:  
                Layers[l].backprop(self)
            else:
                Layers[l].backprop(Layers[l+1])
