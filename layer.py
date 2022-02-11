import numpy as np
import functions as F

class MLP:
    def __init__(self, dims, **kwargs):
        self.id = "MLP"
        self.dims_in = dims[0]
        self.dims_out = dims[1]
        self.neurons = self.dims_out
        self.DwL = None
        self.loss = None
        self.x = None
        self.z = None
        self.y = None

        if "w" in kwargs.keys():
            self.w = kwargs["w"]
        else:
            self.w = np.random.uniform(-np.sqrt(1/self.dims_in), np.sqrt(1/self.dims_in), size=[self.dims_in, self.dims_out])
        if "b" in kwargs.keys():
            self.b = kwargs["b"]
        else:
            self.b = np.random.uniform(-np.sqrt(1/self.dims_in), np.sqrt(1/self.dims_in), size=[1, self.w.shape[1]])
        if "activation" in kwargs.keys():
            if kwargs["activation"] == "softmax":
                self.f = F.softmax
            if kwargs["activation"] == "relu":
                self.f = F.relu
            if kwargs["activation"] == "sigmoid":
                self.f = F.sigmoid
        else:
            self.f = F.relu
        
        

    def __repr__(self):
        return f"Layer(type={self.id},in={self.dims_in}, out={self.dims_out})"
        
    def forward(self, x):
        """
        x: input of current nodes
        In the comments:
            - d: number of examples in data set
            - n: number of neurons in current layer
            - m: number of neurons in the next layer
        """
        self.x = x # d x n
        assert self.x.shape[1] == self.dims_in
        self.z = np.dot(self.x, self.w) + self.b # (d x n) @ (n x m) .+ (1 x m) = d x m
        self.y = self.f(self.z) # f(d x m) = d x m point wise function
        return self.y
    
    def backprop(self, node):
        """
         ------------------------     Forward    -------------------------- 
        |                        |  ----------> |                          |  
        | layer-l (current node) |              | layer-(l+1) (ahead node) | 
        |                        |  <---------- |                          | 
         ------------------------     Backward   --------------------------     
        node: is the layer-(l+1) which is ahead from the current node (layer-l). 
              The last node is presume to be the loss function, 
              while to find the DwL from the last layer 
              we 'backpropagate' from the loss function node.

        self.DzL: else known as delta, is apparent that is calculated with different ways that are depended
                  in the type of the ahead layer (node.id). When the code is extended this layer
                  will calulated DzL in multiple ways according to object node.id. For now the only layer
                  that lies ahead is an "loss" layer or an "MLP" layer as this one.

        In the comments:
            - n3: number of neurons in the layer ahead
            - n2: number of neurons in the current layer
            - n1: number of neurons in the layer before
            - d: number of examples in data set
            - n: number of classes in the dataset or number of neurons in the last layer
        """
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
        """
        In the commnets:
            - d: number of examples
            - n: number of classes = number of neurons in last layer
        """
        self.id = "loss"
        self.y = y # d x n
        self.y_hat = y_hat # d x n
        self.type = type
        if self.type == "square":
            self.errors = np.sum((self.y_hat-self.y) * (self.y_hat-self.y), axis=1)

        if self.type == "entropy":
            self.errors = -np.sum(y * np.log(y_hat), axis=1) # sum((d x n) * (d x n)) = d x 1 

    def backprop(self, Layers):
        """ Backpropagate one by one from (l+1)-layer --> l-layer """
        for l in range(len(Layers)-1, -1, -1):
            if l == len(Layers)-1:  
                Layers[l].backprop(self)
            else:
                Layers[l].backprop(Layers[l+1])
