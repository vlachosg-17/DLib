from turtle import forward
from . import np
from utils.functions import sigmoid, relu, softmax, conv2d, rot180

class MLP:
    def __init__(self, dims, **kwargs):
        """
        dims = [N1, N2]
        N2: number of neurons in the layer ahead
        N1: number of neurons in the current layer
        D: number of examples in data set
        dims(input) = D x N1
        dims(weights) = N1 x N2
        dims(bias) = 1 x N2
        dims(output) = D x N2
        """
        self.id = "MLP"
        self.d_in = dims[0]
        self.d_out = dims[1]
        self.neurons = self.d_out
        self.num_params = self.d_in * self.d_out + self.d_out # inputs x outputs + biases
        self.DwE = None
        self.loss = None
        self.x = None
        self.z = None
        self.y = None
        self.weights, self.bias = None, None
        self.params = [self.weights, self.bias]
        if "w" in kwargs.keys():
            self.weights = kwargs["w"]
        if "b" in kwargs.keys():
            self.bias = kwargs["b"]
    def __repr__(self):
        return f"Layer(type={self.id},in={self.d_in}, out={self.d_out})"
    
    def __call__(self, X):
        return self.forward(X)
        
    def forward(self, X):
        # D x N1
        self.x = X
        # (D x N2) @ (N1 x N2) + (1 x N2) = D x N2
        self.z = np.dot(self.x, self.weights) + self.bias
        # f(D x N2) = D x N2 
        self.y = self.z 
        return self.y
    
    def backward(self, node):
        # D x N2 * D x N2 = D x N2 
        # self.DzE = self.f(self.z, der = True) * node.DxE 
        self.DzE = node.DxE
        # D x [N1 x 1 @ 1 x N2] = D x N1 x N2
        self.DwE = np.array([np.dot(self.x[d,np.newaxis].T, self.DzE[d,np.newaxis]) for d in range(self.DzE.shape[0])]) 
        # D x N2 = D x N2
        self.DbE = self.DzE 
        # D x N2 @ N2 x N1 = D x N1
        self.DxE = np.dot(self.DzE, self.weights.T) 
        return self.DxE        

class CNL2D:
    def __init__(self, dims, padding=(0,0), stride=(1,1), **kwargs):
        """
        dims = [(C0, H, W), (C1, Hz, Wz)]
        C1: # number of kernels or channels of output feature map
        C0: channels of starting feature Map
        H: height feature map, Hk: height kernel
        W: width feature map, Wk: height kernel
        Hp x Wp = H+2p x W+2p
        Hz x Wz = (Hp-Hk)/s + 1 x (Wp-Wk)/s +1
        dims(input) = D x C0 x Hp x Wp
        dims(kernel) = C1 x C0 x Hk x Wk
        dims(bias) = C1 x Hz x Wz
        dims(output) = D x C1 x Hz x Wz
        """
        self.id = "CNL2D"
        self.pad_x = padding[0]
        self.pad_y = padding[1]
        self.xy_pad = padding
        self.stride_x = stride[0]
        self.stride_y = stride[1]
        self.xy_stride = stride
        self.d_in = dims[0]
        self.d_out = dims[1]
        self.C0, self.H, self.W = self.d_in
        self.C1, self.Hz, self.Wz = self.d_out
        self.Hk = self.H + 2*self.pad_x - self.stride_x*(self.Hz-1)
        self.Wk = self.W + 2*self.pad_y - self.stride_y*(self.Wz-1)
        assert (self.H+2*self.pad_x-self.Hk)%self.stride_x==0 \
            and (self.W+2*self.pad_y-self.Wk)%self.stride_y==0
        # self.C1, self.C0, self.Hk, self.Wk = self.weights.shape
        # self.Hz = (self.H+2*self.pad_x-self.Hk)//self.stride_x+1
        # self.Wz = (self.W+2*self.pad_y-self.Wk)//self.stride_y+1
        self.weights, self.bias = None, None
        self.params = [self.weights, self.bias]
        if "kernels" in kwargs.keys():
            self.weights = kwargs["kernels"]
        if "bias" in kwargs.keys():
            self.bias = kwargs["bias"]
    def __call__(self, X):
        return self.forward(X)

    def pad(self, X):
        print(X.shape)
        return np.pad(X, [(0, 0), (self.pad_x, self.pad_x), (self.pad_y,self.pad_y)])
    
    def conv2d_channels_(self, X, K, mode="valid"):
        """ full/norm 2D convolution of a feature map with channels """
        if mode=="valid":
            Z = np.zeros(self.d_out)
            for c in range(Z.shape[0]):
                Z[c] = sum([conv2d(xc, kc, xy_stride=self.xy_stride, tp=mode) for (xc, kc) in zip(X, K[c])])
        if mode=="full":
            Z = np.zeros(self.d_in)
            for c in range(Z.shape[0]):
                Z[c] = sum([conv2d(xc, kc, xy_stride=self.xy_stride, tp=mode) for (xc, kc) in zip(X, K[:, c])])
                # Not complete must find a way to make K[c] and K[:, c] be the same to remove the repetition
        return Z
    
    def conv2d_cartesian_(self, A, B):
        """ All possible 2D convlolution beteewn A, B """
        C1, C0 = A.shape[0], B.shape[0]
        return np.array([[conv2d(A[c0], B[c1]) for c1 in range(C1)] for c0 in range(C0)])

    def forward(self, X):
        # Dx C0 x H x W
        self.x = X 
        # D x C0 x Hp x Wp
        self.x_pad = np.array([self.pad(self.x[d]) for d in range(self.x.shape[0])]) 
        #  D x C0 x Hp x Wp <x> C1 x C0 x Hk x Wk = D x C1 x (Hp-Hk)/s x (Wp-Wk)/s = D x C1 x Hz x Wz
        self.z = np.array([self.conv2d_channels_(self.x_pad[d], self.weights) + self.bias for d in range(self.x.shape[0])])
        # D x C1 x Hz x Wz = f(D x C1 x Hz x Wz)
        self.y = self.z
        return self.y

    def backward(self, node):
        # # D x C1 x Hz x Wz = f'(D x C1 x Hz x Wz) * D x C1 x Hz x Wz
        # self.DzE = self.f(self.z, der=True) * node.DxE
        self.DzE = node.DxE
        # D x C1 x C0 x Hk x Wk = D x C0 x Hp x Wp <x> D x C1 x Hz x Wz
        self.DwE = np.array([self.conv2d_cartesian_(self.x_pad[d], self.DzE[d]) for d in range(self.x.shape[0])])
        # D x C1 x Hz x Wz = D x C1 x Hz x Wz
        self.DbE = self.DzE
        # D x C0 x Hp x Wp = D x C1 x Hz x Wz (x) C1 x C0 x Hk x Wk (fully convolve in C1)
        self.DxE = np.array([self.conv2d_channels_(self.DzE[d], rot180(self.weights), tp="full") for d in range(self.x.shape[0])])
        return self.DxE

class Activation:
    def __init__(self, f):
        """ Sigmoid, Relu, Tanh """
        self.id=f"{f.__name__}"
        self.f = f   
    
    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.X = X
        return self.f(X)

    def backward(self, node):
        self.DxE = self.f(self.X, der=True) * node.DxE
        return self.DxE

class ReLu:
    def __init__(self):
        self.id="relu"
        
    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.X = X
        return relu(X)

    def backward(self, node):
        self.DxE = relu(self.X, der=True) * node.DxE
        return self.DxE

class Sigmoid:
    def __init__(self):
        self.id="sigmoid"
        
    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.X = X
        return sigmoid(X)

    def backward(self, node):
        self.DxE = sigmoid(self.X, der=True) * node.DxE
        return self.DxE

class Softmax:
    def __init__(self):
        self.id="softmax"

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.y = softmax(X)
        return self.y

    def backward(self, node):
        self.DxE = np.zeros(self.y.shape)
        for d in range(self.y.shape[0]):
            Y = np.tile(self.y[d], [self.y.shape[1],1])
            I = np.identity(self.y.shape[1])
            self.DxE[d] = np.dot(node.DxE[d], Y* (I - Y.T))
        return self.DxE

class Flatten:
    def __init__(self):
        self.id="flatten"
    
    def __call__(self, X):
        return self.forward(X)
    
    def forward(self, X):
        self.dims = X.shape
        return X.reshape(self.dims[0], np.prod(self.dims[1:]))
    
    def backward(self, node): 
        self.DxE = node.DxE.reshape(self.dims)
        return self.DxE

class Loss:
    def __init__(self, type="entropy"):
        """
        d: number of examples
        n: number of classes = number of neurons in last layer
        """
        self.id = "loss"
        self.type = type

    def __call__(self, y, o, no_grad=False):
        return self.finish(y, o, no_grad)

    def finish(self, y, o, no_grad):
        self.y = y
        self.o = o
        if self.type == "entropy":
            # self.errors = -np.nansum(self.y * np.log(self.o), axis=1)
            self.errors = -np.sum(self.y * np.log(self.o), axis=1)
            if not no_grad:
                self.DxE = -self.y/self.o
        if self.type == "square":
            # self.errors = 1/2*np.nansum((self.o-self.y) * (self.o-self.y), axis=1)
            self.errors = 1/2*np.sum((self.o-self.y) * (self.o-self.y), axis=1)
            if not no_grad:
                self.DxE = self.o-self.y
        if self.type == "entropy-softmax":
            # self.errors = -np.nansum(y * np.log(self.o), axis=1) # sum((d x n) * (d x n)) = d x 1 
            self.errors = -np.sum(y * np.log(self.o), axis=1) # sum((d x n) * (d x n)) = d x 1
            if not no_grad:
                self.DxE = self.y - self.o
        return self.errors

    def backward(self, Layers):
        """ Backpropagate one by one from (l+1)-layer --> l-layer """
        for l in range(len(Layers)-1, -1, -1):
            if l == len(Layers)-1:  
                Layers[l].backward(self)
            else:
                Layers[l].backward(Layers[l+1])

# class LossSoftxax:
#     def __init__(self, type="entropy"):
#         """
#         d: number of examples
#         n: number of classes = number of neurons in last layer
#         """
#         self.id = "losssoftmax"
#         self.type = type

#     def __call__(self, y, x):
#         return self.finish(y, x)

#     def finish(self, y, x):
#         self.y = y # d x n
#         self.x = x # d x n
#         self.y_hat=softmax(self.x)
#         if self.type == "square":
#             self.errors = (1/2)*np.nansum((self.y_hat-self.y) * (self.y_hat-self.y), axis=1)
#             self.DxE = np.zeros(self.y_hat.shape)
#             for d in range(self.y_hat.shape[0]):
#                 Y = np.tile(self.y_hat[d], [self.y_hat.shape[1],1])
#                 I = np.identity(self.y_hat.shape[1])
#                 self.DxE[d] = np.dot(self.y_hat[d]-self.y[d], Y.T * (I - Y))
        
#         if self.type == "entropy":
#             self.errors = -np.nansum(y * np.log(self.y_hat), axis=1) # sum((d x n) * (d x n)) = d x 1 
#             self.DxE = self.y_hat-self.y
        
#     def backward(self, Layers):
#         """ Backpropagate one by one from (l+1)-layer --> l-layer """
#         for l in range(len(Layers)-1, -1, -1):
#             if l == len(Layers)-1:  
#                 Layers[l].backward(self)
#             else:
#                 Layers[l].backward(Layers[l+1])



if __name__ == '__main__':
    import numpy as np
    from PIL import Image
    import matplotlib.image as mpimg
    from matplotlib import pyplot as plt
    import cv2
    
    with open("P:/data/cifar10/labels.txt") as f:
        print(f.name)
        classes = np.array([l.strip() for l in f])
    img = Image.open("P:/data/cifar10/test/airplane/3_airplane.png")# .convert("L")
    data = np.array(img).reshape(1, 3, 32, 32)
    data.shape
    
    k1 = np.random.uniform(-np.sqrt(1/(3 * 7 * 7)), np.sqrt(1/(3 * 7 * 7)), size = [100, 3, 7, 7])
    b1 = np.random.uniform(-np.sqrt(1/(3 * 7 * 7)), np.sqrt(1/(3 * 7 * 7)), size = [100, 32-7+1, 32-7+1])
    ConvLayer1 = CNL2D([(32, 32), ], (0,0), (1,1), kernels=k1, bias=b1)
    fms1 = ConvLayer1(data)
    # print("Starting image shape:", data.shape, "Feature map's shape:", fms1.shape)

