import numpy as np

class Opt:
    def __init__(self, lr):
        self.lr = lr

    def sgd(self, Layers):
        for d in range(Layers[0].x.shape[0]):
            for Layer in Layers:
                Layer.w -= self.lr * Layer.DwL[d, :, :]
                Layer.b -= self.lr * Layer.DbL[d, :]

    def gd(self, Layers):
        for Layer in Layers: 
            Layer.w -= self.lr * np.sum(Layer.DwL, axis=0)
            Layer.b -= self.lr * np.sum(Layer.DbL, axis=0)
