from . import np

class Opt:
    def __init__(self, lr):
        self.lr = lr

    def sgd(self, Layers):
        for d in range(Layers[0].x.shape[0]):
            for Layer in Layers:
                # for param, grad in zip(Layer.params, Layer.param_gradients):
                if Layer.id in ["MLP", "CNL2D"]:
                    Layer.weights -= self.lr * Layer.DwE[d]
                    Layer.bias -= self.lr * Layer.DbE[d]

    def gd(self, Layers):
        for Layer in Layers: 
            if Layer.id in ["MLP", "CNL2D"]:
                Layer.weights -= self.lr * np.sum(Layer.DwE, axis=0)
                Layer.bias -= self.lr * np.sum(Layer.DbE, axis=0)
