import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import argparse

from mynn.model import Model
from mynn.layers import MLP, ReLu, Softmax
from utils.functions import *
from utils.dbs import DataBase


parser = argparse.ArgumentParser()
parser.add_argument("--main_path",              type=str,   default="H:\My Drive\ML\DLib")
parser.add_argument("--data_path",              type=str,   default="P:\data")
parser.add_argument("--train",                  type=bool,  default=True)
parser.add_argument("--epochs",                 type=int,   default=1000)
parser.add_argument("--lr",                     type=float, default=0.003)
parser.add_argument("--batch_size",             type=int,   default=100)
# parser.add_argument("--hidden_layer_neurons",   type=int,   default=[200, 100])
parser.add_argument("--pars_save_path",         type=str,   default="pars/cifar10")
parser.add_argument("--latest_checkpoint_path", type=str,   default=None)
hpars = parser.parse_args()

if __name__=="__main__":
    train_data, train_labels = DataBase(hpars.data_path).load_cifar10("train_cifar.csv")
    test_data, test_labels = DataBase(hpars.data_path).load_cifar10("test_cifar.csv", lab_nom=True)
    train_labs = one_hot(train_labels)
    
    X, y = shuffle(train_data, train_labs)
    testX, testY = shuffle(test_data, test_labels)
    print("Trainset data dims: ", X.shape)
    print("Trainset labels dims: ", y.shape)
    print("Testset data dims:", testX.shape)
    print("Testset labels dims:", testY.shape)

    # Neural Net's Architecture
    layers = [
         MLP([X.shape[1], 500])
        ,ReLu() # end of 1st hidden layer
        ,MLP([500, 300])
        ,ReLu() # end of 1st hidden layer
        ,MLP([300, 100])
        ,ReLu() # end of 1st hidden layer
        ,MLP([100, y.shape[1]])
        ,Softmax() # end 2nd hidden or output layer
        ]
    Net = Model(pipline=layers, loss="entropy")
    if hpars.train:
        Net.train(to_01(X), y, epochs=hpars.epochs, lr=hpars.lr, batch_size=hpars.batch_size, save_pars_path=None)
    
    testX01 = to_01(testX)
    y_prob = Net.prob(testX01)
    y_pred = Net.predict(testX01)
    y_test = testY

    cm = confmtx(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print("Accuracy:", np.diag(cm.to_numpy()).sum()/cm.to_numpy().sum())
    print("AUC:", roc_auc_score(y_test, y_prob, multi_class="ovr"))

    plt.plot([e for e in range(len(Net.train_errors))], Net.train_errors, label="train error")
    plt.plot([e for e in range(len(Net.valid_errors))], Net.valid_errors, label="test error")
    plt.legend(loc="upper right")
    plt.show()

    # Results of confusion matrix:
    #       p_1  p_2  p_3  p_4  p_5  p_6  p_7  p_8  p_9  p_10
    # t_1   264  156   26    6   84   46  122   64  179    53
    # t_2    39  678    1    9   18   22   34   24   72   103
    # t_3   103   87   90   29  149  166  224   58   64    30
    # t_4    45  152   29   95   96  225  172   65   62    59
    # t_5    83  122   40   11  317   59  202   75   64    27
    # t_6    38   96   36   61   73  354  153   76   72    41
    # t_7    52  155   17   16   87   87  475   38   44    29
    # t_8    34  102   13   26   82  106   75  425   61    76
    # t_9    97  179    3    8   33   30   36   27  491    96
    # t_10   10  445    7   14   17   33   27   24   74   349
    # Accuracy: 0.3538
    # AUC: 0.8069610222222222
    #
    # This model's result will be approximently the same each time you run it 
    # The model could not hold many information and thus it need convolutions to draw the information localy.
    # I tried for multiple epochs, batch sizes, learning rates and larger architectures, however it is clear 
    # that multiplication must be changed with convolution to let the model focus in local inforamtion and 
    # not in all the image all the time.

