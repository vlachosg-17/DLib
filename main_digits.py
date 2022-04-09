import numpy as np
import matplotlib.pyplot as plt
import argparse

from mynn.model import Model
from mynn.layers import MLP, ReLu, Softmax
from utils.functions import *
from utils.dbs import DataBase

parser = argparse.ArgumentParser()
parser.add_argument("--main_path",              type=str,   default="H:\My Drive\ML\DLib")
parser.add_argument("--data_path",              type=str,   default="P:\data")
parser.add_argument("--train",                  type=bool,  default=False)
parser.add_argument("--epochs",                 type=int,   default=100)
parser.add_argument("--lr",                     type=float, default=0.001)
parser.add_argument("--batch_size",             type=int,   default=100)
# parser.add_argument("--hidden_layer_neurons",   type=int,   default=[100])
parser.add_argument("--save_dir",         type=str,   default="pars/digits")
parser.add_argument("--latest_checkpoint_path", type=str,   default=None)
hpars = parser.parse_args()

if __name__=="__main__":
    train_data, train_labels = DataBase(hpars.data_path).load_digits("train_digits.csv")
    testX = DataBase(hpars.data_path).load_digits("test_digits.csv", lables_true=False)
    train_labs = one_hot(train_labels)
    
    X, y = shuffle(train_data, train_labs)
    # X=to_01(X)
    # testX, testY = shuffle(test_data, test_labels)
    print("Trainset data dims: ", X.shape)
    print("Trainset labels dims: ", y.shape)
    # print("Testset data dims:", testX.shape)
    # print("Testset labels dims:", testY.shape)
    
    # Neural Net's Architecture
    layers = [
         MLP([X.shape[1], 100])
        ,ReLu() # end of 1st hidden layer
        # ,MLP([100, 80])
        # ,ReLu() # end of 2nd hidden layer
        # ,MLP([80, 50])
        # ,ReLu() # end of 3rd hidden layer
        ,MLP([100, y.shape[1]])
        ,Softmax() # end 4th hidden or output layer
        ]
    Net = Model(pipline=layers, loss="entropy", stored_path = f"{hpars.main_path}\{hpars.save_dir}")
    if hpars.train:
        Net.train(
            X=to_01(X), 
            y=y, 
            epochs=hpars.epochs, 
            lr=hpars.lr, 
            batch_size=hpars.batch_size, 
            save_path=f"{hpars.main_path}\{hpars.save_dir}"
        )
    
    testX01 = to_01(testX)
    y_prob = np.round(Net.prob(testX01), 3)
    y_pred = Net.predict(testX01)
    
    sample=np.random.choice(testX01.shape[0], 1)
    # sample=range(0, 5)
    for p, img in zip(y_prob[sample], testX01[sample]):
        f, (ax1, ax2) = plt.subplots(1, 2) 
        ax1.imshow(img.reshape(28, 28))
        ax1.set_title('The image')

        ax2.bar([i for i in range(10)], p)
        ax2.set_title('The prediction')
        ax2.set_xticks([i for i in range(10)])

        plt.tight_layout()
        plt.show()

    n=5
    sample=np.random.choice(testX01.shape[0], n * n)
    testXs = testX01[sample]
    y_preds = y_pred[sample]
    fig, axs = plt.subplots(n, n)
    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            axs[i, j].imshow(testXs[i+n*j].reshape(28, 28))
            axs[i, j].set_title("Pred: "+ str(y_preds[i+n*j]))
            axs[i, j].set_axis_off()
    plt.tight_layout()
    plt.show()


    plt.plot([e for e in range(len(Net.train_errors))], Net.train_errors, label="train error")
    plt.plot([e for e in range(len(Net.valid_errors))], Net.valid_errors, label="test error")
    plt.legend(loc="upper right")
    plt.show()