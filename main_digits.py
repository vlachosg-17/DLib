import numpy as np
import sys
from model import Model
from layer import MLP
from functions import *
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from dbs import DataBase
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--train",               type=bool, default=True)
parser.add_argument("--epochs",               type=int, default=100)
parser.add_argument("--lr",                   type=float, default=0.001)
parser.add_argument("--batch_size",           type=int, default=100)
parser.add_argument("--hidden_layer_neurons", type=int, default=[100])
parser.add_argument("--test_prc",             type=int, default=0.4)
parser.add_argument("--pars_save_path",       type=str, default="pars/digits")
parser.add_argument("--latest_checkpoint_path", type=str, default=None)
hpars = parser.parse_args()

if __name__=="__main__":
    train_data, train_labels = DataBase("data").load_digits("train_digits.csv")
    testX = DataBase("data").load_digits("test_digits.csv", lables_true=False)
    train_labs = one_hot(train_labels)
    
    X, y = shuffle(train_data, train_labs)
    # testX, testY = shuffle(test_data, test_labels)
    print("Trainset data dims: ", X.shape)
    print("Trainset labels dims: ", y.shape)
    # print("Testset data dims:", testX.shape)
    # print("Testset labels dims:", testY.shape)

    # Train the Neural Net
    Net = Model(loss = "square", stored_path=hpars.latest_checkpoint_path, step=1)
    nrs = [X.shape[1]] + hpars.hidden_layer_neurons + [y.shape[1]]
    for l in range(len(nrs)-1):
        if l != len(nrs)-2:
            Net.add(MLP(dims = [nrs[l], nrs[l+1]], activation="sigmoid"))
        else:
            Net.add(MLP(dims = [nrs[l], nrs[l+1]], activation="softmax"))
    print(Net)    

    if hpars.train:
        Net.train(X, y, epochs=hpars.epochs, lr=hpars.lr, batch_size=hpars.batch_size, save_pars_path=hpars.pars_save_path)
    
    
    levels = eye_levels(np.unique(train_labels))
    y_prob = np.round(Net.prob(testX), 3)
    y_pred = np.argmax(y_prob, axis=1)
    # y_pred = Net.predict(testX, classes=levels)
    # y_test = testY

    sample=np.random.choice(testX.shape[0], 1)
    for p, img in zip(y_prob[sample], testX[sample]):
        f, (ax1, ax2) = plt.subplots(1, 2) 
        ax1.imshow(img.reshape(28, 28))
        ax1.set_title('The image')

        ax2.bar([i for i in range(10)], p)
        ax2.set_title('The prediction')
        ax2.set_xticks([i for i in range(10)])

        plt.tight_layout()
        plt.show()

    n=5
    sample=np.random.choice(testX.shape[0], n * n)
    testXs = testX[sample]
    y_preds = y_pred[sample]
    fig, axs = plt.subplots(n, n)
    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            axs[i, j].imshow(testXs[i+n*j].reshape(28, 28))
            axs[i, j].set_title("Pred: "+ str(y_preds[i+n*j]))
            axs[i, j].set_axis_off()
    plt.tight_layout()
    plt.show()
    # cm = confmtx(y_test, y_pred)
    # print("Confusion Matrix:")
    # print(cm)
    # print("Accuracy:", np.diag(cm.to_numpy()).sum()/cm.to_numpy().sum())
    # print("AUC:", roc_auc_score(y_test, y_prob, multi_class="ovr"))


    plt.plot([e for e in range(len(Net.train_errors))], Net.train_errors, label="train error")
    plt.plot([e for e in range(len(Net.valid_errors))], Net.valid_errors, label="test error")
    plt.legend(loc="upper right")
    plt.show()