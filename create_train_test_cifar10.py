import numpy as np
from tqdm import  tqdm
import os
from PIL import Image
from matplotlib import image
from matplotlib import pyplot as plt

path="P:/data"
data = []
labels = []
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
for c in tqdm(classes, ncols=70):
    L = os.listdir(path+"/cifar10/train/" + c)
    for l in L:
        data.append(np.array(Image.open(path+"/cifar10/train/"+c +"/"+l).convert("L")).flatten())
        labels.append(c)
train_dataset = np.hstack([np.array(data), np.array(labels).reshape(len(labels),1)])

np.savetxt(path+"/train_cifar.csv", train_dataset, fmt='%s', delimiter=",")


data = []
labels = []
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
for c in tqdm(classes, ncols=70):
    L = os.listdir(path+"/cifar10/test/" + c)
    for l in L:
        data.append(np.array(Image.open(path+"/cifar10/test/"+c +"/"+l).convert("L")).flatten())
        labels.append(c)
test_dataset = np.hstack([np.array(data), np.array(labels).reshape(len(labels),1)])

np.savetxt(path+"/test_cifar.csv", test_dataset, fmt='%s', delimiter=",")
