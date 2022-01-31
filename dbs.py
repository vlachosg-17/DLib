from functions import shuffle, split
import numpy as np
import os

class DataBase:
    def __init__(self, path, create=False):
        self.path=path
        self.filepath = None
        if create:
            if not os.path.exists(self.path):
                os.makedirs(self.path)
        self.num_par={}

    def save_drag(self, data, labels, filename):
        self.dims=data.shape
        self.filepath = self.path + "/" + filename
        with open(self.filepath, "w") as f:
            for d, l in zip(data, labels):
                f.write(str(d.item())+","+str(l)+"\n")
    
    def load_drag(self, filename):
        self.filepath = self.path + "/" + filename
        with open(self.filepath, "r") as f:
            data_observations=[]
            labels_obserations=[]
            for line in f:
                line=line.strip().split(",")
                data_observations.append([line[0]])
                labels_obserations.append([line[1]])
        data=np.array(data_observations, dtype="float32")
        labels=np.array(labels_obserations, dtype="int32")
        return data, labels
    
    def load_iris(self, filename, random_seed=0):
        self.filepath = self.path + "/" + filename
        with open(self.filepath, "r") as f:
            labels=[]
            data=[]
            for line in f:
                line=line.strip()
                if len(line)==0:
                    break
                line=line.split(",")
                data.append(line[:-1])
                labels.append(line[-1])
        data, labels = np.array(data, dtype="float32"), np.array(labels)
        np.random.seed(random_seed)
        return shuffle(data, labels)

    def load_pima(self, filename):
        self.filepath = self.path + "/" + filename
        with open(self.filepath, "r") as f:
            labels=[]
            data=[]
            k=0
            for line in f:
                if k==0:
                    next    
                line=line.strip()
                if len(line)==0:
                    break
                line=line.split(",")
                data.append(line[:-1])
                labels.append(line[-1])
                k+=1
        data, labels = np.array(data, dtype="float32"), np.array(labels, dtype="int8")
        return data, labels
    
    def load_cifar10(self, filename):
        self.filepath = self.path + "/" + filename
        data = []
        labels = []
        with open(self.filepath, "r") as f:
            for line in f:
                l = line.strip().split(",")
                data.append(l[:-1])
                labels.append(l[-1])
        return np.array(data, dtype=np.int32), np.array(labels)

    def load_digits(self, filename, lables_true=True):
        self.filepath = self.path + "/" + filename
        data = []
        labels = []
        k=0
        with open(self.filepath, "r") as f:
            for line in f:
                if k != 0:
                    l = line.strip().split(",")
                    data.append(l[1:])
                    labels.append(l[0])
                    k+=1
                else:
                    k+=1
        d, l = np.array(data, dtype=np.int32), np.array(labels, dtype=np.int32)
        if lables_true:
            return d, l
        else:
            return np.hstack([l.reshape(-1,1), d])

    def save_par(self, parameters, filename):
        self.filepath = self.path + "/" + filename
        with open(self.filepath, "w") as f:
            for l, parameter in enumerate(parameters):
                f.write(f"layer {l}"+":" +"\n")
                for raw in parameter:
                    for r in raw:
                        if raw[-1]==r:
                            f.write(str(r)+"\n")
                        else:
                            f.write(str(r)+",") 

    def save_error(self, errors, filename):
        self.filepath = self.path + "/" + filename
        with open(self.filepath, "w") as f:
            for err in errors:
                f.write(f"{err}"+"\n")

    def load_par(self, filename): 
        self.filepath = self.path + "/" + filename   
        pars = []
        with open(self.filepath, "r") as f:
            while True:
                line = f.readline().strip()
                if len(line) == 0:
                    break
                if line.split(" ")[0] == "layer":
                    pars.append([])
                else:
                    ln = line.split(",")
                    pars[-1].append(ln)

        return [np.array(p, dtype="float32") for p in pars]
    
    def load_errors(self, filename):
        self.filepath = self.path + "/" + filename
        errors=[]
        with open(self.filepath, "r") as f:
            for line in f:
                errors.append(float(line.strip()))
        return errors