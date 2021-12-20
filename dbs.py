from functions import shuffle, split
import numpy as np
import os

class DataBase:
    def __init__(self, filename):
        self.file=filename
        self.dirfile = "/".join(filename.split("/")[:-1])
        self.name = filename.split("/")[-1]
        print(self.name, self.dirfile, self.file)
        if not os.path.exists(self.dirfile):
            os.makedirs(self.dirfile)
        self.num_par={}

    def save_drag(self, data, labels):
        self.dims=data.shape
        with open(self.file, "w") as f:
            for d, l in zip(data, labels):
                f.write(str(d.item())+","+str(l)+"\n")
    
    def load_drag(self):
        with open(self.file, "r") as f:
            data_observations=[]
            labels_obserations=[]
            for line in f:
                line=line.strip().split(",")
                data_observations.append([line[0]])
                labels_obserations.append([line[1]])
        data=np.array(data_observations, dtype="float32")
        labels=np.array(labels_obserations, dtype="int32")
        return data, labels
    
    def load_iris(self):
        with open(self.file, "r") as f:
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
        # np.random.seed(0)
        return shuffle(data, labels)

    def save_par(self, parameters):
        with open(self.file, "w") as f:
            for l, parameter in enumerate(parameters):
                f.write(f"layer {l}"+":" +"\n")
                for raw in parameter:
                    for r in raw:
                        if raw[-1]==r:
                            f.write(str(r)+"\n")
                        else:
                            f.write(str(r)+",")                    
                        
    def load_par(self):    
        pars = []
        with open(self.file, "r") as f:
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
