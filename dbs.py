from functions import shuffle
import numpy as np

class DataBase:
    def __init__(self, filename):
        self.file=filename
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
            for key, parameter in parameters.items():
                self.num_par[key]=len(parameter)
                for l, par in enumerate(parameter):
                    f.write(f"layer_{l}"+":"+key+"\n")
                    for raw in par:
                        for r in raw:
                            if raw[-1]==r:
                                f.write(str(r)+"\n")
                            else:
                                f.write(str(r)+",")                    
                        
    def load_par(self, num_w):    
        parameters={"weights": [None]*num_w, "bias": [None]*num_w}
        with open(self.file, "r") as f:
            l1=-1
            l2=-1
            for line in f:
                line=line.strip()
                if line[:5]=="layer":
                    if line[-7:]=="weights":
                        l1+=1
                        key="weights"
                        parameters[key][l1]=[]
                    elif line[-4:]=="bias":
                        l2+=1
                        key="bias"
                        parameters[key][l2]=[]
                else:
                    line=line.split(",")
                    line=[float(el) for el in line]
                    if key=="weights":
                        parameters[key][l1].append(line)
                    elif key=="bias":
                        parameters[key][l2].append(line)
                    
        for key, par in parameters.items():
            for l in range(len(par)):
                parameters[key][l]=np.array(parameters[key][l], dtype="float32")
        return parameters["weights"], parameters["bias"]

