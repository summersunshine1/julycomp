import numpy as np

path = 'D:/cikm/datasets/data_new/CIKM2017_train/train.txt'
base_path = 'D:/cikm/datasets/data/'

def distribution():
    # f = open(path,"r")  
    i = 0
    # lines = f.readlines()
    with open(path,"r") as f:
        for line in f:
            f2=open(base_path+str(i)+'.txt',"w")
            f2.write(line.strip('\n'))
            f2.close()
            i+=1
            if i==5:
                break
        
if __name__=="__main__":
    distribution()
