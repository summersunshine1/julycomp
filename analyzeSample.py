import numpy as np
from getPath import *

pardir = getparentdir()

path = pardir+'/datasets/train/train.txt'
base_path = pardir+'/datasets/data/'

path = pardir+'/datasets/test/testA.txt'
base_path = pardir+'/datasets/datatest/'

def distribution():
    i = 0
    j=1
    totalline = ""
    with open(path,"r") as f:
        for line in f:
            totalline+=line
            # if i%50==0 and not i==0:
            f2=open(base_path+str(j)+'.txt',"w")
            f2.write(totalline)
            f2.close()
            totalline = ""
            j+=1
            i+=1
            
if __name__=="__main__":
    distribution()
