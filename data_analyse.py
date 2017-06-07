import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pylab import *

from commonLib import *
from getPath import *
pardir = getparentdir()

datadir = pardir+'/datasets/datatrain/'
def getTrainData(path):
    x = []
    y = []
    with open(path,'r') as f:
        for line in f:
            arr = line.split(',')
            x_temp = arr[2].split()
            x.append([float(t) for t in x_temp])
            y.append([float(arr[1])])
    x = np.array(x)
    y = np.array(y)
    return x,y
	
def analyze():
    file_list = listfiles(datadir)
    arr_x = []
    arr_y = []
    for file in file_list:
        x,y = getTrainData(file)
        arr_x.append(x[0][50*101+50]/255)
        arr_y.append(y[0])
    plt.scatter(arr_x,arr_y)
    plt.show()
    
def showImg(path):
    x,y = getTrainData(path)
    timelen = 15
    heightlen = 4
    for i in range(timelen):
        plt.figure()
        for j in range(heightlen):
            tempindex = i*101*101*4+j*101*101
            img = x[0][tempindex:tempindex+101*101]
            img = img.reshape(101,101)
            plt.subplot(220+j+1)
            plt.imshow(img)
            plt.title("time:"+str(i+1)+" height:"+str(j))
        plt.show()

if __name__=="__main__":
    showImg(datadir+'1.txt')
    
	

	
