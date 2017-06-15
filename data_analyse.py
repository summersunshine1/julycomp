import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pylab import *
import time
from sklearn.decomposition import IncrementalPCA
import os

from commonLib import *
from getPath import *
pardir = getparentdir()
datadir = pardir+'/datasets/datatrain/'
batch_size = 64

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
	
def pca_analyze():
    file_list = listfiles(datadir)
    i = 0
    x_arr = []
    y_arr = []
    n_components =  101*101
    for file in file_list:
        if i%batch_size==0 and not i==0:
            ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size,whiten  = True)
            new_arr = ipca.fit_transform(x_arr)
            write_pca_to_file(new_arr,y_arr)
            print("write:"+str(i))
            y_arr = []
            x_arr = []
        x,y = getTrainData(file)
        x_arr.append(x[0])
        y_arr.append(str(y[0][0]))
        i+=1
        
def write_pca_to_file(new_arr,y_arr):
    filepath = pardir+'/julycomp/pca.txt'
    if not os.path.exists(filepath):
        fw = open(filepath,'w')
    else:
        fw = open(filepath,'a')
    for i in range(len(new_arr)):
        info = [y_arr[i]]
        temp = [str(t) for t in new_arr[i]]
        line = ';'.join(temp)
        info.append(line)
        outline = '|'.join(info)+'\n'
        fw.writelines(outline)

    fw.close() 
    
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
        
def showPrecipitation():
    file_list = listfiles(datadir)
    arr_y = []
    i = 0
    duration = 0
    for file in file_list:
        start_time = time.time()
        x,y = getTrainData(file)
        duration += time.time()-start_time
        if i%100==0:
            print(str(i)+":"+str(duration))
            duration = 0
        arr_y.append(y[0])
        i+=1
    plt.hist(arr_y)
    plt.show()
    

if __name__=="__main__":
    # showImg(datadir+'1.txt')
    pca_analyze()
    
	

	
