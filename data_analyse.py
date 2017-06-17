import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pylab import *
import time
from sklearn.decomposition import IncrementalPCA
import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

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
    
def goal_position():
    file_list = listfiles(datadir)
    i = 0
    x_arr = []
    y_arr = []
    for file in file_list:
        x,y = getTrainData(file)
        temp = []
        for i in range(4*15):
            temp.append(x[0][i*101*101+50*101+50])
        x_arr.append(temp)
        y_arr.append(y[0][0])
    y_arr = np.array(y_arr)
    return x_arr,y_arr
	
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
    print(y)
    timelen = 15
    heightlen = 4
    for i in range(timelen):
        for j in range(heightlen):
            tempindex = i*101*101*heightlen+j*101*101
            img = x[0][tempindex:tempindex+101*101]
            img = img.reshape(101,101)
            plt.subplot(heightlen, timelen, j*timelen + i+1)#m表示是图排成m行，n表示图排成n列(m,n,p) ,p表示第几个图
            plt.imshow(img)
            plt.title(str(i)+","+str(j))
    plt.suptitle(str(y[0][0]))
    plt.show()
    
def write_precipitation(arr_y):
    fw = open(pardir+'/julycomp/precip.txt','w')
    arr_y = [str(t) for t in arr_y]
    line = ','.join(arr_y)
    fw.writelines(line)
    fw.close()

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
    write_precipitation(arr_y)
    plt.hist(arr_y)
    plt.show()
    
def encoder(arr, bins):
    ohe = OneHotEncoder(sparse=False,n_values = bins)#categorical_features='all',
    ohe.fit(arr)
    return ohe.transform(arr)   
    
def read_preci():
    path = pardir+'/julycomp/precip.txt'
    with open(path,'r') as fw:
        for line in fw:
            arr = line.split(',')
    arr = [float(t) for t in arr]
    bins_ = int(np.max(arr)-np.min(arr))
    labels_ = []
    for i in range(bins_):
        labels_.append(i)
    print(bins_)
    out = pd.cut(arr,bins = int((np.max(arr)-np.min(arr))),labels = labels_,include_lowest=True)
    
    out = [[o] for o in out]
    code = encoder(out, bins_)
    print(len(code[0]))
    
    # print(colbin)
    # print(np.max(arr))
    # print(np.min(arr))
    # plt.hist(arr, int((np.max(arr)-np.min(arr))/1))
    # plt.show()

if __name__=="__main__":
    # file_list = listfiles(datadir) 
    # for file in file_list:
        # showImg(file)
    # showImg(datadir+'1.txt')
    # pca_analyze()
    # showPrecipitation()
    read_preci()
	

	
