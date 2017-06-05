import numpy as np
from sklearn.externals import joblib
from sklearn import preprocessing
from getPath import *
pardir = getparentdir()
datadir = pardir+'/datasets/datatest/'
model_path = pardir+'/julycomp/model/lr.pkl'
result_path = pardir+'/datasets/test/res.csv'

def getTrainData(path):
    x = []
    y = []
    with open(path,'r') as f:
        for line in f:
            arr = line.split(',')
            x.append(arr[2].split())
            y.append([float(arr[1])])
    x = preprocessing.scale(x)
    y = np.array(y)
    return x,y

def predict_and_write():
    model = joblib.load(model_path)

    res = []
    for i in range(1,2001):
        file_path = datadir+str(i)+'.txt'
        train_x,train_y = getTrainData(file_path)
        y = model.predict(train_x)
        res.append(y[0])
    f=open(result_path,"w")
    for r in res:
        f.writelines(str(r)+'\n')
    f.close()
    
if __name__=="__main__":
    predict_and_write()
        
            
        
    