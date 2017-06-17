import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.svm import SVR,LinearSVR
from sklearn import preprocessing
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import KFold
from sklearn.externals import joblib

from getPath import *
from commonLib import *
from data_analyse import *
pardir = getparentdir()

learning_rate = 0.01
training_epochs = 1000
datadir = pardir+'/datasets/data/'
input_nodes = 15*4*101*101

def getTrainData(path):
    x = []
    y = []
    with open(path,'r') as f:
        for line in f:
            arr = line.split(',')
            x.append(arr[2].split())
            y.append([float(arr[1])])
    # x = preprocessing.scale(x)
    y = np.array(y)
    return x,y
    
def createmodel():
    x = tf.placeholder(tf.float32, shape = [None, input_nodes])
    y = tf.placeholder(tf.float32)
    w = tf.Variable(tf.truncated_normal([input_nodes,1],0.1))
    b = tf.Variable(np.random.randn(), name="bias", dtype=tf.float32)
    pred = tf.add(tf.matmul(x,w),b)
    cost = tf.reduce_mean(tf.pow(pred - y, 2))/2
    train_op =  tf.train.AdamOptimizer(learning_rate).minimize(cost) 
    file_list = listfiles(datadir)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for file in file_list:
            train_x,train_y = getTrainData(file)
            for epoch in range(training_epochs):
                _,c,predic_y,w_,b_=sess.run((train_op, cost,pred,w,b), feed_dict={x: train_x, y: train_y})
                if epoch%100==0 and not epoch == 0:
                    print("cost is "+str(c))
            plt.plot(train_y)
            plt.plot(predic_y)
            plt.show()
            break

# Split a dataset into k folds
def cross_validation_split(file_list, n_folds):
    kf = KFold(n_splits=n_folds,random_state=None,shuffle=False)
    kf.split(file_list)
    train_indexs = []
    test_indexs = []
    for train_index, test_index in kf.split(file_list):
        train_indexs.append(train_index)
        test_indexs.append(test_index)
    return train_indexs,test_indexs  

def split_train_test(filelist,percent):
    splitpoint = int(len(filelist)*percent)
    train_files = filelist[:splitpoint]
    test_files = filelist[splitpoint:]
    return train_files,test_files

def rmse(predict_y,ground_y):
    ground_y = np.array(ground_y)
    predict_y = np.array(predict_y)
    score = np.sqrt(np.mean(np.power(ground_y - predict_y, 2)))
    return score

def createlinearmodel():
    file_list = listfiles(datadir)
    train_files,test_files = split_train_test(file_list,0.9)
    train_indexs,test_indexs = cross_validation_split(train_files,10)
    model = SGDRegressor()
    
    scores = []
    k = 1 
    for i in range(len(train_indexs)):
        predict_y = []
        ground_y = []
        for j in range(len(train_indexs[i])):
            train_x,train_y = getTrainData(train_files[j])
            model.partial_fit(train_x, train_y.ravel())
            print("train"+str(k))
            k+=1
        for j in range(len(test_indexs[i])):
            train_x,train_y = getTrainData(train_files[j])
            y = model.predict(train_x)
            ground_y.append(train_y)
            predict_y.append(y)
        score = rmse(predict_y,ground_y)
        print(str(i)+" fold: "+str(score))
        scores.append(score)        
    print(scores)
    
    #predict
    predict_y = []
    ground_y = []
    for file in test_files:
        train_x,train_y = getTrainData(file)
        y = model.predict(train_x)
        ground_y.append(train_y)
        predict_y.append(y)
    score = rmse(predict_y,ground_y)
    print("final result" +str(score))
    path = pardir+'/julycomp/model/lr.pkl'
    joblib.dump(model, path)
    # for file in file_list:
        # print(file)
        # train_x,train_y = getTrainData(file)
        # if i>9000: 
            # y = model.predict(train_x)
            # ground_y.append(train_y)
            # predict_y.append([y])
            # break
        # else:
            # model.partial_fit(train_x, train_y.ravel())
            # print("train"+str(i))
        # i+=1
    # predict_y = np.array(predict_y)
    # ground_y = np.array(ground_y)
    # print(np.sqrt(np.mean(np.power(ground_y - predict_y, 2))))  

def continue_train():
    file_list = listfiles(datadir)
    train_files,test_files = split_train_test(file_list,0.9)
    path = pardir+'/julycomp/model/lr.pkl'
    model = joblib.load(path)
    train_files,test_files = split_train_test(test_files,0.9)
    # i=0
    # for file in train_files:
        # train_x,train_y = getTrainData(file)
        # model.partial_fit(train_x, train_y.ravel())
        # print("train:"+str(i))
        # i+=1
    predict_y = []
    ground_y = []
    for file in test_files:
        train_x,train_y = getTrainData(file)
        y = model.predict(train_x)
        ground_y.append(train_y)
        predict_y.append(y)
    print(predict_y)
    score = rmse(predict_y,ground_y)
    print("final result" +str(score))
    # joblib.dump(model, path)
    
def get_pca_data(path):
    y_arr = []
    x_arr = []
    with open(path,'r') as f:
        for line in f:
            arr = line.split('|')
            y_arr.append([float(arr[0])])
            temp = [float(t) for t in arr[1].split(';')]
            x_arr.append(temp)
    y_arr = np.array(y_arr)
    return x_arr,y_arr
    
def create_one_model():
    # datapath = pardir+'/julycomp/pca.txt'
    x_arr,y_arr = goal_position()
    clf = LinearSVR(C=1, epsilon=0.1)
    clf.fit(x_arr, y_arr.ravel()) 
    score = make_scorer(rmse, greater_is_better=False)
    scores = -cross_val_score(clf, x_arr,y_arr.ravel(),cv=10, scoring = score)
    print(np.mean(scores))
    path = pardir+'/julycomp/model/onelr.pkl'
    joblib.dump(clf, path)
    
if __name__=="__main__":
    # createlinearmodel()
    create_one_model()        

    
                
        
    
    
    