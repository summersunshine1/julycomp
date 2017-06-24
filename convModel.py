from numpy import *
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pylab import *
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import math

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from getPath import *
pardir = getparentdir()
train_data_dir = pardir+'/datasets/datatrain/'
test_data_dir = pardir +'/datasets/datatest/'
result_path = pardir+'/datasets/test/resconv.csv'
from commonLib import *

pic_size = 101

conv1_kernal_size = 101
conv2_kernal_size = 3
conv1_num = 64
conv2_num = 16

pool1 = 2
pool2 = 2

fc_hidden_num = 1024
# fc1_hidden_num = 1024

# learning_rate = 1e-4
batch_size = 64
dropout_prob = 0.75
channels = 4
pic_length = 15
epochs = 20
strides_1 = 101
strides_2 = 1

def batchnorm(Ylogits, is_test, iteration, offset):
    y_shape = Ylogits.get_shape()
    axis = list(range(len(y_shape) - 1))

    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999) # adding the iteration prevents from averaging across non-existing iterations
    bnepsilon = 1e-5
    mean, variance = tf.nn.moments(Ylogits, axis)
    update_moving_everages = exp_moving_avg.apply([mean, variance])#adds shadow copies of trained variables and add ops that maintain a moving average of the trained variables in their shadow copies
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)#give access to the shadow variables and their names
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
    return Ybn, update_moving_everages

def encoder(arr,bins):
    ohe = OneHotEncoder(sparse=False,n_values = bins)#categorical_features='all',
    ohe.fit(arr)
    return ohe.transform(arr) 

def get_y():
    path = pardir+'/julycomp/precip.txt'
    with open(path,'r') as fw:
        for line in fw:
            arr = line.split(',')
    arr = [float(t) for t in arr]
    bins_ = int(np.max(arr)-np.min(arr))
    labels_ = []
    for i in range(bins_):
        labels_.append(i)
    out = pd.cut(arr,bins = int((np.max(arr)-np.min(arr))),labels = labels_,include_lowest=True)
    out = [[o] for o in out]
    out = encoder(out,bins_)
    print(len(out[0]))
    return out,bins_,arr
    
def getTrainData(path):
    x = []
    y = []
    with open(path,'r') as f:
        for line in f:
            arr = line.split(',')
            x_temp = arr[2].split()
            x = [float(t) for t in x_temp]
            c_x = []
            k = 0
            for i in range(channels):
                for j in range(pic_length):
                    tempindex = j*channels + i
                    img = x[tempindex*101*101:(tempindex+1)*101*101]
                    c_x = c_x+img
            # y.append(float(arr[1]))
    x = np.array(c_x)
    # y = np.array(y)
    return x,y
    
def variable_with_weight_loss(shape, stddev, w1):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if w1 is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), w1, name = 'weight_loss')
        tf.add_to_collection("losses", weight_loss)
    return var

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)
    
def bias_variable(shape):
    initial = tf.constant(0.1,shape = shape)
    return tf.Variable(initial)
    
def conv2d(x,w,strides):
    return tf.nn.conv2d(x,w,[1,strides,strides,1],padding = 'SAME')#batch height width channel
    
def max_pool(x,klen):
    return tf.nn.max_pool(x,ksize = [1,klen,klen,1], strides = [1,klen,klen,1],padding = 'SAME')

def print_tensor(t):
    print(t.get_shape().as_list())
    
def covNetwork():
    train_y, category,_ = get_y()
    max_learning_rate = 0.02
    min_learning_rate = 0.001
    decay_speed = 1600
    
    tst = tf.placeholder(tf.bool)
    iter = tf.placeholder(tf.int32)
    lr = tf.placeholder(tf.float32)
    
    x = tf.placeholder(tf.float32,[None,pic_size*pic_size*pic_length*channels])
    y = tf.placeholder(tf.float32,shape = [None, category])
    x_img = tf.reshape(x,[-1, pic_size, pic_size*pic_length, channels])
    
    w_conv1 = weight_variable([conv1_kernal_size, conv1_kernal_size, channels, conv1_num])
    b_conv1 = bias_variable([conv1_num])
    
    h_conv1 = conv2d(x_img, w_conv1, strides_1)
    x1bn,update_ema1   = batchnorm(h_conv1, tst, iter, b_conv1)
    h_conv1 = tf.nn.relu(x1bn)
    
    h_pool1 = max_pool(h_conv1, pool1)
    print_tensor(h_conv1)
    print_tensor(h_pool1)
    
    w_conv2 = weight_variable([conv2_kernal_size, conv2_kernal_size, conv1_num, conv2_num])
    b_conv2 = bias_variable([conv2_num])
    h_conv2 = conv2d(h_pool1, w_conv2, strides_2)
    x2bn, update_ema2  = batchnorm(h_conv2, tst, iter, b_conv2)
    h_conv2 = tf.nn.relu(x2bn)
    
    # h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2)+b_conv2)
    h_pool2 = max_pool(h_conv2, pool2)
    print_tensor(h_conv2)
    print_tensor(h_pool2)
    
    temp_w = math.ceil(pic_size/(strides_1*pool1*strides_2*pool2))
    temp_h = math.ceil(pic_size*pic_length/(strides_1*pool1*strides_2*pool2))
    # print_tensor(h_conv2)
    # print_tensor(h_pool2)
    # batch_size = x.get_shape()[0]
    # h_pool2_size = tf.size(h_pool2)
    temp = temp_w*temp_h*conv2_num
    h_pool2_flat = tf.reshape(h_pool2, [-1,temp])
    
    w_fc1 = weight_variable([temp, fc_hidden_num])
    # w_fc1 = variable_with_weight_loss([dim, fc_hidden_num], stddev=0.04, w1 = 0.004)
    bias_fc1 = bias_variable([fc_hidden_num])
    h_pool2_f = tf.matmul(h_pool2_flat, w_fc1)
    fc1_bn,update_ema3  = batchnorm(h_pool2_f, tst, iter, bias_fc1)
    h_fc1 = tf.nn.relu(fc1_bn)
    # h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + bias_fc1)
    
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
  
    w_fc2 = weight_variable([fc_hidden_num, category])
    # w_fc2 = variable_with_weight_loss([fc_hidden_num, category],stddev=0.04, w1 = 0.004)
    bias_fc2 = bias_variable([category])
    # h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, w_fc2) + bias_fc2)
    # h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
    # w_fc3 = weight_variable([fc1_hidden_num, category])
    # w_fc2 = variable_with_weight_loss([fc_hidden_num, category],stddev=0.04, w1 = 0.004)
    # bias_fc3 = bias_variable([category])
    # h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, w_fc3) + bias_fc3)
    # h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)
    # w_fc4 = weight_variable([fc3_hidden_num, category])
    # w_fc2 = variable_with_weight_loss([fc_hidden_num, category],stddev=0.04, w1 = 0.004)
    # bias_fc4 = bias_variable([category])
    update_ema = tf.group(update_ema1, update_ema2, update_ema3)
    if category>1:
        y_conv = tf.nn.softmax(tf.matmul(h_fc1, w_fc2) + bias_fc2) 
        y_res = tf.argmax(y_conv,1)
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y*tf.log(y_conv),reduction_indices = [1]))
    else:
        y_conv = tf.matmul(h_fc1_drop, w_fc2) + bias_fc2
        # y_res = tf.argmax(y_conv)
        cross_entropy = tf.reduce_sum(tf.pow(y-y_conv,2))
            # tf.add_to_collection('losses',cross_entropy)
            # loss = tf.add_n(tf.get_collection('losses'))
    train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
    if category>1:
        correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    else:
        accuracy = tf.reduce_mean(tf.reduce_sum(tf.pow(y-y_conv,2)))

    saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    file_list = listfiles(train_data_dir)
    i = 0  
    x_arr = []
    y_arr = []
 
    for k in range(epochs):
        i = 0
        t_count = 0
        for file in file_list:
            train_x,_ = getTrainData(file)
            if i%batch_size==0 and not i==0:
                x_arr = np.array(x_arr)
                y_arr = np.array(y_arr)
                learning_rate = min_learning_rate #+ (max_learning_rate - min_learning_rate) * math.exp(-t_count/decay_speed)
                _,train_accuracy,loss=sess.run([train_step, accuracy,cross_entropy], feed_dict = {x:x_arr,y:y_arr,keep_prob:dropout_prob, iter:t_count, lr:learning_rate,tst:False})
                sess.run(update_ema, {x:x_arr,y:y_arr, tst: False, iter: t_count, keep_prob:dropout_prob}) 
                x_arr = []
                y_arr = []
                # train_step.run(feed_dict = {x:x_arr,y:y_arr,keep_prob:dropout_prob})
                # train_accuracy = accuracy.eval(feed_dict = {x:x_arr,y:y_arr,keep_prob:dropout_prob})
                print("step %d, epoch %d, accuracy %g,loss %g"%(i/batch_size,k,train_accuracy,loss))
                t_count+=1
            x_arr.append(train_x)
            y_arr.append(train_y[i])
            i+=1
    if len(x_arr)>0:
        x_arr = np.array(x_arr)
        y_arr = np.array(y_arr)
        learning_rate = min_learning_rate #+ (max_learning_rate - min_learning_rate) * math.exp(-t_count/decay_speed)
        _,train_accuracy=sess.run([train_step, accuracy], feed_dict = {x:x_arr,y:y_arr,keep_prob:dropout_prob, iter:t_count, lr:learning_rate,tst:False})
        sess.run(update_ema, {x:x_arr,y:y_arr, tst: False, iter: t_count, keep_prob:dropout_prob})
        print("final accuracy %g"%(train_accuracy))
        x_arr = []
        y_arr = []
    save_path = saver.save(sess, pardir+"/julycomp/model/cnn.ckpt")
    file_list = listfiles(test_data_dir)
    res = []
    for file in file_list:
        test_arr,_ = getTrainData(file)
        predict_y = y_res.eval(feed_dict = {x:[test_arr],keep_prob:1, iter:t_count,tst:True})
        res.append(predict_y[0])
    sess.close()
    writeres(res)
    
def writeres(res):
    f=open(result_path,"w")
    for r in res:
        f.writelines(str(r)+'\n')
    f.close()
    
def main(argv=None):
    covNetwork()
    
if __name__=="__main__":
    tf.app.run(main)
    
    
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    