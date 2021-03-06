import tensorflow as tf
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import math
from tensorflow.contrib import rnn

from commonLib import *

from getPath import *
pardir = getparentdir()
train_data_dir = pardir+'/datasets/datatrain/'
test_data_dir = pardir +'/datasets/datatest/'
result_path = pardir+'/datasets/test/resconvlstmlast.csv'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
pic_size = 101

conv1_kernal_size = 3
conv2_kernal_size = 3
conv1_num = 16
conv2_num = 64

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
strides_1 = 4
strides_2 = 2

num_layers = 2
hidden_size = 300


def getTrainData(path):
    x_arr = []
    with open(path,'r') as f:
        for line in f:
            arr = line.split(',')
            x_temp = arr[2].split()
            x = [float(t) for t in x_temp]
            for i in range(pic_length):
                x_arr.append(x[101*101*4*i:101*101*4*(i+1)])
    x_arr = np.array(x_arr)
    return x_arr

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

def conv(x_img, tst,keep_prob):
    w_conv1 = weight_variable([conv1_kernal_size, conv1_kernal_size, channels, conv1_num])
    b_conv1 = bias_variable([conv1_num])
    
    h_conv1 = conv2d(x_img, w_conv1, strides_1)
    x1bn,update_ema1   = batchnorm(h_conv1, tst, iter, b_conv1)
    h_conv1 = tf.nn.relu(x1bn)
    
    h_pool1 = max_pool(h_conv1, pool1)
    # print_tensor(h_conv1)
    # print_tensor(h_pool1)
    
    w_conv2 = weight_variable([conv2_kernal_size, conv2_kernal_size, conv1_num, conv2_num])
    b_conv2 = bias_variable([conv2_num])    
    h_conv2 = conv2d(h_pool1, w_conv2, strides_2)
    x2bn, update_ema2  = batchnorm(h_conv2, tst, iter, b_conv2)
    h_conv2 = tf.nn.relu(x2bn)
    
    # h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2)+b_conv2)
    h_pool2 = max_pool(h_conv2, pool2)
    # print_tensor(h_conv2)
    # print_tensor(h_pool2)
    
    temp_w = math.ceil(pic_size/(strides_1*pool1*strides_2*pool2))
    temp_h = math.ceil(pic_size/(strides_1*pool1*strides_2*pool2))
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
    

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    update_ema = tf.group(update_ema1, update_ema2, update_ema3)
    return h_fc1_drop,update_ema
    
def printtensor(tensor):
    list = tensor.get_shape().as_list()
    print(list)
    
def lstmcell(keep_prob):
    lstm_cell = rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, reuse=tf.get_variable_scope().reuse)
    lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob = keep_prob)
    return lstm_cell
    
def lstm(x,category,keep_prob):
    # printtensor(x) 
    x = tf.reshape(x,[pic_length,-1, fc_hidden_num])
    # lstm_cell = rnn.BasicLSTMCell(hidden_size, forget_bias=1.0)
    with tf.variable_scope('lstm1') as scope:
        # lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0)
        # cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob = keep_prob)
        cell = tf.contrib.rnn.MultiRNNCell([lstmcell(keep_prob) for _ in range(num_layers)],state_is_tuple = True)
        state = cell.zero_state(batch_size, tf.float32)
        outputs = []
        w = weight_variable([hidden_size, category])
        b = bias_variable([category])
        for i in range(pic_length):
            if i>0:
                scope.reuse_variables()
            output, state = cell(x[i,:,:],state)#none*300
            outputs.append(output)
    # outputs, states = rnn.static_rnn(lstm_cell,x, dtype=tf.float32)
    outputs = output
    
    # outputs, states = cell(x, state,)
    return tf.matmul(outputs, w) + b
    
def convlstm():
    train_y, category,_ = get_y()
    max_learning_rate = 0.02
    min_learning_rate = 0.0001
    decay_speed = 1600
    
    tst = tf.placeholder(tf.bool)
    iter = tf.placeholder(tf.int32)
    lr = tf.placeholder(tf.float32)
    
    x = tf.placeholder(tf.float32,[None,pic_size*pic_size*channels])
    y = tf.placeholder(tf.float32,shape = [None, category])
    keep_prob = tf.placeholder(tf.float32)
    x_img = tf.reshape(x,[-1, pic_size, pic_size, channels])
    conv_output,update_ema = conv(x_img,tst,keep_prob)
    
    y_pred= lstm(conv_output,category,keep_prob)
    # y_pred = tf.reduce_mean(y_pred, 0)
    
    y_res = tf.argmax(y_pred, 1)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=min_learning_rate).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(y_pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    file_list = listfiles(train_data_dir)
    learning_rate = min_learning_rate #+ (max_learning_rate - min_learning_rate) * math.exp(-t_count/decay_speed)
    x_arr = []
    y_arr = []
    t_count = 0
    i=0
    for k in range(epochs):
        for file in file_list:
            train_x= getTrainData(file)
            if i%batch_size==0 and not i==0:
                _,train_accuracy,loss,_=sess.run([optimizer, accuracy,cost,update_ema], feed_dict = {x:x_arr,y:y_arr,keep_prob:dropout_prob, iter:t_count, lr:learning_rate,tst:False})
                print("step %d, epoch %d, accuracy %g,loss %g"%(i/batch_size,k,train_accuracy,loss))
                t_count+=1  
                x_arr = []
                y_arr = []
            y_arr.append(train_y[i%10000])
            for t in train_x:
                x_arr.append(t)
            i+=1
        # break
        # if len(x_arr)>0:
            # x_arr = np.array(x_arr)
            # y_arr = np.array(y_arr)
            # learning_rate = min_learning_rate #+ (max_learning_rate - min_learning_rate) * math.exp(-t_count/decay_speed)
            # _,train_accuracy,loss,_=sess.run([optimizer, accuracy,cost,update_ema], feed_dict = {x:x_arr,y:y_arr,keep_prob:dropout_prob, iter:t_count, lr:learning_rate,tst:False})
            # print("final accuracy %g"%(train_accuracy))
            # x_arr = []
            # y_arr = []
            # t_count+=1
    file_list = listfiles(test_data_dir)
    res = []
    i=0
    x_arr = []
    for file in file_list:
        test_arr= getTrainData(file)
        if i%batch_size==0 and not i==0:
            # o = sess.run([conv_output], feed_dict = {x:test_arr,keep_prob:1, iter:t_count,tst:True})
            # print(np.shape(o))
            predict_y = y_res.eval(feed_dict = {x:x_arr,keep_prob:1, iter:t_count,tst:True})
            print(np.shape(predict_y))
            res+=list(predict_y)
            # t_count+=1  
            x_arr = []
        for t in test_arr:
            x_arr.append(t)
        i+=1
        # predict_y = y_res.eval(feed_dict = {x:test_arr,keep_prob:1, iter:t_count,tst:True})
        # res.append(predict_y[0])
    predict_y = y_res.eval(feed_dict = {x:x_arr,keep_prob:1, iter:t_count,tst:True})
    print(np.shape(predict_y))
    res+=list(predict_y)
    save_path = saver.save(sess, pardir+"/julycomp/model/cnnlstmlast.ckpt")
    writeres(res)
    
    sess.close()
    
def writeres(res):
    f=open(result_path,"w")
    for r in res:
        f.writelines(str(r)+'\n')
    f.close()
    
if __name__=="__main__":
   convlstm() 
    
    






