import tensorflow as tf
import numpy as np
from convlstm2 import *

test_data = pardir +'/julycomp/data'
conv1_num = 16

def getTrainData(path):
    x_arr = []
    with open(path,'r') as f:
        for line in f:
            arr = line.split(',')
            x_temp = arr[2].split()
            x = [float(t) for t in x_temp]
            for i in range(pic_length):
                temparr = []
                for j in range(channels):
                    index = i*channels+j
                    temp = x[index*101*101:(index+1)*101*101]
                    temp = np.reshape(temp,[101*101,1])
                    temparr.append(temp)
                arr = temparr[0]
                for i in range(1,4):
                    arr = np.hstack((arr, temparr[i]))
                arr = np.reshape(arr,[101*101*4])
                x_arr.append(arr)           
    x_arr = np.array(x_arr)
    return x_arr
    
def batch_norm(input,tst):
    return tf.contrib.layers.batch_norm(input, center=True, scale=True, is_training=tst,updates_collections = None)

def convtest():
    # train_y, category,arr= get_y()
    x = tf.placeholder(tf.float32,[None,pic_size*pic_size*channels])
    # y = tf.placeholder(tf.float32,shape = [None, category])
    x_img = tf.reshape(x,[-1, pic_size, pic_size, channels])
    w_conv1 = weight_variable([conv1_kernal_size, conv1_kernal_size, channels, conv1_num])
    b_conv1 = bias_variable([conv1_num])
    h_conv1 = conv2d(x_img, w_conv1,strides_1)
    tst = tf.placeholder(tf.bool)
    iter = tf.placeholder(tf.int32)
    x1bn = batch_norm(h_conv1, tst)

    # h_conv1 = tf.nn.relu(h_conv1)
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    # print(test_data)
    file_list = listfiles(test_data)
    # print(file_list)
    k=0
    for file in file_list:
        train_x = getTrainData(file)
        if k<4:
            init_img, img, bn= sess.run([x_img,h_conv1,x1bn], feed_dict = {x:train_x,tst:True,iter:1})
        else:
            init_img, img, bn= sess.run([x_img,h_conv1,x1bn], feed_dict = {x:train_x,tst:False,iter:1})
        # print(np.shape(init_img))
        # print(np.shape(img))
        for i in range(1):
            for j in range(channels):
                plt.subplot(3, conv1_num, j+1)               
                plt.imshow(init_img[i,:,:,j])
            # plt.subplot(1,2,2)
            for j in range(conv1_num):
                plt.subplot(3, conv1_num, conv1_num+j+1) 
                print(img[i,:,:,j][0][0])
                plt.imshow(img[i,:,:,j])
                break
                
                
            for j in range(conv1_num):
                plt.subplot(3, conv1_num, conv1_num*2+j+1) 
                print(bn[i,:,:,j][0][0])
                plt.imshow(bn[i,:,:,j])
                break
            break
        break
            # plt.show()
        k+=1
        # img = np.squeeze(img,axis=0)
        # img = img.reshape(img.shape[:2])
        
    sess.close()

if __name__=="__main__":    
    convtest()