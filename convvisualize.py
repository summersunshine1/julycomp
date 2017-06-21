import tensorflow as tf
from data_analyse import *
from convModel import *
import numpy as np

def convtest():
    train_y, category,arr= get_y()
    x = tf.placeholder(tf.float32,[None,pic_size*pic_size*pic_length*channels])
    y = tf.placeholder(tf.float32,shape = [None, category])
    x_img = tf.reshape(x,[-1, pic_size, pic_size*pic_length, channels])
    w_conv1 = weight_variable([101, 101, channels,1])
    h_conv1 = conv2d(x_img, w_conv1)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    file_list = listfiles(train_data_dir)
    i=0
    for file in file_list:
        train_x,_ = getTrainData(file)
        img = sess.run(h_conv1, feed_dict = {x:[train_x]})
        print_tensor(h_conv1)
        img = np.squeeze(img,axis=0)
        img = img.reshape(img.shape[:2])
        plt.imshow(img)
        plt.title(arr[i])
        plt.show()
        i+=1
    sess.close()
        
convtest()