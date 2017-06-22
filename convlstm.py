from tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
pic_size = 101

conv1_kernal_size = 5
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
strides_1 = 2
strides_2 = 1

num_layers = 1

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

def conv(x_img):
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
    
    return h_fc1_drop
    
def lstm(x,category):
    list = x.getshape().as_list()
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias = 1)
    # lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob = keep_prob)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell], num_layers)
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
    w = weight_variable([list[1], category])
    b = bias_variable([category])
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
    
    x = tf.placeholder(tf.float32,[None,pic_size*pic_size*pic_length*channels])
    y = tf.placeholder(tf.float32,shape = [None, category])
    x_img = tf.reshape(x,[-1, pic_size, pic_size*pic_length, channels])
    conv_output = conv(x_img)
    pred = lstm(conv_output,category)
    y_pred = tf.reduce_mean(pred, reduction_indices = [1])
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(y_pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    






