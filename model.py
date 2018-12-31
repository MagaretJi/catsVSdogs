import tensorflow as tf
import tensorflow.contrib.layers as layers

weights={
    'w1':tf.Variable(tf.truncated_normal([3,3,3,16],stddev=0.1)),
    'w2':tf.Variable(tf.truncated_normal([3,3,16,16],stddev=0.1)),
    'wc2':tf.Variable(tf.truncated_normal([128,128],stddev=0.1)),
    'out':tf.Variable(tf.truncated_normal([128,2],stddev=0.1))
}
biases={
    'b1':tf.Variable(tf.random_normal([16])),
    'b2':tf.Variable(tf.random_normal([16])),
    'bc1':tf.Variable(tf.random_normal([128])),
    'bc2':tf.Variable(tf.random_normal([128])),
    'out':tf.Variable(tf.random_normal([2]))
}

def conv2d(x, W,strides=[1,1,1,1]):
    return tf.nn.conv2d(x, W, strides, padding='SAME')

def max_pool_2x2(x,ksize=[1,2,2,1],strides=[1,2,2,1]):
    return tf.nn.max_pool(x, ksize,strides, padding='SAME')

def norm(x,lsize,bias=1.0):
    return tf.nn.lrn(x,lsize,bias,alpha=0.001/9.0,beta=0.75)

def net(images):
    #第一层卷积层
    conv1=conv2d(images,weights['w1'],strides=[1,1,1,1])+biases['b1']
    conv1=tf.nn.relu(conv1)
    #池化
    pool1=max_pool_2x2(conv1)
    #n标准化
    norm1=norm(pool1,lsize=4,bias=1.0)

    #第二层卷积
    conv2=conv2d(norm1,weights['w2'],strides=[1,3,3,1])+biases['b2']
    conv2=tf.nn.relu(conv2)

    pool2=max_pool_2x2(conv2)
    norm2=norm(pool2,lsize=4)

    #全连接1
    reshaped=layers.flatten(norm2)
    nodes=reshaped.get_shape()[1].value
    weights['wc1']=tf.Variable(tf.truncated_normal([nodes,128],stddev=0.1))
    fc1=tf.nn.relu(tf.matmul(reshaped,weights['wc1'])+biases['bc1'])

    #全连接2
    fc2=tf.nn.relu(tf.matmul(fc1,weights['wc2'])+biases['bc2'])

    #softmax
    softmax_linear=tf.matmul(fc2,weights['out']+biases['out'])

    return softmax_linear

def losses(logits,labels):
    with tf.variable_scope('loss'):
        cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels)
        loss=tf.reduce_mean(cross_entropy)

    return loss

def evaluation(logits,labels):
    with tf.variable_scope('accuracy'):
        correct=tf.nn.in_top_k(logits,labels,1)
        correct=tf.cast(correct,tf.float16)
        accuracy=tf.reduce_mean(correct)

    return accuracy

