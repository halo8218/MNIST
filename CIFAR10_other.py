import tensorflow as tf
import numpy as np
import utils
import matplotlib
matplotlib.use('Agg')
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt

from tensorflow.keras.datasets.cifar10 import load_data

(x_train, y_train), (x_test, y_test) = load_data()
y_train_one_hot = np.eye(10)[y_train.reshape(-1)]
y_test_one_hot = np.eye(10)[y_test.reshape(-1)]

'''''''''
Build base model
'''''''''
X = tf.placeholder(tf.float32, [None, 32, 32, 3])
X_small = tf.placeholder(tf.float32, [None, 32, 32, 3])
X = tf.placeholder(tf.float32, [None, 32, 32, 3])
X_small = tf.placeholder(tf.float32, [None, 32, 32, 3])
#tmp_comp_1 = tf.placeholder(tf.float32, [32, 32, 64])
#tmp_comp_2 = tf.placeholder(tf.float32, [16, 16, 64])
#tmp_comp_3 = tf.placeholder(tf.float32, [8, 8, 128])
#tmp_comp_4 = tf.placeholder(tf.float32, [8, 8, 128])
#tmp_comp_5 = tf.placeholder(tf.float32, [8, 8, 128])
#tmp_comp_6 = tf.placeholder(tf.float32, [384])
#comp_map_place_holder_list = [tmp_comp_1, tmp_comp_2, tmp_comp_3, tmp_comp_4, tmp_comp_5, tmp_comp_6]
tmp_comp_1 = tf.placeholder(tf.float32, [32, 32, 64])
tmp_comp_2 = tf.placeholder(tf.float32, [32, 32, 64])
tmp_comp_3 = tf.placeholder(tf.float32, [16, 16, 128])
tmp_comp_4 = tf.placeholder(tf.float32, [16, 16, 128])
tmp_comp_5 = tf.placeholder(tf.float32, [8, 8, 256])
tmp_comp_6 = tf.placeholder(tf.float32, [8, 8, 256])
tmp_comp_7 = tf.placeholder(tf.float32, [4, 4, 512])
tmp_comp_8 = tf.placeholder(tf.float32, [4, 4, 512])
tmp_comp_9 = tf.placeholder(tf.float32, [512])
comp_map_place_holder_list = [tmp_comp_1, tmp_comp_2, tmp_comp_3, tmp_comp_4, tmp_comp_5, tmp_comp_6, tmp_comp_7, tmp_comp_8, tmp_comp_9]

Y = tf.placeholder(tf.float32, [None, 10])
Y_small = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
is_first = tf.placeholder(tf.bool)
is_training = tf.placeholder(tf.bool)

lr = tf.placeholder(tf.float32)

rate_place_holder = tf.placeholder(tf.float32, [])

W_conv1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 3, 64], stddev=5e-2))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[64]))
h_conv1_prev = tf.nn.conv2d(X, W_conv1, strides=[1, 1, 1, 1], padding='SAME') 
h_conv1_prev = tf.layers.batch_normalization(h_conv1_prev + b_conv1, training=is_training, name='h1')
h_conv1 = tf.nn.relu(h_conv1_prev)

W_conv2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 64, 64], stddev=5e-2))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
h_conv2_prev = tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
h_conv2_prev = tf.layers.batch_normalization(h_conv2_prev + b_conv2, training=is_training, name='h2')
h_conv2 = tf.nn.relu(h_conv2_prev)

h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

W_conv3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], stddev=5e-2))
b_conv3 = tf.Variable(tf.constant(0.1, shape=[128]))
h_conv3_prev = tf.nn.conv2d(h_pool2, W_conv3, strides=[1, 1, 1, 1], padding='SAME')
h_conv3_prev = tf.layers.batch_normalization(h_conv3_prev + b_conv3, training=is_training, name='h3')
h_conv3 = tf.nn.relu(h_conv3_prev)

W_conv4 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 128], stddev=5e-2))
b_conv4 = tf.Variable(tf.constant(0.1, shape=[128])) 
h_conv4_prev = tf.nn.conv2d(h_conv3, W_conv4, strides=[1, 1, 1, 1], padding='SAME') 
h_conv4_prev = tf.layers.batch_normalization(h_conv4_prev + b_conv4, training=is_training, name='h4')
h_conv4 = tf.nn.relu(h_conv4_prev)

h_pool4 = tf.nn.max_pool(h_conv4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

W_conv5 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 256], stddev=5e-2))
b_conv5 = tf.Variable(tf.constant(0.1, shape=[256]))
h_conv5_prev = tf.nn.conv2d(h_pool4, W_conv5, strides=[1, 1, 1, 1], padding='SAME') 
h_conv5_prev = tf.layers.batch_normalization(h_conv5_prev + b_conv5, training=is_training, name='h5')
h_conv5 = tf.nn.relu(h_conv5_prev)

W_conv6 = tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 256], stddev=5e-2))
b_conv6 = tf.Variable(tf.constant(0.1, shape=[256]))
h_conv6_prev = tf.nn.conv2d(h_conv5, W_conv6, strides=[1, 1, 1, 1], padding='SAME') 
h_conv6_prev = tf.layers.batch_normalization(h_conv6_prev + b_conv6, training=is_training, name='h6')
h_conv6 = tf.nn.relu(h_conv6_prev)

h_pool6 = tf.nn.max_pool(h_conv6, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

W_conv7 = tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 512], stddev=5e-2))
b_conv7 = tf.Variable(tf.constant(0.1, shape=[512]))
h_conv7_prev = tf.nn.conv2d(h_pool6, W_conv7, strides=[1, 1, 1, 1], padding='SAME') 
h_conv7_prev = tf.layers.batch_normalization(h_conv7_prev + b_conv7, training=is_training, name='h7')
h_conv7 = tf.nn.relu(h_conv7_prev)

W_conv8 = tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 512], stddev=5e-2))
b_conv8 = tf.Variable(tf.constant(0.1, shape=[512]))
h_conv8_prev = tf.nn.conv2d(h_conv7, W_conv8, strides=[1, 1, 1, 1], padding='SAME') 
h_conv8_prev = tf.layers.batch_normalization(h_conv8_prev + b_conv8, training=is_training, name='h8')
h_conv8 = tf.nn.relu(h_conv8_prev)

W_fc1 = tf.Variable(tf.truncated_normal(shape=[4 * 4 * 512, 512], stddev=5e-2))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[512]))

h_conv8_flat = tf.reshape(h_conv8, [-1, 4*4*512])
h_fc1_prev = tf.matmul(h_conv8_flat, W_fc1)
h_fc1_prev = tf.layers.batch_normalization(h_fc1_prev + b_fc1, training=is_training, name='fc1')
h_fc1 = tf.nn.relu(h_fc1_prev)

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob) 

W_fc2 = tf.Variable(tf.truncated_normal(shape=[512, 10], stddev=5e-2))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
model = tf.matmul(h_fc1_drop,W_fc2) + b_fc2

coeff = 0.01
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
#cost = cost + coeff*tf.nn.l2_loss(W_conv1) + coeff*tf.nn.l2_loss(W_conv2) + coeff*tf.nn.l2_loss(W_conv3) + coeff*tf.nn.l2_loss(W_conv4) + coeff*tf.nn.l2_loss(W_conv5)+ coeff*tf.nn.l2_loss(W_conv6)+ coeff*tf.nn.l2_loss(W_conv7)+ coeff*tf.nn.l2_loss(W_conv8) + coeff*tf.nn.l2_loss(W_fc1) + coeff*tf.nn.l2_loss(W_fc2)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(lr).minimize(cost)
#optimizer = tf.train.GradientDescentOptimizer(lr).minimize(cost)

'''''''''
Make the node generating adv' examples
'''''''''
grad = tf.gradients(cost, X)
tmp_ep = [0, 1, 2, 3, 4, 6, 8]
#tmp_ep = [0, 8]
epsilon = np.array(tmp_ep) / 1.0
xadv = [tf.stop_gradient(X + e*tf.sign(grad)) for e in epsilon]
xadv = [tf.clip_by_value(adv, 0., 255.) for adv in xadv]
xadv = [tf.reshape(adv,[-1,32, 32, 3]) for adv in xadv]
yadv = Y
'''''''''
Build replica model for comparing
'''''''''
h_conv1_comp_prev = tf.nn.conv2d(X_small, W_conv1, strides=[1, 1, 1, 1], padding='SAME') 
h_conv1_comp_prev = tf.layers.batch_normalization(h_conv1_comp_prev + b_conv1, training=is_training, reuse=True, name='h1')
h_conv1_comp = tf.nn.relu(h_conv1_comp_prev)

h_conv2_comp_prev = tf.nn.conv2d(h_conv1_comp, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
h_conv2_comp_prev = tf.layers.batch_normalization(h_conv2_comp_prev + b_conv2, training=is_training, reuse=True, name='h2')
h_conv2_comp = tf.nn.relu(h_conv2_comp_prev)

h_pool2_comp = tf.nn.max_pool(h_conv2_comp, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

h_conv3_comp_prev = tf.nn.conv2d(h_pool2_comp, W_conv3, strides=[1, 1, 1, 1], padding='SAME')
h_conv3_comp_prev = tf.layers.batch_normalization(h_conv3_comp_prev + b_conv3, training=is_training, reuse=True, name='h3')
h_conv3_comp = tf.nn.relu(h_conv3_comp_prev)

h_conv4_comp_prev = tf.nn.conv2d(h_conv3_comp, W_conv4, strides=[1, 1, 1, 1], padding='SAME') 
h_conv4_comp_prev = tf.layers.batch_normalization(h_conv4_comp_prev + b_conv4, training=is_training, reuse=True, name='h4')
h_conv4_comp = tf.nn.relu(h_conv4_comp_prev)

h_pool4_comp = tf.nn.max_pool(h_conv4_comp, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

h_conv5_comp_prev = tf.nn.conv2d(h_pool4_comp, W_conv5, strides=[1, 1, 1, 1], padding='SAME') 
h_conv5_comp_prev = tf.layers.batch_normalization(h_conv5_comp_prev + b_conv5, training=is_training, reuse=True, name='h5')
h_conv5_comp = tf.nn.relu(h_conv5_comp_prev)

h_conv6_comp_prev = tf.nn.conv2d(h_conv5_comp, W_conv6, strides=[1, 1, 1, 1], padding='SAME') 
h_conv6_comp_prev = tf.layers.batch_normalization(h_conv6_comp_prev + b_conv6, training=is_training, reuse=True, name='h6')
h_conv6_comp = tf.nn.relu(h_conv6_comp_prev)

h_pool6_comp = tf.nn.max_pool(h_conv6_comp, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

h_conv7_comp_prev = tf.nn.conv2d(h_pool6_comp, W_conv7, strides=[1, 1, 1, 1], padding='SAME') 
h_conv7_comp_prev = tf.layers.batch_normalization(h_conv7_comp_prev + b_conv7, training=is_training, reuse=True, name='h7')
h_conv7_comp = tf.nn.relu(h_conv7_comp_prev)

h_conv8_comp_prev = tf.nn.conv2d(h_conv7_comp, W_conv8, strides=[1, 1, 1, 1], padding='SAME') 
h_conv8_comp_prev = tf.layers.batch_normalization(h_conv8_comp_prev + b_conv8, training=is_training, reuse=True, name='h8')
h_conv8_comp = tf.nn.relu(h_conv8_comp_prev)

h_conv8_comp_flat = tf.reshape(h_conv8_comp, [-1, 4*4*512])
h_fc1_comp_prev = tf.matmul(h_conv8_comp_flat, W_fc1)
h_fc1_comp_prev = tf.layers.batch_normalization(h_fc1_comp_prev + b_fc1, training=is_training, reuse=True, name='fc1')
h_fc1_comp = tf.nn.relu(h_fc1_comp_prev)

h_fc1_comp_drop = tf.nn.dropout(h_fc1_comp, keep_prob) 

model_comp = tf.matmul(h_fc1_comp_drop,W_fc2) + b_fc2


cost_comp = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model_comp, labels=Y_small))
grad_comp = tf.gradients(cost_comp, X_small)
xadv_small = tf.stop_gradient(X_small + 0.3*tf.sign(grad_comp))
xadv_small = tf.clip_by_value(xadv_small, 0., 1.)
xadv_small = tf.reshape(xadv_small, [-1, 32, 32, 3])


h_conv1_comp_adv_prev = tf.nn.conv2d(xadv_small, W_conv1, strides=[1, 1, 1, 1], padding='SAME') 
h_conv1_comp_adv_prev_bn = tf.layers.batch_normalization(h_conv1_comp_adv_prev + b_conv1, training=is_training, reuse=True, name='h1')
h_conv1_comp_adv = tf.nn.relu(h_conv1_comp_adv_prev_bn)

h_conv2_comp_adv_prev = tf.nn.conv2d(h_conv1_comp_adv, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
h_conv2_comp_adv_prev_bn = tf.layers.batch_normalization(h_conv2_comp_adv_prev + b_conv2, training=is_training, reuse=True, name='h2')
h_conv2_comp_adv = tf.nn.relu(h_conv2_comp_adv_prev_bn)

h_pool2_comp_adv = tf.nn.max_pool(h_conv2_comp_adv, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

h_conv3_comp_adv_prev = tf.nn.conv2d(h_pool2_comp_adv, W_conv3, strides=[1, 1, 1, 1], padding='SAME')
h_conv3_comp_adv_prev_bn = tf.layers.batch_normalization(h_conv3_comp_adv_prev + b_conv3, training=is_training, reuse=True, name='h3')
h_conv3_comp_adv = tf.nn.relu(h_conv3_comp_adv_prev_bn)

h_conv4_comp_adv_prev = tf.nn.conv2d(h_conv3_comp_adv, W_conv4, strides=[1, 1, 1, 1], padding='SAME') 
h_conv4_comp_adv_prev_bn = tf.layers.batch_normalization(h_conv4_comp_adv_prev + b_conv4, training=is_training, reuse=True, name='h4')
h_conv4_comp_adv = tf.nn.relu(h_conv4_comp_adv_prev_bn)

h_pool4_comp_adv = tf.nn.max_pool(h_conv4_comp_adv, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

h_conv5_comp_adv_prev = tf.nn.conv2d(h_pool4_comp_adv, W_conv5, strides=[1, 1, 1, 1], padding='SAME') 
h_conv5_comp_adv_prev_bn = tf.layers.batch_normalization(h_conv5_comp_adv_prev + b_conv5, training=is_training, reuse=True, name='h5')
h_conv5_comp_adv = tf.nn.relu(h_conv5_comp_adv_prev_bn)

h_conv6_comp_adv_prev = tf.nn.conv2d(h_conv5_comp_adv, W_conv6, strides=[1, 1, 1, 1], padding='SAME') 
h_conv6_comp_adv_prev_bn = tf.layers.batch_normalization(h_conv6_comp_adv_prev + b_conv6, training=is_training, reuse=True, name='h6')
h_conv6_comp_adv = tf.nn.relu(h_conv6_comp_adv_prev_bn)

h_pool6_comp_adv = tf.nn.max_pool(h_conv6_comp_adv, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

h_conv7_comp_adv_prev = tf.nn.conv2d(h_pool6_comp_adv, W_conv7, strides=[1, 1, 1, 1], padding='SAME') 
h_conv7_comp_adv_prev_bn = tf.layers.batch_normalization(h_conv7_comp_adv_prev + b_conv7, training=is_training, reuse=True, name='h7')
h_conv7_comp_adv = tf.nn.relu(h_conv7_comp_adv_prev_bn)

h_conv8_comp_adv_prev = tf.nn.conv2d(h_conv7_comp_adv, W_conv8, strides=[1, 1, 1, 1], padding='SAME') 
h_conv8_comp_adv_prev_bn = tf.layers.batch_normalization(h_conv8_comp_adv_prev + b_conv8, training=is_training, reuse=True, name='h8')
h_conv8_comp_adv = tf.nn.relu(h_conv8_comp_adv_prev_bn)

h_conv8_comp_adv_flat = tf.reshape(h_conv8_comp_adv, [-1, 4*4*512])
h_fc1_comp_adv_prev = tf.matmul(h_conv8_comp_adv_flat, W_fc1)
h_fc1_comp_adv_prev_bn = tf.layers.batch_normalization(h_fc1_comp_adv_prev + b_fc1, training=is_training, reuse=True, name='fc1')
h_fc1_comp_adv = tf.nn.relu(h_fc1_comp_adv_prev_bn)

h_fc1_comp_adv_drop = tf.nn.dropout(h_fc1_comp_adv, keep_prob) 

model_comp_adv = tf.matmul(h_fc1_comp_adv_drop,W_fc2) + b_fc2

'''''''''
Magnitude Based Activation Pruning Model
'''''''''
def MBAP(pruning_rate_per_layer, is_first):
    _, mask_1 = utils.prune_conv_feature(h_conv1, pruning_rate_per_layer)
    h_conv1_ap = tf.cond(is_first, lambda: h_conv1 * mask_1, lambda: h_conv1)
    h_conv2_ap_prev = tf.nn.conv2d(h_conv1_ap, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
    h_conv2_ap_prev = tf.layers.batch_normalization(h_conv2_ap_prev + b_conv2, training=is_training, reuse=True, name='h2')
    h_conv2_ap = tf.nn.relu(h_conv2_ap_prev)
    h_conv2_ap,_ = utils.prune_conv_feature(h_conv2_ap, pruning_rate_per_layer)
    h_pool2_ap = tf.nn.max_pool(h_conv2_ap, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    h_conv3_ap_prev = tf.nn.conv2d(h_pool2_ap, W_conv3, strides=[1, 1, 1, 1], padding='SAME')
    h_conv3_ap_prev = tf.layers.batch_normalization(h_conv3_ap_prev + b_conv3, training=is_training, reuse=True, name='h3')
    h_conv3_ap = tf.nn.relu(h_conv3_ap_prev)
    h_conv3_ap,_ = utils.prune_conv_feature(h_conv3_ap, pruning_rate_per_layer)
    h_conv4_ap_prev = tf.nn.conv2d(h_conv3_ap, W_conv4, strides=[1, 1, 1, 1], padding='SAME') 
    h_conv4_ap_prev = tf.layers.batch_normalization(h_conv4_ap_prev + b_conv4, training=is_training, reuse=True, name='h4')
    h_conv4_ap = tf.nn.relu(h_conv4_ap_prev)
    h_conv4_ap,_ = utils.prune_conv_feature(h_conv4_ap, pruning_rate_per_layer)
    h_pool4_ap = tf.nn.max_pool(h_conv4_ap, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    h_conv5_ap_prev = tf.nn.conv2d(h_pool4_ap, W_conv5, strides=[1, 1, 1, 1], padding='SAME') 
    h_conv5_ap_prev = tf.layers.batch_normalization(h_conv5_ap_prev + b_conv5, training=is_training, reuse=True, name='h5')
    h_conv5_ap = tf.nn.relu(h_conv5_ap_prev)
    h_conv5_ap,_ = utils.prune_conv_feature(h_conv5_ap, pruning_rate_per_layer)
    h_conv6_ap_prev = tf.nn.conv2d(h_conv5_ap, W_conv6, strides=[1, 1, 1, 1], padding='SAME') 
    h_conv6_ap_prev = tf.layers.batch_normalization(h_conv6_ap_prev + b_conv6, training=is_training, reuse=True, name='h6')
    h_conv6_ap = tf.nn.relu(h_conv6_ap_prev)
    h_conv6_ap,_ = utils.prune_conv_feature(h_conv6_ap, pruning_rate_per_layer)
    h_pool6_ap = tf.nn.max_pool(h_conv6_ap, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    h_conv7_ap_prev = tf.nn.conv2d(h_pool6_ap, W_conv7, strides=[1, 1, 1, 1], padding='SAME') 
    h_conv7_ap_prev = tf.layers.batch_normalization(h_conv7_ap_prev + b_conv7, training=is_training, reuse=True, name='h7')
    h_conv7_ap = tf.nn.relu(h_conv7_ap_prev)
    h_conv7_ap,_ = utils.prune_conv_feature(h_conv7_ap, pruning_rate_per_layer)
    h_conv8_ap_prev = tf.nn.conv2d(h_conv7_ap, W_conv8, strides=[1, 1, 1, 1], padding='SAME') 
    h_conv8_ap_prev = tf.layers.batch_normalization(h_conv8_ap_prev + b_conv8, training=is_training, reuse=True, name='h8')
    h_conv8_ap = tf.nn.relu(h_conv8_ap_prev)
    h_conv8_ap,_ = utils.prune_conv_feature(h_conv8_ap, pruning_rate_per_layer)
    h_conv8_ap_flat = tf.reshape(h_conv8_ap, [-1, 4*4*512])
    h_fc1_ap_prev = tf.matmul(h_conv8_ap_flat, W_fc1)
    h_fc1_ap_prev = tf.layers.batch_normalization(h_fc1_ap_prev + b_fc1, training=is_training, reuse=True, name='fc1')
    h_fc1_ap = tf.nn.relu(h_fc1_ap_prev)
    h_fc1_ap,_ = utils.prune(h_fc1_ap, pruning_rate_per_layer)
    h_fc1_ap_drop = tf.nn.dropout(h_fc1_ap, keep_prob) 
    
    model_ap = tf.matmul(h_fc1_ap_drop,W_fc2) + b_fc2

    return model_ap

'''''''''
Adversarial Feature Drop Model
'''''''''
def compare():
    diff = tf.norm(model_comp - model_comp_adv, 2) 
    cand = [h_conv1_comp_adv_prev, h_conv2_comp_adv_prev, h_conv3_comp_adv_prev, h_conv4_comp_adv_prev, h_conv5_comp_adv_prev, h_conv6_comp_adv_prev, h_conv7_comp_adv_prev, h_conv8_comp_adv_prev,  h_fc1_comp_adv_prev]
    grad_on_diff = tf.gradients(diff, cand) 
    comp_map = [tf.reduce_sum(tf.abs(grad_on_diff[i]), axis=0) for i in range(len(cand))]
    return comp_map

def AFD(pruning_rate_per_layer, is_first, comp_map):
    adv_feat = comp_map
    mask = [utils.mask_map(adv_feat[i], pruning_rate_per_layer) for i in range(len(adv_feat))]

    h_conv1_af_prev = tf.nn.conv2d(X, W_conv1, strides=[1, 1, 1, 1], padding='SAME') 
    h_conv1_af_prev = tf.cond(is_first, lambda: h_conv1_af_prev * mask[0], lambda: h_conv1_af_prev)
    h_conv1_af_prev = tf.layers.batch_normalization(h_conv1_af_prev + b_conv1, training=is_training, reuse=True, name='h1')
    h_conv1_af = tf.nn.relu(h_conv1_af_prev)
    h_conv2_af_prev = tf.nn.conv2d(h_conv1_af, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
    h_conv2_af_prev = tf.layers.batch_normalization(h_conv2_af_prev * mask[1] + b_conv2, training=is_training, reuse=True, name='h2')
    h_conv2_af = tf.nn.relu(h_conv2_af_prev)
    h_pool2_af = tf.nn.max_pool(h_conv2_af, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    h_conv3_af_prev = tf.nn.conv2d(h_pool2_af, W_conv3, strides=[1, 1, 1, 1], padding='SAME')
    h_conv3_af_prev = tf.layers.batch_normalization(h_conv3_af_prev * mask[2] + b_conv3, training=is_training, reuse=True, name='h3')
    h_conv3_af = tf.nn.relu(h_conv3_af_prev)
    h_conv4_af_prev = tf.nn.conv2d(h_conv3_af, W_conv4, strides=[1, 1, 1, 1], padding='SAME') 
    h_conv4_af_prev = tf.layers.batch_normalization(h_conv4_af_prev * mask[3] + b_conv4, training=is_training, reuse=True, name='h4')
    h_conv4_af = tf.nn.relu(h_conv4_af_prev)
    h_pool4_af = tf.nn.max_pool(h_conv4_af, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    h_conv5_af_prev = tf.nn.conv2d(h_pool4_af, W_conv5, strides=[1, 1, 1, 1], padding='SAME') 
    h_conv5_af_prev = tf.layers.batch_normalization(h_conv5_af_prev * mask[4] + b_conv5, training=is_training, reuse=True, name='h5')
    h_conv5_af = tf.nn.relu(h_conv5_af_prev)
    h_conv6_af_prev = tf.nn.conv2d(h_conv5_af, W_conv6, strides=[1, 1, 1, 1], padding='SAME') 
    h_conv6_af_prev = tf.layers.batch_normalization(h_conv6_af_prev * mask[5] + b_conv6, training=is_training, reuse=True, name='h6')
    h_conv6_af = tf.nn.relu(h_conv6_af_prev)
    h_pool6_af = tf.nn.max_pool(h_conv6_af, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    h_conv7_af_prev = tf.nn.conv2d(h_pool6_af, W_conv7, strides=[1, 1, 1, 1], padding='SAME') 
    h_conv7_af_prev = tf.layers.batch_normalization(h_conv7_af_prev * mask[6] + b_conv7, training=is_training, reuse=True, name='h7')
    h_conv7_af = tf.nn.relu(h_conv7_af_prev)
    h_conv8_af_prev = tf.nn.conv2d(h_conv7_af, W_conv8, strides=[1, 1, 1, 1], padding='SAME') 
    h_conv8_af_prev = tf.layers.batch_normalization(h_conv8_af_prev * mask[7] + b_conv8, training=is_training, reuse=True, name='h8')
    h_conv8_af = tf.nn.relu(h_conv8_af_prev)
    h_conv8_af_flat = tf.reshape(h_conv8_af, [-1, 4*4*512])
    h_fc1_af_prev = tf.matmul(h_conv8_af_flat, W_fc1)
    h_fc1_af_prev = tf.layers.batch_normalization(h_fc1_af_prev * mask[8] + b_fc1, training=is_training, reuse=True, name='fc1')
    h_fc1_af = tf.nn.relu(h_fc1_af_prev)
    h_fc1_af_drop = tf.nn.dropout(h_fc1_af, keep_prob) 
    
    model_af = tf.matmul(h_fc1_af_drop,W_fc2) + b_fc2
    
    return model_af

'''''''''
Random Feature Drop Model
'''''''''
def RFD(pruning_rate_per_layer, is_first):
    ran_feat = utils.random_map(h_conv1, h_conv2, h_conv3, h_conv4, h_conv5, h_conv6, h_conv7, h_conv8, h_fc1)
    mask = [utils.mask_map(ran_feat[i], pruning_rate_per_layer) for i in range(len(ran_feat))]

    h_conv1_rd_prev = tf.nn.conv2d(X, W_conv1, strides=[1, 1, 1, 1], padding='SAME') 
    h_conv1_rd_prev = tf.cond(is_first, lambda: h_conv1_rd_prev * mask[0], lambda: h_conv1_rd_prev)
    h_conv1_rd_prev = tf.layers.batch_normalization(h_conv1_rd_prev + b_conv1, training=is_training, reuse=True, name='h1')
    h_conv1_rd = tf.nn.relu(h_conv1_rd_prev)
    h_conv2_rd_prev = tf.nn.conv2d(h_conv1_rd, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
    h_conv2_rd_prev = tf.layers.batch_normalization(h_conv2_rd_prev * mask[1] + b_conv2, training=is_training, reuse=True, name='h2')
    h_conv2_rd = tf.nn.relu(h_conv2_rd_prev)
    h_pool2_rd = tf.nn.max_pool(h_conv2_rd, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    h_conv3_rd_prev = tf.nn.conv2d(h_pool2_rd, W_conv3, strides=[1, 1, 1, 1], padding='SAME')
    h_conv3_rd_prev = tf.layers.batch_normalization(h_conv3_rd_prev * mask[2] + b_conv3, training=is_training, reuse=True, name='h3')
    h_conv3_rd = tf.nn.relu(h_conv3_rd_prev)
    h_conv4_rd_prev = tf.nn.conv2d(h_conv3_rd, W_conv4, strides=[1, 1, 1, 1], padding='SAME') 
    h_conv4_rd_prev = tf.layers.batch_normalization(h_conv4_rd_prev * mask[3] + b_conv4, training=is_training, reuse=True, name='h4')
    h_conv4_rd = tf.nn.relu(h_conv4_rd_prev)
    h_pool4_rd = tf.nn.max_pool(h_conv4_rd, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    h_conv5_rd_prev = tf.nn.conv2d(h_pool4_rd, W_conv5, strides=[1, 1, 1, 1], padding='SAME') 
    h_conv5_rd_prev = tf.layers.batch_normalization(h_conv5_rd_prev * mask[4] + b_conv5, training=is_training, reuse=True, name='h5')
    h_conv5_rd = tf.nn.relu(h_conv5_rd_prev)
    h_conv6_rd_prev = tf.nn.conv2d(h_conv5_rd, W_conv6, strides=[1, 1, 1, 1], padding='SAME') 
    h_conv6_rd_prev = tf.layers.batch_normalization(h_conv6_rd_prev * mask[5] + b_conv6, training=is_training, reuse=True, name='h6')
    h_conv6_rd = tf.nn.relu(h_conv6_rd_prev)
    h_pool6_rd = tf.nn.max_pool(h_conv6_rd, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    h_conv7_rd_prev = tf.nn.conv2d(h_pool6_rd, W_conv7, strides=[1, 1, 1, 1], padding='SAME') 
    h_conv7_rd_prev = tf.layers.batch_normalization(h_conv7_rd_prev * mask[6] + b_conv7, training=is_training, reuse=True, name='h7')
    h_conv7_rd = tf.nn.relu(h_conv7_rd_prev)
    h_conv8_rd_prev = tf.nn.conv2d(h_conv7_rd, W_conv8, strides=[1, 1, 1, 1], padding='SAME') 
    h_conv8_rd_prev = tf.layers.batch_normalization(h_conv8_rd_prev * mask[7] + b_conv8, training=is_training, reuse=True, name='h8')
    h_conv8_rd = tf.nn.relu(h_conv8_rd_prev)
    h_conv8_rd_flat = tf.reshape(h_conv8_rd, [-1, 4*4*512])
    h_fc1_rd_prev = tf.matmul(h_conv8_rd_flat, W_fc1)
    h_fc1_rd_prev = tf.layers.batch_normalization(h_fc1_rd_prev * mask[8] + b_fc1, training=is_training, reuse=True, name='fc1')
    h_fc1_rd = tf.nn.relu(h_fc1_rd_prev)
    h_fc1_rd_drop = tf.nn.dropout(h_fc1_rd, keep_prob) 
    
    model_rd = tf.matmul(h_fc1_rd_drop,W_fc2) + b_fc2

    return model_rd

'''''''''
Iterative Adversarial Feature Drop Model
'''''''''
def _body(pruning_rate_per_step, mask):
    '''''''''
    Build replica model for iterative comparing
    '''''''''
    h_conv1_comp_prev = tf.nn.conv2d(X_small, W_conv1, strides=[1, 1, 1, 1], padding='SAME') 
    h_conv1_comp_prev = tf.layers.batch_normalization(h_conv1_comp_prev + b_conv1, training=is_training, reuse=True, name='h1')
    h_conv1_comp = tf.nn.relu(h_conv1_comp_prev)
    
    h_conv2_comp_prev = tf.nn.conv2d(h_conv1_comp, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
    h_conv2_comp_prev = tf.layers.batch_normalization(h_conv2_comp_prev + b_conv2, training=is_training, reuse=True, name='h2')
    h_conv2_comp = tf.nn.relu(h_conv2_comp_prev)
    
    h_pool2_comp = tf.nn.max_pool(h_conv2_comp, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    h_conv3_comp_prev = tf.nn.conv2d(h_pool2_comp, W_conv3, strides=[1, 1, 1, 1], padding='SAME')
    h_conv3_comp_prev = tf.layers.batch_normalization(h_conv3_comp_prev + b_conv3, training=is_training, reuse=True, name='h3')
    h_conv3_comp = tf.nn.relu(h_conv3_comp_prev)
    
    h_conv4_comp_prev = tf.nn.conv2d(h_conv3_comp, W_conv4, strides=[1, 1, 1, 1], padding='SAME') 
    h_conv4_comp_prev = tf.layers.batch_normalization(h_conv4_comp_prev + b_conv4, training=is_training, reuse=True, name='h4')
    h_conv4_comp = tf.nn.relu(h_conv4_comp_prev)
    
    h_pool4_comp = tf.nn.max_pool(h_conv4_comp, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    h_conv5_comp_prev = tf.nn.conv2d(h_pool4_comp, W_conv5, strides=[1, 1, 1, 1], padding='SAME') 
    h_conv5_comp_prev = tf.layers.batch_normalization(h_conv5_comp_prev + b_conv5, training=is_training, reuse=True, name='h5')
    h_conv5_comp = tf.nn.relu(h_conv5_comp_prev)
    
    h_conv6_comp_prev = tf.nn.conv2d(h_conv5_comp, W_conv6, strides=[1, 1, 1, 1], padding='SAME') 
    h_conv6_comp_prev = tf.layers.batch_normalization(h_conv6_comp_prev + b_conv6, training=is_training, reuse=True, name='h6')
    h_conv6_comp = tf.nn.relu(h_conv6_comp_prev)
    
    h_pool6_comp = tf.nn.max_pool(h_conv6_comp, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    h_conv7_comp_prev = tf.nn.conv2d(h_pool6_comp, W_conv7, strides=[1, 1, 1, 1], padding='SAME') 
    h_conv7_comp_prev = tf.layers.batch_normalization(h_conv7_comp_prev + b_conv7, training=is_training, reuse=True, name='h7')
    h_conv7_comp = tf.nn.relu(h_conv7_comp_prev)
    
    h_conv8_comp_prev = tf.nn.conv2d(h_conv7_comp, W_conv8, strides=[1, 1, 1, 1], padding='SAME') 
    h_conv8_comp_prev = tf.layers.batch_normalization(h_conv8_comp_prev + b_conv8, training=is_training, reuse=True, name='h8')
    h_conv8_comp = tf.nn.relu(h_conv8_comp_prev)
    
    h_conv8_comp_flat = tf.reshape(h_conv8_comp, [-1, 4*4*512])
    h_fc1_comp_prev = tf.matmul(h_conv8_comp_flat, W_fc1)
    h_fc1_comp_prev = tf.layers.batch_normalization(h_fc1_comp_prev + b_fc1, training=is_training, reuse=True, name='fc1')
    h_fc1_comp = tf.nn.relu(h_fc1_comp_prev)
    
    h_fc1_comp_drop = tf.nn.dropout(h_fc1_comp, keep_prob) 
    
    model_comp = tf.matmul(h_fc1_comp_drop,W_fc2) + b_fc2
    
    
    cost_comp = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model_comp, labels=Y_small))
    grad_comp = tf.gradients(cost_comp, X_small)
    xadv_small = tf.stop_gradient(X_small + 0.3*tf.sign(grad_comp))
    xadv_small = tf.clip_by_value(xadv_small, 0., 1.)
    xadv_small = tf.reshape(xadv_small, [-1, 32, 32, 3])
    
    
    h_conv1_comp_adv_prev = tf.nn.conv2d(xadv_small, W_conv1, strides=[1, 1, 1, 1], padding='SAME') 
    h_conv1_comp_adv_prev_bn = tf.layers.batch_normalization(h_conv1_comp_adv_prev + b_conv1, training=is_training, reuse=True, name='h1')
    h_conv1_comp_adv = tf.nn.relu(h_conv1_comp_adv_prev_bn)
    
    h_conv2_comp_adv_prev = tf.nn.conv2d(h_conv1_comp_adv, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
    h_conv2_comp_adv_prev_bn = tf.layers.batch_normalization(h_conv2_comp_adv_prev + b_conv2, training=is_training, reuse=True, name='h2')
    h_conv2_comp_adv = tf.nn.relu(h_conv2_comp_adv_prev_bn)
    
    h_pool2_comp_adv = tf.nn.max_pool(h_conv2_comp_adv, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    h_conv3_comp_adv_prev = tf.nn.conv2d(h_pool2_comp_adv, W_conv3, strides=[1, 1, 1, 1], padding='SAME')
    h_conv3_comp_adv_prev_bn = tf.layers.batch_normalization(h_conv3_comp_adv_prev + b_conv3, training=is_training, reuse=True, name='h3')
    h_conv3_comp_adv = tf.nn.relu(h_conv3_comp_adv_prev_bn)
    
    h_conv4_comp_adv_prev = tf.nn.conv2d(h_conv3_comp_adv, W_conv4, strides=[1, 1, 1, 1], padding='SAME') 
    h_conv4_comp_adv_prev_bn = tf.layers.batch_normalization(h_conv4_comp_adv_prev + b_conv4, training=is_training, reuse=True, name='h4')
    h_conv4_comp_adv = tf.nn.relu(h_conv4_comp_adv_prev_bn)
    
    h_pool4_comp_adv = tf.nn.max_pool(h_conv4_comp_adv, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    h_conv5_comp_adv_prev = tf.nn.conv2d(h_pool4_comp_adv, W_conv5, strides=[1, 1, 1, 1], padding='SAME') 
    h_conv5_comp_adv_prev_bn = tf.layers.batch_normalization(h_conv5_comp_adv_prev + b_conv5, training=is_training, reuse=True, name='h5')
    h_conv5_comp_adv = tf.nn.relu(h_conv5_comp_adv_prev_bn)
    
    h_conv6_comp_adv_prev = tf.nn.conv2d(h_conv5_comp_adv, W_conv6, strides=[1, 1, 1, 1], padding='SAME') 
    h_conv6_comp_adv_prev_bn = tf.layers.batch_normalization(h_conv6_comp_adv_prev + b_conv6, training=is_training, reuse=True, name='h6')
    h_conv6_comp_adv = tf.nn.relu(h_conv6_comp_adv_prev_bn)
    
    h_pool6_comp_adv = tf.nn.max_pool(h_conv6_comp_adv, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    h_conv7_comp_adv_prev = tf.nn.conv2d(h_pool6_comp_adv, W_conv7, strides=[1, 1, 1, 1], padding='SAME') 
    h_conv7_comp_adv_prev_bn = tf.layers.batch_normalization(h_conv7_comp_adv_prev + b_conv7, training=is_training, reuse=True, name='h7')
    h_conv7_comp_adv = tf.nn.relu(h_conv7_comp_adv_prev_bn)
    
    h_conv8_comp_adv_prev = tf.nn.conv2d(h_conv7_comp_adv, W_conv8, strides=[1, 1, 1, 1], padding='SAME') 
    h_conv8_comp_adv_prev_bn = tf.layers.batch_normalization(h_conv8_comp_adv_prev + b_conv8, training=is_training, reuse=True, name='h8')
    h_conv8_comp_adv = tf.nn.relu(h_conv8_comp_adv_prev_bn)
    
    h_conv8_comp_adv_flat = tf.reshape(h_conv8_comp_adv, [-1, 4*4*512])
    h_fc1_comp_adv_prev = tf.matmul(h_conv8_comp_adv_flat, W_fc1)
    h_fc1_comp_adv_prev_bn = tf.layers.batch_normalization(h_fc1_comp_adv_prev + b_fc1, training=is_training, reuse=True, name='fc1')
    h_fc1_comp_adv = tf.nn.relu(h_fc1_comp_adv_prev_bn)
    
    h_fc1_comp_adv_drop = tf.nn.dropout(h_fc1_comp_adv, keep_prob) 
    
    model_comp_adv = tf.matmul(h_fc1_comp_adv_drop,W_fc2) + b_fc2
    '''''''''
    '''''''''
    diff = tf.norm(model_comp - model_comp_adv, 2) 
    cand = [h_conv1_comp_adv_prev, h_conv2_comp_adv_prev, h_conv3_comp_adv_prev, h_conv4_comp_adv_prev, h_conv5_comp_adv_prev, h_conv6_comp_adv_prev, h_conv7_comp_adv_prev, h_conv8_comp_adv_prev,  h_fc1_comp_adv_prev]
    grad_on_diff = tf.gradients(diff, cand) 
    adv_feat = [tf.reduce_sum(tf.abs(grad_on_diff[i]), axis=0) * mask[i] for i in range(len(cand))]

    tmp_mask = [utils.mask_map(adv_feat[i], pruning_rate_per_step) * mask[i] for i in range(len(cand))]
    return tmp_mask

'''''''''
Model creation
'''''''''
model_ap = MBAP(rate_place_holder, is_first)
comp_map_af_op = compare()
comp_map_ia_op = _body(rate_place_holder, comp_map_place_holder_list)
model_af = AFD(rate_place_holder, is_first, comp_map_place_holder_list)
model_rd = RFD(rate_place_holder, is_first)
ones = [tf.ones(h_conv1.get_shape()[1:]), tf.ones(h_conv2.get_shape()[1:]), tf.ones(h_conv3.get_shape()[1:]), tf.ones(h_conv4.get_shape()[1:]), tf.ones(h_conv5.get_shape()[1:]), tf.ones(h_conv6.get_shape()[1:]), tf.ones(h_conv7.get_shape()[1:]), tf.ones(h_conv8.get_shape()[1:]), tf.ones(h_fc1.get_shape()[1])]
'''''''''
'''''''''
init = tf.global_variables_initializer()
sess = tf.Session()

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
is_correct_ap = tf.equal(tf.argmax(model_ap, 1), tf.argmax(Y, 1))
is_correct_af = tf.equal(tf.argmax(model_af, 1), tf.argmax(Y, 1))
is_correct_rd = tf.equal(tf.argmax(model_rd, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
accuracy_ap = tf.reduce_mean(tf.cast(is_correct_ap, tf.float32))
accuracy_af = tf.reduce_mean(tf.cast(is_correct_af, tf.float32))
accuracy_rd = tf.reduce_mean(tf.cast(is_correct_rd, tf.float32))
X_comp = x_train
Y_comp = y_train_one_hot

batch_size = 128
print('batch_size = %d'%batch_size)
total_batch_train = int(len(x_train) / batch_size)
print('total_batch_train = %d'%total_batch_train)
batch_size_test = 512
total_batch_test = int(len(x_test) / batch_size_test)
print('total_batch_test = %d'%total_batch_test)

rate_axis = range(0,100,5)

dict = {}
dict['acc_base'] = []
dict['acc_leg_ap_not'] = []
dict['acc_leg_ap_all'] = []
dict['acc_leg_af_gra_not'] = []
dict['acc_leg_af_gra_all'] = []
dict['acc_leg_ia_not'] = []
dict['acc_leg_ia_all'] = []
dict['acc_leg_rd_not'] = []
dict['acc_leg_rd_all'] = []
num_avg = 3
np_ones = np.array(sess.run(ones))
for k in range(num_avg):
    print('%d trial'%k)
    acc_base = []
    acc_leg_ap_not = []
    acc_leg_ap_all = []
    acc_leg_af_gra_not = []
    acc_leg_af_gra_all = []
    acc_leg_ia_not = []
    acc_leg_ia_all = []
    acc_leg_rd_not = []
    acc_leg_rd_all = []
    val = [0]
    init_lr = 0.001
    init_ratio = 1.0
    val_idx= 0

    sess.run(init)

    x_test_batches, y_test_batches = utils.shuffle_and_devide_into_batches(batch_size_test, x_test, y_test_one_hot)
    for epoch in range(150):
        x_train_batches, y_train_batches = utils.shuffle_and_devide_into_batches(batch_size, x_train, y_train_one_hot)
        total_cost = 0
        total_acc = 0
        total_val_acc = 0
    
        print('learning rate is :%f'%(init_lr * init_ratio))
        for i in range(total_batch_train):
            batch_xs = x_train_batches[i]
            batch_ys = y_train_batches[i]
            acc_, _, cost_val = sess.run([accuracy, optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 1.0, lr: init_lr * init_ratio, is_training: True})
            total_cost += cost_val
            total_acc += acc_
    
        print('Epoch:', '%04d' % (epoch + 1),
              'Avg. acc =', '{:.4f}'.format(total_acc / total_batch_train))
        if epoch % 5 is 0:
            for i in range(total_batch_test):
                batch_xs = x_test_batches[i]
                batch_ys = y_test_batches[i]
                val_acc = sess.run(accuracy, feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 1., is_training: False})
                total_val_acc += val_acc
            #val_acc = sess.run(accuracy,feed_dict={X: x_test_batches[0],
            #                                       keep_prob: 1.,
            #                                       Y: y_test_batches[0]})
            avg_val_acc = total_val_acc / total_batch_test
            val.append(avg_val_acc)
            val_idx += 1
            init_ratio = init_ratio * 0.5 if (val[val_idx] - val[val_idx -1] < 0.0) else init_ratio 
    print('Training completed!')
    base_acc = 0
    for t in range(total_batch_test): 
        base_acc += sess.run(accuracy,feed_dict={X: x_test_batches[t],
                                           keep_prob: 1.,
                                           is_training: False,
                                           Y: y_test_batches[t]})
    print('Acc on legitimate:', base_acc / float(total_batch_test))
    
    print('Make adversarial test sets')
    XADV = []
    YADV = y_test_batches
    for t in range(total_batch_test):
        XADV.append(sess.run(xadv, feed_dict={X: x_test_batches[t], Y: y_test_batches[t], is_training: False, keep_prob: 1.}))
    comp_map_af = np_ones

    for t in range(total_batch_train):
        comp_map_af += np.array(sess.run(comp_map_af_op,
                                feed_dict={X_small: x_train_batches[t],
                                           Y_small: y_train_batches[t],
                                           is_training: False,
                                           keep_prob: 1.}))
    for j in range(len(tmp_ep)):
        print('%.2f epsilon trial'%epsilon[j])
        acc_leg_base = 0
        for b in range(total_batch_test):
            acc_leg_base += sess.run(accuracy,
                                    feed_dict={X: XADV[b][j],
                                               Y: YADV[b],
                                               is_training: False,
                                               keep_prob: 1.})
        print('Acc on adversarial examples:', acc_leg_base / float(total_batch_test))
        for i in range(20):
            acc_base.append(acc_leg_base / float(total_batch_test))
        for i in range(20):
            acc_ap_not = 0
            for b in range(total_batch_test):
                acc_ap_not += sess.run(accuracy_ap,
                                        feed_dict={X: XADV[b][j],
                                                   Y: YADV[b],
                                                   keep_prob: 1.,
                                                   is_training: False,
                                                   is_first: False,
                                                   rate_place_holder: i*5})
            acc_leg_ap_not.append(acc_ap_not / float(total_batch_test))
            print('Acc MBAP-not on legitimate, pruning rate: %d:'%(i*5), acc_leg_ap_not[i + j*20])
        for i in range(20):
            acc_ap_all = 0
            for b in range(total_batch_test):
                acc_ap_all += sess.run(accuracy_ap,
                                        feed_dict={X: XADV[b][j],
                                                   Y: YADV[b],
                                                   is_training: False,
                                                   keep_prob: 1.,
                                                   is_first: True,
                                                   rate_place_holder: i*5})
            acc_leg_ap_all.append(acc_ap_all / float(total_batch_test))
            print('Acc MBAP-all on legitimate, pruning rate: %d:'%(i*5), acc_leg_ap_all[i + j*20])
        for i in range(20):
            acc_af_gra_not = 0
            for b in range(total_batch_test):
                acc_af_gra_not += sess.run(accuracy_af,
                                        feed_dict={X: XADV[b][j],
                                                   Y: YADV[b],
                                                   tmp_comp_1: comp_map_af[0],
                                                   tmp_comp_2: comp_map_af[1],
                                                   tmp_comp_3: comp_map_af[2],
                                                   tmp_comp_4: comp_map_af[3],
                                                   tmp_comp_5: comp_map_af[4],
                                                   tmp_comp_6: comp_map_af[5],
                                                   tmp_comp_7: comp_map_af[6],
                                                   tmp_comp_8: comp_map_af[7],
                                                   tmp_comp_9: comp_map_af[8],
                                                   keep_prob: 1.,
                                                   is_training: False,
                                                   is_first: False,
                                                   rate_place_holder: i*5})
            acc_leg_af_gra_not.append(acc_af_gra_not / float(total_batch_test))
            print('Acc AFD-not on legitimate, pruning rate: %d:'%(i*5), acc_leg_af_gra_not[i+ j*20])
        for i in range(20):
            acc_af_gra_all = 0
            for b in range(total_batch_test):
                acc_af_gra_all += sess.run(accuracy_af,
                                        feed_dict={X: XADV[b][j],
                                                   Y: YADV[b],
                                                   tmp_comp_1: comp_map_af[0],
                                                   tmp_comp_2: comp_map_af[1],
                                                   tmp_comp_3: comp_map_af[2],
                                                   tmp_comp_4: comp_map_af[3],
                                                   tmp_comp_5: comp_map_af[4],
                                                   tmp_comp_6: comp_map_af[5],
                                                   tmp_comp_7: comp_map_af[6],
                                                   tmp_comp_8: comp_map_af[7],
                                                   tmp_comp_9: comp_map_af[8],
                                                   keep_prob: 1.,
                                                   is_training: False,
                                                   is_first: True,
                                                   rate_place_holder: i*5})
            acc_leg_af_gra_all.append(acc_af_gra_all / float(total_batch_test))
            print('Acc AFD-all on legitimate, pruning rate: %d:'%(i*5), acc_leg_af_gra_all[i+ j*20]) 
        for i in range(20):
            acc_rd_not = 0
            for b in range(total_batch_test):
                acc_rd_not += sess.run(accuracy_rd,
                                        feed_dict={X: XADV[b][j],
                                                   Y: YADV[b],
                                                   keep_prob: 1.,
                                                   is_training: False,
                                                   is_first: False,
                                                   rate_place_holder: i*5})
            acc_leg_rd_not.append(acc_rd_not / float(total_batch_test))
            print('Acc RFD-not on legitimate, pruning rate: %d:'%(i*5), acc_leg_rd_not[i+ j*20]) 
        for i in range(20):
            acc_rd_all = 0
            for b in range(total_batch_test):
                acc_rd_all += sess.run(accuracy_rd,
                                        feed_dict={X: XADV[b][j],
                                                   Y: YADV[b],
                                                   keep_prob: 1.,
                                                   is_training: False,
                                                   is_first: True,
                                                   rate_place_holder: i*5})
            acc_leg_rd_all.append(acc_rd_all / float(total_batch_test))
            print('Acc RFD-all on legitimate, pruning rate: %d:'%(i*5), acc_leg_rd_all[i+ j*20]) 
        for i in range(20):
            acc_ia_not = 0
            comp_map_ia = np_ones
            for b in range(total_batch_train):
                comp_map_ia = sess.run(comp_map_ia_op,
                                        feed_dict={X_small: x_train_batches[b],
                                                   Y_small: y_train_batches[b],
                                                   keep_prob: 1.,
                                                   tmp_comp_1: comp_map_ia[0],
                                                   tmp_comp_2: comp_map_ia[1],
                                                   tmp_comp_3: comp_map_ia[2],
                                                   tmp_comp_4: comp_map_ia[3],
                                                   tmp_comp_5: comp_map_ia[4],
                                                   tmp_comp_6: comp_map_ia[5],
                                                   tmp_comp_7: comp_map_ia[6],
                                                   tmp_comp_8: comp_map_ia[7],
                                                   tmp_comp_9: comp_map_ia[8],
                                                   is_first: False,
                                                   is_training: False,
                                                   rate_place_holder: i*5 / float(total_batch_train)})
            for b in range(total_batch_test):
                acc_ia_not += sess.run(accuracy_af,
                                        feed_dict={X: XADV[b][j],
                                                   Y: YADV[b],
                                                   keep_prob: 1.,
                                                   tmp_comp_1: comp_map_ia[0],
                                                   tmp_comp_2: comp_map_ia[1],
                                                   tmp_comp_3: comp_map_ia[2],
                                                   tmp_comp_4: comp_map_ia[3],
                                                   tmp_comp_5: comp_map_ia[4],
                                                   tmp_comp_6: comp_map_ia[5],
                                                   tmp_comp_7: comp_map_ia[6],
                                                   tmp_comp_8: comp_map_ia[7],
                                                   tmp_comp_9: comp_map_ia[8],
                                                   is_first: False,
                                                   is_training: False,
                                                   rate_place_holder: i*5})
            acc_leg_ia_not.append(acc_ia_not / float(total_batch_test))
            print('Acc IAFD-not on legitimate, pruning rate: %d:'%(i*5), acc_leg_ia_not[i+ j*20])
        for i in range(20):
            acc_ia_all = 0
            comp_map_ia = np_ones
            for b in range(total_batch_train):
                comp_map_ia = sess.run(comp_map_ia_op,
                                        feed_dict={X_small: x_train_batches[b],
                                                   Y_small: y_train_batches[b],
                                                   keep_prob: 1.,
                                                   tmp_comp_1: comp_map_ia[0],
                                                   tmp_comp_2: comp_map_ia[1],
                                                   tmp_comp_3: comp_map_ia[2],
                                                   tmp_comp_4: comp_map_ia[3],
                                                   tmp_comp_5: comp_map_ia[4],
                                                   tmp_comp_6: comp_map_ia[5],
                                                   tmp_comp_7: comp_map_ia[6],
                                                   tmp_comp_8: comp_map_ia[7],
                                                   tmp_comp_9: comp_map_ia[8],
                                                   is_first: True,
                                                   is_training: False,
                                                   rate_place_holder: i*5 / float(total_batch_train)})
            for b in range(total_batch_test):
                acc_ia_all += sess.run(accuracy_af,
                                        feed_dict={X: XADV[b][j],
                                                   Y: YADV[b],
                                                   keep_prob: 1.,
                                                   tmp_comp_1: comp_map_ia[0],
                                                   tmp_comp_2: comp_map_ia[1],
                                                   tmp_comp_3: comp_map_ia[2],
                                                   tmp_comp_4: comp_map_ia[3],
                                                   tmp_comp_5: comp_map_ia[4],
                                                   tmp_comp_6: comp_map_ia[5],
                                                   tmp_comp_7: comp_map_ia[6],
                                                   tmp_comp_8: comp_map_ia[7],
                                                   tmp_comp_9: comp_map_ia[8],
                                                   is_first: True,
                                                   is_training: False,
                                                   rate_place_holder: i*5})
            acc_leg_ia_all.append(acc_ia_all / float(total_batch_test))
            print('Acc IAFD-all on legitimate, pruning rate: %d:'%(i*5), acc_leg_ia_all[i+ j*20])
    dict['acc_base'].append(acc_base)
    dict['acc_leg_ap_not'].append(acc_leg_ap_not)   
    dict['acc_leg_ap_all'].append(acc_leg_ap_all)
    dict['acc_leg_af_gra_not'].append(acc_leg_af_gra_not)
    dict['acc_leg_af_gra_all'].append(acc_leg_af_gra_all)
    dict['acc_leg_ia_not'].append(acc_leg_ia_not) 
    dict['acc_leg_ia_all'].append(acc_leg_ia_all) 
    dict['acc_leg_rd_not'].append(acc_leg_rd_not)
    dict['acc_leg_rd_all'].append(acc_leg_rd_all)

for key in dict.keys():
    dict[key] = utils.avg_of_lists(dict[key])
'''''''''
Graph settings
'''''''''
x_axis = rate_axis
fontP = FontProperties()
fontP.set_size('small')
for i in range(len(tmp_ep)-1):
    fig = plt.figure()
    graph_base = fig.add_subplot(311)
    graph_adv = fig.add_subplot(312)
    plt.tight_layout()
    y_0 = dict['acc_base'][0:len(x_axis)]
    y_1_not = dict['acc_leg_ap_not'][0:len(x_axis)]
    y_1_all = dict['acc_leg_ap_all'][0:len(x_axis)]
    y_3_not = dict['acc_leg_af_gra_not'][0:len(x_axis)]
    y_3_all = dict['acc_leg_af_gra_all'][0:len(x_axis)]
    y_6_not = dict['acc_leg_rd_not'][0:len(x_axis)]
    y_6_all = dict['acc_leg_rd_all'][0:len(x_axis)]
    y_8_not = dict['acc_leg_ia_not'][0:len(x_axis)]
    y_8_all = dict['acc_leg_ia_all'][0:len(x_axis)]

    idx_start = (i+1)*len(x_axis)
    idx_end = (i+2)*len(x_axis)
    y_adv_0 = dict['acc_base'][idx_start:idx_end]
    y_adv_1_not = dict['acc_leg_ap_not'][idx_start:idx_end]
    y_adv_1_all = dict['acc_leg_ap_all'][idx_start:idx_end]
    y_adv_3_not = dict['acc_leg_af_gra_not'][idx_start:idx_end]
    y_adv_3_all = dict['acc_leg_af_gra_all'][idx_start:idx_end]
    y_adv_6_not = dict['acc_leg_rd_not'][idx_start:idx_end]
    y_adv_6_all = dict['acc_leg_rd_all'][idx_start:idx_end]
    y_adv_8_not = dict['acc_leg_ia_not'][idx_start:idx_end]
    y_adv_8_all = dict['acc_leg_ia_all'][idx_start:idx_end]

    graph_base.plot(x_axis, y_1_not, label='activation pruning - not')
    graph_base.plot(x_axis, y_1_all, label='activation pruning - all')
    graph_base.plot(x_axis, y_3_not, label='adversarial feature drop - not ')
    graph_base.plot(x_axis, y_3_all, label='adversarial feature drop - all ')
    graph_base.plot(x_axis, y_6_not, label='random feature drop - not')
    graph_base.plot(x_axis, y_6_all, label='random feature drop - all')
    graph_base.plot(x_axis, y_8_not, label='iterative feature drop - not')
    graph_base.plot(x_axis, y_8_all, label='iterative feature drop - all')
    graph_base.plot(x_axis, y_0, '--', label='base')
    graph_base.set_xlabel('Pruning rate')
    graph_base.set_ylabel('Accuracy in clean MNIST')

    graph_adv.plot(x_axis, y_adv_1_not,  label='activation pruning - not')
    graph_adv.plot(x_axis, y_adv_1_all,  label='activation pruning - all')
    graph_adv.plot(x_axis, y_adv_3_not,  label='adversarial feature drop - not')
    graph_adv.plot(x_axis, y_adv_3_all,  label='adversarial feature drop - all')
    graph_adv.plot(x_axis, y_adv_6_not,  label='random feature drop - not')
    graph_adv.plot(x_axis, y_adv_6_all,  label='random feature drop - all')
    graph_adv.plot(x_axis, y_adv_8_not, label='iterative feature drop - not')
    graph_adv.plot(x_axis, y_adv_8_all, label='iterative feature drop - all')
    graph_adv.plot(x_axis, y_adv_0, '--', label='base')
    graph_adv.set_xlabel('Pruning rate')
    graph_adv.set_ylabel('Accuracy in adv MNIST; epsilon = %.2f'%epsilon[i+1])
    
    plt.legend(loc='lower center', ncol=2, prop=fontP, bbox_to_anchor=(0.5, -1.5))
    plt.savefig('cifar%d_dropout.png'%tmp_ep[i+1])
    #plt.legend(loc='best')
    #plt.show()

