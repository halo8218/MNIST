import tensorflow as tf
import numpy as np
import utils
import matplotlib
matplotlib.use('tkagg')
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
tmp_comp_1 = tf.placeholder(tf.float32, [32, 32, 64])
tmp_comp_2 = tf.placeholder(tf.float32, [16, 16, 64])
tmp_comp_3 = tf.placeholder(tf.float32, [8, 8, 128])
tmp_comp_4 = tf.placeholder(tf.float32, [8, 8, 128])
tmp_comp_5 = tf.placeholder(tf.float32, [8, 8, 128])
tmp_comp_6 = tf.placeholder(tf.float32, [384])
comp_map_place_holder_list = [tmp_comp_1, tmp_comp_2, tmp_comp_3, tmp_comp_4, tmp_comp_5, tmp_comp_6]

Y = tf.placeholder(tf.float32, [None, 10])
Y_small = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
is_first = tf.placeholder(tf.bool)

rate_place_holder = tf.placeholder(tf.float32, [])

W_conv1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 3, 64], stddev=5e-2))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[64]))
h_conv1_prev = tf.nn.conv2d(X, W_conv1, strides=[1, 1, 1, 1], padding='SAME') 
h_conv1 = tf.nn.relu(h_conv1_prev + b_conv1)

h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

W_conv2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 64, 64], stddev=5e-2))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
h_conv2_prev = tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
h_conv2 = tf.nn.relu(h_conv2_prev + b_conv2)

h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

W_conv3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], stddev=5e-2))
b_conv3 = tf.Variable(tf.constant(0.1, shape=[128]))
h_conv3_prev = tf.nn.conv2d(h_pool2, W_conv3, strides=[1, 1, 1, 1], padding='SAME')
h_conv3 = tf.nn.relu(h_conv3_prev + b_conv3)

W_conv4 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 128], stddev=5e-2))
b_conv4 = tf.Variable(tf.constant(0.1, shape=[128])) 
h_conv4_prev = tf.nn.conv2d(h_conv3, W_conv4, strides=[1, 1, 1, 1], padding='SAME') 
h_conv4 = tf.nn.relu(h_conv4_prev + b_conv4)

W_conv5 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 128], stddev=5e-2))
b_conv5 = tf.Variable(tf.constant(0.1, shape=[128]))
h_conv5_prev = tf.nn.conv2d(h_conv4, W_conv5, strides=[1, 1, 1, 1], padding='SAME') 
h_conv5 = tf.nn.relu(h_conv5_prev + b_conv5)

W_fc1 = tf.Variable(tf.truncated_normal(shape=[8 * 8 * 128, 384], stddev=5e-2))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[384]))

h_conv5_flat = tf.reshape(h_conv5, [-1, 8*8*128])
h_fc1_prev = tf.matmul(h_conv5_flat, W_fc1)
h_fc1 = tf.nn.relu(h_fc1_prev + b_fc1)

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob) 

W_fc2 = tf.Variable(tf.truncated_normal(shape=[384, 10], stddev=5e-2))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
model = tf.matmul(h_fc1_drop,W_fc2) + b_fc2

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

'''''''''
Make the node generating adv' examples
'''''''''
grad = tf.gradients(cost, X)
#tmp_ep = [0, 1, 2, 4, 8, 16, 32, 64]
tmp_ep = [0, 8]
epsilon = np.array(tmp_ep) / 256.0
xadv = [tf.stop_gradient(X + e*tf.sign(grad)) for e in epsilon]
xadv = [tf.clip_by_value(adv, 0., 1.) for adv in xadv]
xadv = [tf.reshape(adv,[-1,32, 32, 3]) for adv in xadv]
yadv = Y
'''''''''
Build replica model for comparing
'''''''''
h_conv1_comp_prev = tf.nn.conv2d(X_small, W_conv1, strides=[1, 1, 1, 1], padding='SAME') 
h_conv1_comp = tf.nn.relu(h_conv1_comp_prev + b_conv1)
h_pool1_comp = tf.nn.max_pool(h_conv1_comp, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
h_conv2_comp_prev = tf.nn.conv2d(h_pool1_comp, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
h_conv2_comp = tf.nn.relu(h_conv2_comp_prev + b_conv2)
h_pool2_comp = tf.nn.max_pool(h_conv2_comp, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
h_conv3_comp_prev = tf.nn.conv2d(h_pool2_comp, W_conv3, strides=[1, 1, 1, 1], padding='SAME')
h_conv3_comp = tf.nn.relu(h_conv3_comp_prev + b_conv3)
h_conv4_comp_prev = tf.nn.conv2d(h_conv3_comp, W_conv4, strides=[1, 1, 1, 1], padding='SAME') 
h_conv4_comp = tf.nn.relu(h_conv4_comp_prev + b_conv4)
h_conv5_comp_prev = tf.nn.conv2d(h_conv4_comp, W_conv5, strides=[1, 1, 1, 1], padding='SAME') 
h_conv5_comp = tf.nn.relu(h_conv5_comp_prev + b_conv5)
h_conv5_comp_flat = tf.reshape(h_conv5_comp, [-1, 8*8*128])
h_fc1_comp_prev = tf.matmul(h_conv5_comp_flat, W_fc1)
h_fc1_comp = tf.nn.relu(h_fc1_comp_prev + b_fc1)

model_comp = tf.matmul(h_fc1_comp,W_fc2) + b_fc2

cost_comp = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model_comp, labels=Y_small))
grad_comp = tf.gradients(cost_comp, X_small)
xadv_small = tf.stop_gradient(X_small + 0.3*tf.sign(grad_comp))
xadv_small = tf.clip_by_value(xadv_small, 0., 1.)
xadv_small = tf.reshape(xadv_small, [-1, 32, 32, 3])

h_conv1_comp_adv_prev = tf.nn.conv2d(xadv_small, W_conv1, strides=[1, 1, 1, 1], padding='SAME') 
h_conv1_comp_adv = tf.nn.relu(h_conv1_comp_adv_prev + b_conv1)
h_pool1_comp_adv = tf.nn.max_pool(h_conv1_comp_adv, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
h_conv2_comp_adv_prev = tf.nn.conv2d(h_pool1_comp_adv, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
h_conv2_comp_adv = tf.nn.relu(h_conv2_comp_adv_prev + b_conv2)
h_pool2_comp_adv = tf.nn.max_pool(h_conv2_comp_adv, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
h_conv3_comp_adv_prev = tf.nn.conv2d(h_pool2_comp_adv, W_conv3, strides=[1, 1, 1, 1], padding='SAME')
h_conv3_comp_adv = tf.nn.relu(h_conv3_comp_adv_prev + b_conv3)
h_conv4_comp_adv_prev = tf.nn.conv2d(h_conv3_comp_adv, W_conv4, strides=[1, 1, 1, 1], padding='SAME') 
h_conv4_comp_adv = tf.nn.relu(h_conv4_comp_adv_prev + b_conv4)
h_conv5_comp_adv_prev = tf.nn.conv2d(h_conv4_comp_adv, W_conv5, strides=[1, 1, 1, 1], padding='SAME') 
h_conv5_comp_adv = tf.nn.relu(h_conv5_comp_adv_prev + b_conv5)
h_conv5_comp_adv_flat = tf.reshape(h_conv5_comp_adv, [-1, 8*8*128])
h_fc1_comp_adv_prev = tf.matmul(h_conv5_comp_adv_flat, W_fc1)
h_fc1_comp_adv = tf.nn.relu(h_fc1_comp_adv_prev + b_fc1)

model_comp_adv = tf.matmul(h_fc1_comp_adv,W_fc2) + b_fc2

'''''''''
Magnitude Based Activation Pruning Model
'''''''''
def MBAP(pruning_rate_per_layer, is_first):
    _, mask_1 = utils.prune_conv_feature(h_conv1, pruning_rate_per_layer)
    h_conv1_ap = tf.cond(is_first, lambda: h_conv1 * mask_1, lambda: h_conv1)
    h_pool1_ap = tf.nn.max_pool(h_conv1_ap, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    h_conv2_ap_prev = tf.nn.conv2d(h_pool1_ap, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
    h_conv2_ap = tf.nn.relu(h_conv2_ap_prev + b_conv2)
    h_conv2_ap, _ = utils.prune_conv_feature(h_conv2_ap, pruning_rate_per_layer)
    h_pool2_ap = tf.nn.max_pool(h_conv2_ap, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    h_conv3_ap_prev = tf.nn.conv2d(h_pool2_ap, W_conv3, strides=[1, 1, 1, 1], padding='SAME')
    h_conv3_ap = tf.nn.relu(h_conv3_ap_prev + b_conv3)
    h_conv3_ap, _ = utils.prune_conv_feature(h_conv3_ap, pruning_rate_per_layer)
    h_conv4_ap_prev = tf.nn.conv2d(h_conv3_ap, W_conv4, strides=[1, 1, 1, 1], padding='SAME') 
    h_conv4_ap = tf.nn.relu(h_conv4_ap_prev + b_conv4)
    h_conv4_ap, _ = utils.prune_conv_feature(h_conv4_ap, pruning_rate_per_layer)
    h_conv5_ap_prev = tf.nn.conv2d(h_conv4_ap, W_conv5, strides=[1, 1, 1, 1], padding='SAME') 
    h_conv5_ap = tf.nn.relu(h_conv5_ap_prev + b_conv5)
    h_conv5_ap, _ = utils.prune_conv_feature(h_conv5_ap, pruning_rate_per_layer)
    h_conv5_ap_flat = tf.reshape(h_conv5_ap, [-1, 8*8*128])
    h_fc1_ap_prev = tf.matmul(h_conv5_ap_flat, W_fc1)
    h_fc1_ap = tf.nn.relu(h_fc1_ap_prev + b_fc1)
    h_fc1_ap, _ = utils.prune(h_fc1_ap, pruning_rate_per_layer)

    model_ap = tf.matmul(h_fc1_ap,W_fc2) + b_fc2

    return model_ap

'''''''''
Adversarial Feature Drop Model
'''''''''
def compare():
    diff = tf.norm(model_comp - model_comp_adv, 2) 
    cand = [h_conv1_comp_adv_prev, h_conv2_comp_adv_prev, h_conv3_comp_adv_prev, h_conv4_comp_adv_prev, h_conv5_comp_adv_prev, h_fc1_comp_adv_prev]
    grad_on_diff = tf.gradients(diff, cand) 
    comp_map = [tf.reduce_sum(tf.abs(grad_on_diff[i]), axis=0) for i in range(len(cand))]
    return comp_map

def AFD(pruning_rate_per_layer, is_first, comp_map):
    adv_feat = comp_map
    mask = [utils.mask_map(adv_feat[i], pruning_rate_per_layer) for i in range(len(adv_feat))]

    h_conv1_af_prev = tf.nn.conv2d(X, W_conv1, strides=[1, 1, 1, 1], padding='SAME') 
    h_conv1_af_prev = tf.cond(is_first, lambda: h_conv1_af_prev * mask[0], lambda: h_conv1_af_prev) 
    h_conv1_af = tf.nn.relu(h_conv1_af_prev + b_conv1)
    h_pool1_af = tf.nn.max_pool(h_conv1_af, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    h_conv2_af_prev = tf.nn.conv2d(h_pool1_af, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
    h_conv2_af_prev = h_conv2_af_prev * mask[1]
    h_conv2_af = tf.nn.relu(h_conv2_af_prev + b_conv2)
    h_pool2_af = tf.nn.max_pool(h_conv2_af, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    h_conv3_af_prev = tf.nn.conv2d(h_pool2_af, W_conv3, strides=[1, 1, 1, 1], padding='SAME')
    h_conv3_af_prev = h_conv3_af_prev * mask[2]
    h_conv3_af = tf.nn.relu(h_conv3_af_prev + b_conv3)
    h_conv4_af_prev = tf.nn.conv2d(h_conv3_af, W_conv4, strides=[1, 1, 1, 1], padding='SAME') 
    h_conv4_af_prev = h_conv4_af_prev * mask[3]
    h_conv4_af = tf.nn.relu(h_conv4_af_prev + b_conv4)
    h_conv5_af_prev = tf.nn.conv2d(h_conv4_af, W_conv5, strides=[1, 1, 1, 1], padding='SAME') 
    h_conv5_af_prev = h_conv5_af_prev * mask[4]
    h_conv5_af = tf.nn.relu(h_conv5_af_prev + b_conv5)
    h_conv5_af_flat = tf.reshape(h_conv5_af, [-1, 8*8*128])
    h_fc1_af_prev = tf.matmul(h_conv5_af_flat, W_fc1)
    h_fc1_af_prev = h_fc1_af_prev * mask[5]
    h_fc1_af = tf.nn.relu(h_fc1_af_prev + b_fc1)
    
    model_af = tf.matmul(h_fc1_af,W_fc2) + b_fc2
    return model_af

'''''''''
Random Feature Drop Model
'''''''''
def RFD(pruning_rate_per_layer, is_first):
    ran_feat = utils.random_map(h_conv1, h_conv2, h_conv3, h_conv4, h_conv5, h_fc1)
    mask = [utils.mask_map(ran_feat[i], pruning_rate_per_layer) for i in range(len(ran_feat))]

    h_conv1_rd_prev = tf.nn.conv2d(X, W_conv1, strides=[1, 1, 1, 1], padding='SAME') 
    h_conv1_rd_prev = tf.cond(is_first, lambda: h_conv1_rd_prev * mask[0], lambda: h_conv1_rd_prev) 
    h_conv1_rd = tf.nn.relu(h_conv1_rd_prev + b_conv1)
    h_pool1_rd = tf.nn.max_pool(h_conv1_rd, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    h_conv2_rd_prev = tf.nn.conv2d(h_pool1_rd, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
    h_conv2_rd_prev = h_conv2_rd_prev * mask[1]
    h_conv2_rd = tf.nn.relu(h_conv2_rd_prev + b_conv2)
    h_pool2_rd = tf.nn.max_pool(h_conv2_rd, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    h_conv3_rd_prev = tf.nn.conv2d(h_pool2_rd, W_conv3, strides=[1, 1, 1, 1], padding='SAME')
    h_conv3_rd_prev = h_conv3_rd_prev * mask[2]
    h_conv3_rd = tf.nn.relu(h_conv3_rd_prev + b_conv3)
    h_conv4_rd_prev = tf.nn.conv2d(h_conv3_rd, W_conv4, strides=[1, 1, 1, 1], padding='SAME') 
    h_conv4_rd_prev = h_conv4_rd_prev * mask[3]
    h_conv4_rd = tf.nn.relu(h_conv4_rd_prev + b_conv4)
    h_conv5_rd_prev = tf.nn.conv2d(h_conv4_rd, W_conv5, strides=[1, 1, 1, 1], padding='SAME') 
    h_conv5_rd_prev = h_conv5_rd_prev * mask[4]
    h_conv5_rd = tf.nn.relu(h_conv5_rd_prev + b_conv5)
    h_conv5_rd_flat = tf.reshape(h_conv5_rd, [-1, 8*8*128])
    h_fc1_rd_prev = tf.matmul(h_conv5_rd_flat, W_fc1)
    h_fc1_rd_prev = h_fc1_rd_prev * mask[5]
    h_fc1_rd = tf.nn.relu(h_fc1_rd_prev + b_fc1)
    
    model_rd = tf.matmul(h_fc1_rd,W_fc2) + b_fc2

    return model_rd

'''''''''
Iterative Adversarial Feature Drop Model
'''''''''
def _body(pruning_rate_per_step, mask):
    '''''''''
    Build replica model for iterative comparing
    '''''''''
    h_conv1_comp_prev = tf.nn.conv2d(X_small, W_conv1, strides=[1, 1, 1, 1], padding='SAME') 
    h_conv1_comp_prev = tf.cond(is_first, lambda: h_conv1_comp_prev * mask[0], lambda: h_conv1_comp_prev) 
    h_conv1_comp = tf.nn.relu(h_conv1_comp_prev + b_conv1)
    h_pool1_comp = tf.nn.max_pool(h_conv1_comp, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    h_conv2_comp_prev = tf.nn.conv2d(h_pool1_comp, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
    h_conv2_comp = tf.nn.relu(h_conv2_comp_prev * mask[1] + b_conv2)
    h_pool2_comp = tf.nn.max_pool(h_conv2_comp, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    h_conv3_comp_prev = tf.nn.conv2d(h_pool2_comp, W_conv3, strides=[1, 1, 1, 1], padding='SAME')
    h_conv3_comp = tf.nn.relu(h_conv3_comp_prev * mask[2] + b_conv3)
    h_conv4_comp_prev = tf.nn.conv2d(h_conv3_comp, W_conv4, strides=[1, 1, 1, 1], padding='SAME') 
    h_conv4_comp = tf.nn.relu(h_conv4_comp_prev * mask[3] + b_conv4)
    h_conv5_comp_prev = tf.nn.conv2d(h_conv4_comp, W_conv5, strides=[1, 1, 1, 1], padding='SAME') 
    h_conv5_comp = tf.nn.relu(h_conv5_comp_prev * mask[4] + b_conv5)
    h_conv5_comp_flat = tf.reshape(h_conv5_comp, [-1, 8*8*128])
    h_fc1_comp_prev = tf.matmul(h_conv5_comp_flat, W_fc1)
    h_fc1_comp = tf.nn.relu(h_fc1_comp_prev * mask[5] + b_fc1)
    
    model_comp = tf.matmul(h_fc1_comp,W_fc2) + b_fc2
    
    cost_comp = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model_comp, labels=Y_small))
    grad_comp = tf.gradients(cost_comp, X_small)
    xadv_small = tf.stop_gradient(X_small + 0.3*tf.sign(grad_comp))
    xadv_small = tf.clip_by_value(xadv_small, 0., 1.)
    xadv_small = tf.reshape(xadv_small, [-1, 32, 32, 3])
    
    h_conv1_comp_adv_prev = tf.nn.conv2d(xadv_small, W_conv1, strides=[1, 1, 1, 1], padding='SAME') 
    h_conv1_comp_adv_prev = tf.cond(is_first, lambda: h_conv1_comp_adv_prev * mask[0], lambda: h_conv1_comp_adv_prev) 
    h_conv1_comp_adv = tf.nn.relu(h_conv1_comp_adv_prev + b_conv1)
    h_pool1_comp_adv = tf.nn.max_pool(h_conv1_comp_adv, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    h_conv2_comp_adv_prev = tf.nn.conv2d(h_pool1_comp_adv, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
    h_conv2_comp_adv = tf.nn.relu(h_conv2_comp_adv_prev * mask[1] + b_conv2)
    h_pool2_comp_adv = tf.nn.max_pool(h_conv2_comp_adv, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    h_conv3_comp_adv_prev = tf.nn.conv2d(h_pool2_comp_adv, W_conv3, strides=[1, 1, 1, 1], padding='SAME')
    h_conv3_comp_adv = tf.nn.relu(h_conv3_comp_adv_prev * mask[2] + b_conv3)
    h_conv4_comp_adv_prev = tf.nn.conv2d(h_conv3_comp_adv, W_conv4, strides=[1, 1, 1, 1], padding='SAME') 
    h_conv4_comp_adv = tf.nn.relu(h_conv4_comp_adv_prev * mask[3] + b_conv4)
    h_conv5_comp_adv_prev = tf.nn.conv2d(h_conv4_comp_adv, W_conv5, strides=[1, 1, 1, 1], padding='SAME') 
    h_conv5_comp_adv = tf.nn.relu(h_conv5_comp_adv_prev * mask[4] + b_conv5)
    h_conv5_comp_adv_flat = tf.reshape(h_conv5_comp_adv, [-1, 8*8*128])
    h_fc1_comp_adv_prev = tf.matmul(h_conv5_comp_adv_flat, W_fc1)
    h_fc1_comp_adv = tf.nn.relu(h_fc1_comp_adv_prev * mask[5] + b_fc1)
    
    model_comp_adv = tf.matmul(h_fc1_comp_adv,W_fc2) + b_fc2
    '''''''''
    '''''''''
    diff = tf.norm(model_comp - model_comp_adv, 2) 
    cand = [h_conv1_comp_adv_prev, h_conv2_comp_adv_prev, h_conv3_comp_adv_prev, h_conv4_comp_adv_prev, h_conv5_comp_adv_prev, h_fc1_comp_adv_prev]
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
ones = [tf.ones(h_conv1.get_shape()[1:]), tf.ones(h_conv2.get_shape()[1:]), tf.ones(h_conv3.get_shape()[1:]), tf.ones(h_conv4.get_shape()[1:]), tf.ones(h_conv5.get_shape()[1:]), tf.ones(h_fc1.get_shape()[1])]
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

batch_size = 512
total_batch_train = int(len(x_train) / batch_size)
batch_size_test = 2048
total_batch_test = int(len(x_test) / batch_size_test)

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
num_avg = 1
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

    sess.run(init)

    for epoch in range(50):
        x_train_batches, y_train_batches = utils.shuffle_and_devide_into_batches(batch_size, x_train, y_train_one_hot)
        total_cost = 0
    
        for i in range(total_batch_train):
            batch_xs = x_train_batches[i]
            batch_ys = y_train_batches[i]
    
            _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.8})
            total_cost += cost_val
    
        print('Epoch:', '%04d' % (epoch + 1),
              'Avg. cost =', '{:.3f}'.format(total_cost / total_batch_train))
    
    print('Training completed!')
    x_test_batches, y_test_batches = utils.shuffle_and_devide_into_batches(batch_size_test, x_test, y_test_one_hot)
    base_acc = 0
    for t in range(total_batch_test): 
        base_acc += sess.run(accuracy,feed_dict={X: x_test_batches[t],
                                           keep_prob: 1.,
                                           Y: y_test_batches[t]})
    print('Acc on legitimate:', base_acc / float(total_batch_test))
    
    print('Make adversarial test sets')
    XADV = []
    YADV = y_test_batches
    for t in range(total_batch_test):
        XADV.append(sess.run(xadv, feed_dict={X: x_test_batches[t], Y: y_test_batches[t], keep_prob: 1.}))
    comp_map_af = np_ones
    for t in range(total_batch_train):
        comp_map_af += np.array(sess.run(comp_map_af_op,
                                feed_dict={X_small: x_train_batches[t],
                                           Y_small: y_train_batches[t],
                                           keep_prob: 1.}))
    for j in range(len(tmp_ep)):
        print('%.2f epsilon trial'%epsilon[j])
        acc_leg_base = 0
        for b in range(total_batch_test):
            acc_leg_base += sess.run(accuracy,
                                    feed_dict={X: XADV[b][j],
                                               Y: YADV[b],
                                               keep_prob: 1.})
            print('Acc on adversarial examples:', acc_leg_base)
        for i in range(20):
            acc_base.append(acc_leg_base / float(total_batch_test))
        for i in range(20):
            acc_ap_not = 0
            for b in range(total_batch_test):
                acc_ap_not += sess.run(accuracy_ap,
                                        feed_dict={X: XADV[b][j],
                                                   Y: YADV[b],
                                                   keep_prob: 1.,
                                                   is_first: False,
                                                   rate_place_holder: i*5})
            acc_leg_ap_not.append(acc_ap_not / float(total_batch_test))
            print('Acc MBAP-not on legitimate, pruning rate: %d:'%(i*5), acc_leg_ap_not[i])
        for i in range(20):
            acc_ap_all = 0
            for b in range(total_batch_test):
                acc_ap_all += sess.run(accuracy_ap,
                                        feed_dict={X: XADV[b][j],
                                                   Y: YADV[b],
                                                   keep_prob: 1.,
                                                   is_first: True,
                                                   rate_place_holder: i*5})
            acc_leg_ap_all.append(acc_ap_all / float(total_batch_test))
            print('Acc MBAP-all on legitimate, pruning rate: %d:'%(i*5), acc_leg_ap_all[i])
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
                                                   keep_prob: 1.,
                                                   is_first: False,
                                                   rate_place_holder: i*5})
            acc_leg_af_gra_not.append(acc_af_gra_not / float(total_batch_test))
            print('Acc AFD-not on legitimate, pruning rate: %d:'%(i*5), acc_leg_af_gra_not[i])
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
                                                   keep_prob: 1.,
                                                   is_first: True,
                                                   rate_place_holder: i*5})
            acc_leg_af_gra_all.append(acc_af_gra_all / float(total_batch_test))
            print('Acc AFD-all on legitimate, pruning rate: %d:'%(i*5), acc_leg_af_gra_all[i]) 
        for i in range(20):
            acc_rd_not = 0
            for b in range(total_batch_test):
                acc_rd_not += sess.run(accuracy_rd,
                                        feed_dict={X: XADV[b][j],
                                                   Y: YADV[b],
                                                   keep_prob: 1.,
                                                   is_first: False,
                                                   rate_place_holder: i*5})
            acc_leg_rd_not.append(acc_rd_not / float(total_batch_test))
            print('Acc RFD-not on legitimate, pruning rate: %d:'%(i*5), acc_leg_rd_not[i]) 
        for i in range(20):
            acc_rd_all = 0
            for b in range(total_batch_test):
                acc_rd_all += sess.run(accuracy_rd,
                                        feed_dict={X: XADV[b][j],
                                                   Y: YADV[b],
                                                   keep_prob: 1.,
                                                   is_first: True,
                                                   rate_place_holder: i*5})
            acc_leg_rd_all.append(acc_rd_all / float(total_batch_test))
            print('Acc RFD-all on legitimate, pruning rate: %d:'%(i*5), acc_leg_rd_all[i]) 
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
                                                   is_first: False,
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
                                                   is_first: False,
                                                   rate_place_holder: i*5})
            acc_leg_ia_not.append(acc_ia_not / float(total_batch_test))
            print('Acc IAFD-not on legitimate, pruning rate: %d:'%(i*5), acc_leg_ia_not[i])
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
                                                   is_first: True,
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
                                                   is_first: True,
                                                   rate_place_holder: i*5})
            acc_leg_ia_all.append(acc_ia_all / float(total_batch_test))
            print('Acc IAFD-all on legitimate, pruning rate: %d:'%(i*5), acc_leg_ia_all[i])
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
for i in range(len(tmp_ep)-1):
    fig = plt.figure()
    graph_base = fig.add_subplot(2,1,1)
    graph_adv = fig.add_subplot(2,1,2)
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
    graph_base.plot(x_axis, y_3_not, label='adversarial feature drop, gradients base - not ')
    graph_base.plot(x_axis, y_3_all, label='adversarial feature drop, gradients base - all ')
    graph_base.plot(x_axis, y_6_not, label='random feature drop - not')
    graph_base.plot(x_axis, y_6_all, label='random feature drop - all')
    graph_base.plot(x_axis, y_8_not, label='iterative gradients based feature drop - not')
    graph_base.plot(x_axis, y_8_all, label='iterative gradients based feature drop - all')
    graph_base.plot(x_axis, y_0, '--', label='base')
    graph_base.set_xlabel('Pruning rate')
    graph_base.set_ylabel('Accuracy in clean MNIST')

    graph_adv.plot(x_axis, y_adv_1_not,  label='activation pruning - not')
    graph_adv.plot(x_axis, y_adv_1_all,  label='activation pruning - all')
    graph_adv.plot(x_axis, y_adv_3_not,  label='adversarial feature drop, gradients base - not')
    graph_adv.plot(x_axis, y_adv_3_all,  label='adversarial feature drop, gradients base - all')
    graph_adv.plot(x_axis, y_adv_6_not,  label='random feature drop - not')
    graph_adv.plot(x_axis, y_adv_6_all,  label='random feature drop - all')
    graph_adv.plot(x_axis, y_adv_8_not, label='iterative gradients based feature drop - not')
    graph_adv.plot(x_axis, y_adv_8_all, label='iterative gradients based feature drop - all')
    graph_adv.plot(x_axis, y_adv_0, '--', label='base')
    graph_adv.set_xlabel('Pruning rate')
    graph_adv.set_ylabel('Accuracy in adv MNIST; epsilon = %.2f'%epsilon[i+1])
    
    plt.legend(loc='best')
    plt.show()

#def gen_image(arr):
#    two_d = (np.reshape(arr, (28, 28)) * 255).astype(np.uint8)
#    plt.imshow(two_d, interpolation='nearest')
#    return plt
#
## Get a batch of two random images and show in a pop-up window.
#print(YADV[0])
#for i in range(len(XADV)):
#    gen_image(XADV[i][0]).show()
