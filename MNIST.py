import tensorflow as tf
import numpy as np
import utils
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

'''''''''
Build base model
'''''''''
X = tf.placeholder(tf.float32, [None, 784])
X_small = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])
Y_small = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
is_grad_compare = tf.placeholder(tf.bool)
is_first = tf.placeholder(tf.bool)
is_last = tf.placeholder(tf.bool)

rate_place_holder = tf.placeholder(tf.int32, [])

W1 = tf.Variable(tf.random_normal([784, 512], stddev=0.01))
B1 = tf.Variable(tf.zeros([512]))
L1_prev = tf.matmul(X, W1)
L1 = tf.nn.relu(L1_prev + B1)
L1 = tf.nn.dropout(L1, keep_prob)

W2 = tf.Variable(tf.random_normal([512, 256], stddev=0.01))
B2 = tf.Variable(tf.zeros([256]))
L2_prev = tf.matmul(L1, W2)
L2 = tf.nn.relu(L2_prev + B2)
L2 = tf.nn.dropout(L2, keep_prob)

W3 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
B3 = tf.Variable(tf.zeros([10]))
model = tf.matmul(L2, W3) + B3

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

grad = tf.gradients(cost, X)
epsilon = np.arange(0., 0.35, 0.05)
#epsilon = [0.075]
xadv = [tf.stop_gradient(X + e*tf.sign(grad)) for e in epsilon]
xadv = [tf.clip_by_value(adv, 0., 1.) for adv in xadv]
xadv = [tf.reshape(adv,[-1,784]) for adv in xadv]
yadv = Y
'''''''''
Build replica model for comparing
'''''''''
L1_comp_prev = tf.matmul(X_small, W1)
L1_comp = tf.nn.relu(L1_comp_prev + B1)
L2_comp_prev = tf.matmul(L1_comp, W2)
L2_comp = tf.nn.relu(L2_comp_prev + B2)
model_comp = tf.matmul(L2_comp, W3) + B3

cost_comp = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model_comp, labels=Y_small))
grad_comp = tf.gradients(cost_comp, X_small)
xadv_small = tf.stop_gradient(X_small + 0.3*tf.sign(grad_comp))
xadv_small = tf.clip_by_value(xadv_small, 0., 1.)
xadv_small = tf.reshape(xadv_small, [-1, 784])

L1_comp_prev_adv = tf.matmul(xadv_small, W1)
L1_comp_adv = tf.nn.relu(L1_comp_prev_adv + B1)
L2_comp_prev_adv = tf.matmul(L1_comp_adv, W2)
L2_comp_adv = tf.nn.relu(L2_comp_prev_adv + B2)
model_comp_adv = tf.matmul(L2_comp_adv, W3) + B3

'''''''''
Magnitude Based Activation Pruning Model
'''''''''
def MBAP(pruning_rate_per_layer, is_first, is_last):
    _, mask_1 = utils.prune(L1, pruning_rate_per_layer)
    L1_ap = tf.cond(is_first, lambda: L1 * mask_1, lambda: L1)
    L2_ap = tf.nn.relu(tf.matmul(L1_ap, W2) + B2)
    _, mask_2 = utils.prune(L2_ap, pruning_rate_per_layer)
    pruned_L2_ap = tf.cond(is_last, lambda: L2_ap * mask_2, lambda: L2_ap)
    model_ap = tf.matmul(pruned_L2_ap, W3) + B3
    return model_ap

'''''''''
Gradients Based Activation Pruning Model
'''''''''
def compare_gb():
    diff = tf.norm(model_comp - model_comp_adv, 2) 
    grad_on_diff = tf.gradients(diff, [L1_comp_adv, L2_comp_adv]) 
    comp_vec_1 = tf.reduce_sum(tf.abs(grad_on_diff[0]), axis=0)
    comp_vec_2 = tf.reduce_sum(tf.abs(grad_on_diff[1]), axis=0)
    return comp_vec_1, comp_vec_2

def GBAP(pruning_rate_per_layer):
    adv_feat_1, adv_feat_2 = compare_gb()
    mask = utils.mask_vec(adv_feat_2, pruning_rate_per_layer)
    #L1_af = L1 * utils.mask_vec(adv_feat_1, pruning_rate_per_layer)
    L1_gb = L1
    L2_gb = mask * tf.nn.relu(tf.matmul(L1_gb, W2) + B2)
    model_gb = tf.matmul(L2_gb, W3) + B3
    return model_gb

'''''''''
Magnitude Based Feature Drop Model
'''''''''
def MBFD(pruning_rate_per_layer):
    #L1_mb, _ = utils.prune(L1, pruning_rate_per_layer)
    L1_mb = L1
    L2_mb_prev = tf.matmul(L1_mb, W2)
    pruned_L2_mb_prev, _ = utils.prune(L2_mb_prev, pruning_rate_per_layer)
    L2_mb = tf.nn.relu(pruned_L2_mb_prev + B2)
    model_mb = tf.matmul(L2_mb, W3) + B3
    return model_mb

'''''''''
Adversarial Feature Drop Model
'''''''''
def compare(bool_place_holder):
    diff = tf.norm(model_comp - model_comp_adv, 2) 
    grad_on_diff = tf.gradients(diff, [L1_comp_prev_adv, L2_comp_prev_adv]) 
    comp_vec_1 = tf.cond(bool_place_holder, lambda: tf.reduce_sum(tf.abs(grad_on_diff[0]), axis=0), lambda: tf.reduce_sum(tf.abs(L1_comp_prev - L1_comp_prev_adv), axis=0))
    comp_vec_2 = tf.cond(bool_place_holder, lambda: tf.reduce_sum(tf.abs(grad_on_diff[1]), axis=0), lambda: tf.reduce_sum(tf.abs(L2_comp_prev - L2_comp_prev_adv), axis=0))
    return comp_vec_1, comp_vec_2

def AFD(pruning_rate_per_layer, is_first, is_last):
    adv_feat_1, adv_feat_2 = compare(is_grad_compare)
    mask_1 = utils.mask_vec(adv_feat_1, pruning_rate_per_layer)
    mask_2 = utils.mask_vec(adv_feat_2, pruning_rate_per_layer)
    L1_af_prev = tf.cond(is_first, lambda: L1_prev * mask_1, lambda: L1_prev)
    L1_af = tf.nn.relu(L1_af_prev + B1)
    L2_af = tf.cond(is_last, lambda: tf.nn.relu(tf.matmul(L1_af, W2) * mask_2 + B2), lambda: tf.nn.relu(tf.matmul(L1_af, W2) + B2))
    model_af = tf.matmul(L2_af, W3) + B3
    return model_af

'''''''''
Random Feature Drop Model
'''''''''
def RFD(pruning_rate_per_layer, is_first, is_last):
    ran_feat_1, ran_feat_2 = utils.random_vector(L1, L2)
    mask_1 = utils.mask_vec(ran_feat_1, pruning_rate_per_layer)
    mask_2 = utils.mask_vec(ran_feat_2, pruning_rate_per_layer)
    L1_rd_prev = tf.cond(is_first, lambda: L1_prev * mask_1, lambda: L1_prev)
    L1_rd = tf.nn.relu(L1_rd_prev + B1)
    L2_rd = tf.cond(is_last, lambda: tf.nn.relu(tf.matmul(L1_rd, W2) * mask_2 + B2), lambda: tf.nn.relu(tf.matmul(L1_rd, W2) + B2))
    model_rd = tf.matmul(L2_rd, W3) + B3
    return model_rd

'''''''''
Mild Adversarial Feature Drop Model
'''''''''
def MFD(pruning_rate_per_layer):
    adv_feat_mf_1, adv_feat_mf_2 = compare(is_grad_compare)
    #L1_mf = L1 * utils.mask_vec(adjusted_feat_1, pruning_rate_per_layer)
    L1_mf = L1
    L2_mf_prev = tf.matmul(L1_mf, W2)
    _, mask = utils.prune(L2_mf_prev / adv_feat_mf_2, pruning_rate_per_layer)
    L2_mf_prev = L2_mf_prev * mask
    L2_mf = tf.nn.relu(L2_mf_prev + B2)
    model_mf = tf.matmul(L2_mf, W3) + B3
    return model_mf

'''''''''
Iterative Adversarial Feature Drop Model
'''''''''
iter_num = 5
def _cond(pruning_rate_per_layer, mask_1, mask_2, i):
    return tf.less(i, iter_num)

def _body(pruning_rate_per_layer, mask_1, mask_2, i):
    '''''''''
    Build replica model for iterative comparing
    '''''''''
    tmp_X_small = X_small[i::iter_num, :]
    L1_comp_prev = tf.matmul(tmp_X_small, W1)
    masked_L1_comp_prev = L1_comp_prev * mask_1
    L1_comp = tf.nn.relu(masked_L1_comp_prev + B1)
    L2_comp_prev = tf.matmul(L1_comp, W2)
    masked_L2_comp_prev = L2_comp_prev * mask_2
    L2_comp = tf.nn.relu(masked_L2_comp_prev + B2)
    model_comp = tf.matmul(L2_comp, W3) + B3
    
    cost_comp = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model_comp, labels=Y_small[i::iter_num,:]))
    grad_comp = tf.gradients(cost_comp, tmp_X_small)
    xadv_small = tf.stop_gradient(tmp_X_small + 0.3*tf.sign(grad_comp))
    xadv_small = tf.clip_by_value(xadv_small, 0., 1.)
    xadv_small = tf.reshape(xadv_small, [-1, 784])
    
    L1_comp_prev_adv = tf.matmul(xadv_small, W1)
    masked_L1_comp_prev_adv = L1_comp_prev_adv * mask_1
    L1_comp_adv = tf.nn.relu(masked_L1_comp_prev_adv + B1)
    L2_comp_prev_adv = tf.matmul(L1_comp_adv, W2)
    masked_L2_comp_prev_adv = L2_comp_prev_adv * mask_2
    L2_comp_adv = tf.nn.relu(masked_L2_comp_prev_adv + B2)
    model_comp_adv = tf.matmul(L2_comp_adv, W3) + B3
    '''''''''
    '''''''''
    diff = tf.norm(model_comp - model_comp_adv, 2) 
    grad_on_diff = tf.gradients(diff, [masked_L1_comp_prev_adv, masked_L2_comp_prev_adv]) 
    adv_feat_1 = tf.reduce_sum(tf.abs(grad_on_diff[0]), axis=0) * mask_1
    adv_feat_2 = tf.reduce_sum(tf.abs(grad_on_diff[1]), axis=0) * mask_2

    pruning_rate = pruning_rate_per_layer / iter_num
    tmp_mask_1 = utils.mask_vec(adv_feat_1, pruning_rate) * mask_1
    tmp_mask_2 = utils.mask_vec(adv_feat_2, pruning_rate) * mask_2
    return pruning_rate_per_layer, tmp_mask_1, tmp_mask_2, i+1

def IAFD(pruning_rate_per_layer):
    _, mask_1, mask_2, _ = tf.while_loop(_cond, _body, (pruning_rate_per_layer, tf.ones([L1.shape[1]]), tf.ones([L2.shape[1]]), 0), back_prop=False)
    L1_af_prev = L1_prev * mask_1
    L1_af = tf.nn.relu(L1_af_prev + B1)
    L2_af = tf.nn.relu(tf.matmul(L1_af, W2) * mask_2 + B2)
    model_ia = tf.matmul(L2_af, W3) + B3
    return model_ia

'''''''''
Model creation
'''''''''
model_ap = MBAP(rate_place_holder, is_first, is_last)
#model_gb = GBAP(rate_place_holder)
#model_mb = MBFD(rate_place_holder)
model_af = AFD(rate_place_holder, is_first, is_last)
model_rd = RFD(rate_place_holder, is_first, is_last)
#model_mf = MFD(rate_place_holder)
model_ia = IAFD(rate_place_holder)
'''''''''
'''''''''



init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
is_correct_ap = tf.equal(tf.argmax(model_ap, 1), tf.argmax(Y, 1))
#is_correct_mb = tf.equal(tf.argmax(model_mb, 1), tf.argmax(Y, 1))
is_correct_af = tf.equal(tf.argmax(model_af, 1), tf.argmax(Y, 1))
is_correct_rd = tf.equal(tf.argmax(model_rd, 1), tf.argmax(Y, 1))
#is_correct_mf = tf.equal(tf.argmax(model_mf, 1), tf.argmax(Y, 1))
#is_correct_gb = tf.equal(tf.argmax(model_gb, 1), tf.argmax(Y, 1))
is_correct_ia = tf.equal(tf.argmax(model_ia, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
accuracy_ap = tf.reduce_mean(tf.cast(is_correct_ap, tf.float32))
#accuracy_mb = tf.reduce_mean(tf.cast(is_correct_mb, tf.float32))
accuracy_af = tf.reduce_mean(tf.cast(is_correct_af, tf.float32))
accuracy_rd = tf.reduce_mean(tf.cast(is_correct_rd, tf.float32))
#accuracy_mf = tf.reduce_mean(tf.cast(is_correct_mf, tf.float32))
#accuracy_gb = tf.reduce_mean(tf.cast(is_correct_gb, tf.float32))
accuracy_ia = tf.reduce_mean(tf.cast(is_correct_ia, tf.float32))
X_comp, Y_comp = mnist.train.next_batch(55000)

batch_size = 128
total_batch = int(mnist.train.num_examples / batch_size)

rate_axis = range(0,100,5)
acc_base = []
acc_leg_ap_first = []
acc_leg_ap_last = []
acc_leg_ap_even = []
acc_leg_mb = []
acc_leg_af_gra_first = []
acc_leg_af_gra_last = []
acc_leg_af_gra_even = []
acc_leg_af_mag = []
acc_leg_mf = []
acc_leg_gb = []
acc_leg_ia = []
acc_leg_rd_first = []
acc_leg_rd_last = []
acc_leg_rd_even = []

for epoch in range(30):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.8})
        total_cost += cost_val

    print('Epoch:', '%04d' % (epoch + 1),
          'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))

print('Training completed!')

base_acc = sess.run(accuracy,feed_dict={X: mnist.test.images,
                                   keep_prob: 1.,
                                   Y: mnist.test.labels})
print('Acc on legitimate:', base_acc)

print('Make adversarial test sets')
XADV, YADV = sess.run([xadv, yadv], feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1.})

for j in range(len(XADV)):
    acc_leg_base = sess.run(accuracy,
                            feed_dict={X: XADV[j],
                                       Y: YADV,
                                       keep_prob: 1.})
    print('Acc on adversarial examples:', acc_leg_base)
    for i in range(20):
        acc_base.append(acc_leg_base)
    for i in range(20):
        acc_leg_ap_first.append(sess.run(accuracy_ap,
                                feed_dict={X: XADV[j],
                                           Y: YADV,
                                           keep_prob: 1.,
                                           is_first: True,
                                           is_last: False,
                                           rate_place_holder: i*5}))
        print('Acc MBAP on legitimate, pruning rate: %d:'%(i*5), acc_leg_ap_first[i])
    for i in range(20):
        acc_leg_ap_last.append(sess.run(accuracy_ap,
                                feed_dict={X: XADV[j],
                                           Y: YADV,
                                           keep_prob: 1.,
                                           is_first: False,
                                           is_last: True,
                                           rate_place_holder: i*5}))
        print('Acc MBAP on legitimate, pruning rate: %d:'%(i*5), acc_leg_ap_last[i])
    for i in range(20):
        acc_leg_ap_even.append(sess.run(accuracy_ap,
                                feed_dict={X: XADV[j],
                                           Y: YADV,
                                           keep_prob: 1.,
                                           is_first: True,
                                           is_last: True,
                                           rate_place_holder: i*5}))
        print('Acc MBAP on legitimate, pruning rate: %d:'%(i*5), acc_leg_ap_even[i])
    #for i in range(20):
    #    acc_leg_mb.append(sess.run(accuracy_mb,
    #                            feed_dict={X: XADV[j],
    #                                       Y: YADV,
    #                                       keep_prob: 1.,
    #                                       rate_place_holder: i*5}))
    #    print('Acc MBFD on legitimate, pruning rate: %d:'%(i*5), acc_leg_mb[i])
    for i in range(20):
        acc_leg_af_gra_first.append(sess.run(accuracy_af,
                                feed_dict={X: XADV[j],
                                           Y: YADV,
                                           is_grad_compare: True,
                                           is_first: True,
                                           is_last: False,
                                           X_small: X_comp,
                                           Y_small: Y_comp,
                                           keep_prob: 1.,
                                           rate_place_holder: i*5}))
        print('Acc AFD-gradients on legitimate, pruning rate: %d:'%(i*5), acc_leg_af_gra_first[i]) 
    for i in range(20):
        acc_leg_af_gra_last.append(sess.run(accuracy_af,
                                feed_dict={X: XADV[j],
                                           Y: YADV,
                                           is_grad_compare: True,
                                           is_first: False,
                                           is_last: True,
                                           X_small: X_comp,
                                           Y_small: Y_comp,
                                           keep_prob: 1.,
                                           rate_place_holder: i*5}))
        print('Acc AFD-gradients on legitimate, pruning rate: %d:'%(i*5), acc_leg_af_gra_last[i])
    for i in range(20):
        acc_leg_af_gra_even.append(sess.run(accuracy_af,
                                feed_dict={X: XADV[j],
                                           Y: YADV,
                                           is_grad_compare: True,
                                           is_first: True,
                                           is_last: True,
                                           X_small: X_comp,
                                           Y_small: Y_comp,
                                           keep_prob: 1.,
                                           rate_place_holder: i*5}))
        print('Acc AFD-gradients on legitimate, pruning rate: %d:'%(i*5), acc_leg_af_gra_even[i]) 
    #for i in range(20):
    #    acc_leg_af_mag.append(sess.run(accuracy_af,
    #                            feed_dict={X: XADV[j],
    #                                       Y: YADV,
    #                                       is_grad_compare: False,
    #                                       X_small: X_comp,
    #                                       Y_small: Y_comp,
    #                                       keep_prob: 1.,
    #                                       rate_place_holder: i*5}))
    #    print('Acc AFD-magnitude on legitimate, pruning rate: %d:'%(i*5), acc_leg_af_mag[i]) 
    #for i in range(20):
    #    acc_leg_mf.append(sess.run(accuracy_mf,
    #                            feed_dict={X: XADV[j],
    #                                       Y: YADV,
    #                                       is_grad_compare: False,
    #                                       X_small: X_comp,
    #                                       Y_small: Y_comp,
    #                                       keep_prob: 1.,
    #                                       rate_place_holder: i*5}))
    #    print('Acc MFD on legitimate, pruning rate: %d:'%(i*5), acc_leg_mf[i])
    #for i in range(20):
    #    acc_leg_gb.append(sess.run(accuracy_gb,
    #                            feed_dict={X: XADV[j],
    #                                       Y: YADV,
    #                                       is_grad_compare: False,
    #                                       X_small: X_comp,
    #                                       Y_small: Y_comp,
    #                                       keep_prob: 1.,
    #                                       rate_place_holder: i*5}))
    #    print('Acc MFD on legitimate, pruning rate: %d:'%(i*5), acc_leg_gb[i])
    for i in range(20):
        acc_leg_rd_first.append(sess.run(accuracy_rd,
                                feed_dict={X: XADV[j],
                                           Y: YADV,
                                           keep_prob: 1.,
                                           is_first: True,
                                           is_last: False,
                                           rate_place_holder: i*5}))
        print('Acc RFD on legitimate, pruning rate: %d:'%(i*5), acc_leg_rd_first[i]) 
    for i in range(20):
        acc_leg_rd_last.append(sess.run(accuracy_rd,
                                feed_dict={X: XADV[j],
                                           Y: YADV,
                                           keep_prob: 1.,
                                           is_first: False,
                                           is_last: True,
                                           rate_place_holder: i*5}))
        print('Acc RFD on legitimate, pruning rate: %d:'%(i*5), acc_leg_rd_last[i]) 
    for i in range(20):
        acc_leg_rd_even.append(sess.run(accuracy_rd,
                                feed_dict={X: XADV[j],
                                           Y: YADV,
                                           is_first: True,
                                           is_last: True,
                                           keep_prob: 1.,
                                           rate_place_holder: i*5}))
        print('Acc RFD on legitimate, pruning rate: %d:'%(i*5), acc_leg_rd_even[i]) 
    for i in range(20):
        acc_leg_ia.append(sess.run(accuracy_ia,
                                feed_dict={X: XADV[j],
                                           Y: YADV,
                                           X_small: X_comp,
                                           Y_small: Y_comp,
                                           keep_prob: 1.,
                                           rate_place_holder: i*5}))
        print('Acc MFD on legitimate, pruning rate: %d:'%(i*5), acc_leg_ia[i])

'''''''''
Graph settings
'''''''''
x_axis = rate_axis
for i in range(len(epsilon)-1):
    fig = plt.figure()
    graph_base = fig.add_subplot(2,1,1)
    graph_adv = fig.add_subplot(2,1,2)
    y_0 = acc_base[0:len(x_axis)]
    y_1_first = acc_leg_ap_first[0:len(x_axis)]
    y_1_last = acc_leg_ap_last[0:len(x_axis)]
    y_1_even = acc_leg_ap_even[0:len(x_axis)]
    #y_2 = acc_leg_mb[0:len(x_axis)]
    y_3_first = acc_leg_af_gra_first[0:len(x_axis)]
    y_3_last = acc_leg_af_gra_last[0:len(x_axis)]
    y_3_even = acc_leg_af_gra_even[0:len(x_axis)]
    #y_4 = acc_leg_af_mag[0:len(x_axis)]
    #y_5 = acc_leg_mf[0:len(x_axis)]
    y_6_first = acc_leg_rd_first[0:len(x_axis)]
    y_6_last = acc_leg_rd_last[0:len(x_axis)]
    y_6_even = acc_leg_rd_even[0:len(x_axis)]
    #y_7 = acc_leg_gb[0:len(x_axis)]
    y_8 = acc_leg_ia[0:len(x_axis)]

    idx_start = (i+1)*len(x_axis)
    idx_end = (i+2)*len(x_axis)
    y_adv_0 = acc_base[idx_start:idx_end]
    y_adv_1_first = acc_leg_ap_first[idx_start:idx_end]
    y_adv_1_last = acc_leg_ap_last[idx_start:idx_end]
    y_adv_1_even = acc_leg_ap_even[idx_start:idx_end]
    #y_adv_2 = acc_leg_mb[idx_start:idx_end]
    y_adv_3_first = acc_leg_af_gra_first[idx_start:idx_end]
    y_adv_3_last = acc_leg_af_gra_last[idx_start:idx_end]
    y_adv_3_even = acc_leg_af_gra_even[idx_start:idx_end]
    #y_adv_4 = acc_leg_af_mag[idx_start:idx_end]
    #y_adv_5 = acc_leg_mf[idx_start:idx_end]
    y_adv_6_first = acc_leg_rd_first[idx_start:idx_end]
    y_adv_6_last = acc_leg_rd_last[idx_start:idx_end]
    y_adv_6_even = acc_leg_rd_even[idx_start:idx_end]
    #y_adv_7 = acc_leg_gb[idx_start:idx_end]
    y_adv_8 = acc_leg_ia[idx_start:idx_end]

    graph_base.plot(x_axis, y_1_first, label='activation pruning - first layer')
    graph_base.plot(x_axis, y_1_last, label='activation pruning - last layer')
    graph_base.plot(x_axis, y_1_even, label='activation pruning - both layer')
    #graph_base.plot(x_axis, y_2, label='magnitude based feature drop')
    graph_base.plot(x_axis, y_3_first, label='adversarial feature drop, gradients base - first layer')
    graph_base.plot(x_axis, y_3_last, label='adversarial feature drop, gradients base - last layer')
    graph_base.plot(x_axis, y_3_even, label='adversarial feature drop, gradients base - both layer')
    #graph_base.plot(x_axis, y_4, label='adversarial feature drop, magnitude gap base')
    #graph_base.plot(x_axis, y_5, label='mild adversarial feature drop')
    graph_base.plot(x_axis, y_6_first, label='random feature drop - firtst')
    graph_base.plot(x_axis, y_6_last, label='random feature drop - last')
    graph_base.plot(x_axis, y_6_even, label='random feature drop - even')
    #graph_base.plot(x_axis, y_7, label='gradients based activation pruning')
    graph_base.plot(x_axis, y_8, label='iterative gradients based feature drop - both layer')
    graph_base.plot(x_axis, y_0, '--', label='base')
    graph_base.set_xlabel('Pruning rate')
    graph_base.set_ylabel('Accuracy in clean MNIST')

    graph_adv.plot(x_axis, y_adv_1_first, label='activation pruning - first layer')
    graph_adv.plot(x_axis, y_adv_1_last,  label='activation pruning - last layer ')
    graph_adv.plot(x_axis, y_adv_1_even,  label='activation pruning - both layer ')
    #graph_adv.plot(x_axis, y_adv_2, label='magnitude based feature drop')
    graph_adv.plot(x_axis, y_adv_3_first, label='adversarial feature drop, gradients base - first layer')
    graph_adv.plot(x_axis, y_adv_3_last,  label='adversarial feature drop, gradients base - last layer ')
    graph_adv.plot(x_axis, y_adv_3_even,  label='adversarial feature drop, gradients base - both layer ')
    #graph_adv.plot(x_axis, y_adv_4, label='adversarial feature drop, magnitude gap base')
    #graph_adv.plot(x_axis, y_adv_5, label='mild adversarial feature drop')
    graph_adv.plot(x_axis, y_adv_6_first, label='random feature drop - first layer')
    graph_adv.plot(x_axis, y_adv_6_last,  label='random feature drop - last layer ')
    graph_adv.plot(x_axis, y_adv_6_even,  label='random feature drop - both layer ')
    #graph_adv.plot(x_axis, y_adv_7, label='gradients based activation pruning')
    graph_adv.plot(x_axis, y_adv_8, label='iterative gradients based feature drop - both layer')
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
