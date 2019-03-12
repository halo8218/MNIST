import tensorflow as tf
import numpy as np

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

rate_place_holder = tf.placeholder(tf.int32, [])

W1 = tf.Variable(tf.random_normal([784, 512], stddev=0.01))
B1 = tf.Variable(tf.zeros([512]))
L1 = tf.nn.relu(tf.matmul(X, W1) + B1)
L1 = tf.nn.dropout(L1, keep_prob)

W2 = tf.Variable(tf.random_normal([512, 256], stddev=0.01))
B2 = tf.Variable(tf.zeros([256]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + B2)
L2 = tf.nn.dropout(L2, keep_prob)

W3 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
B3 = tf.Variable(tf.zeros([10]))
model = tf.matmul(L2, W3) + B3

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

grad = tf.gradients(cost, X)
#epsilon = np.arange(0.05, 0.35, 0.05)
epsilon = [0.075]
xadv = [tf.stop_gradient(X + e*tf.sign(grad)) for e in epsilon]
xadv = [tf.clip_by_value(adv, 0., 1.) for adv in xadv]
xadv = [tf.reshape(adv,[-1,784]) for adv in xadv]
yadv = Y
'''''''''
Build replica model for comparing
'''''''''
L1_comp = tf.nn.relu(tf.matmul(X_small, W1) + B1)
L2_comp = tf.nn.relu(tf.matmul(L1_comp, W2) + B2)
model_comp = tf.matmul(L2_comp, W3) + B3

cost_comp = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model_comp, labels=Y_small))
grad_comp = tf.gradients(cost_comp, X_small)
xadv_small = tf.stop_gradient(X_small + 0.3*tf.sign(grad_comp))
xadv_small = tf.clip_by_value(xadv_small, 0., 1.)
xadv_small = tf.reshape(xadv_small, [-1, 784])

L1_comp_adv = tf.nn.relu(tf.matmul(xadv_small, W1) + B1)
L2_comp_adv = tf.nn.relu(tf.matmul(L1_comp_adv, W2) + B2)
model_comp_adv = tf.matmul(L2_comp_adv, W3) + B3

'''''''''
Magnitude Based Feature Drop Model
'''''''''
def MBAP(pruning_rate_per_layer):
    #L1_mb, _ = utils.prune(L1, pruning_rate_per_layer)
    L1_mb = L1
    L2_prev = tf.matmul(L1_mb, W2)
    pruned_L2_prev, _ = utils.prune(L2_prev, pruning_rate_per_layer)
    L2_mb = tf.nn.relu(pruned_L2_prev + B2)
    model_mb = tf.matmul(L2_mb, W3) + B3
    return model_mb

model_mb = MBAP(rate_place_holder)
#cost_mb = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model_mb, labels=Y))
'''''''''
Adversarial Feature Drop Model
'''''''''
def compare(bool_place_holder):
    diff = tf.norm(model_comp - model_comp_adv, 2) 
    grad_on_diff = tf.gradients(diff, [L1_comp_adv, L2_comp_adv]) 
    comp_vec_1 = tf.cond(bool_place_holder, lambda: tf.reduce_sum(tf.abs(grad_of_diff[0]), axis=0) lambda: tf.reduce_sum(tf.abs(L1_comp - L1_comp_adv), axis=0)
    comp_vec_2 = tf.cond(bool_place_holder, lambda: tf.reduce_sum(tf.abs(grad_of_diff[1]), axis=0) lambda: tf.reduce_sum(tf.abs(L2_comp - L2_comp_adv), axis=0)
    return comp_vec_1, comp_vec_2

def AFD(pruning_rate_per_layer):
    adv_feat_1, adv_feat_2 = compare(is_grad_compare)
    mask = utils.mask_vec(adv_feat_2, pruning_rate_per_layer)
    inv_mask = (mask - 1) * -1
    #L1_dc = L1 * utils.mask_vec(adv_feat_1, pruning_rate_per_layer)
    L1_dc = L1
    L2_dc_prev = tf.nn.relu(tf.matmul(L1_dc, W2) + B2)
    L2_dc = L2_dc_prev * mask
    L2_dc = L2_dc + tf.nn.relu(inv_mask * B2)
    model_af = tf.matmul(L2_dc, W3) + B3
    return model_af

model_af = AFD(rate_place_holder)
'''''''''
Random Feature Drop Model
'''''''''
def RAP(pruning_rate_per_layer):
    ran_feat_1, ran_feat_2 = utils.random_vector(L1, L2)
    #L1_rd = L1_rd_prev * utils.mask_vec(adv_feat_1, pruning_rate_per_layer)
    L1_rd = L1
    L2_rd_prev = tf.nn.relu(tf.matmul(L1_rd, W2) + B2)
    L2_rd = L2_rd_prev * utils.mask_vec(ran_feat_2, pruning_rate_per_layer)
    model_rd = tf.matmul(L2_rd, W3) + B3
    return model_rd

model_rd = RAP(rate_place_holder)
'''''''''
New Feature Drop Model
'''''''''
def approx_hessian_vector():
    L1_new = tf.nn.relu(tf.matmul(X_small, W1) + B1)
    L2_new = tf.nn.relu(tf.matmul(L1_new, W2) + B2)
    #model_new = tf.matmul(L2_new, W3) + B3
    #cost_new = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model_new, labels=Y_small))
    #grad_new = tf.gradients(cost_new, [L1_new, L2_new])
    #h_approx_L1 = tf.reduce_sum(tf.abs(grad_new[0]), axis=0)
    #h_approx_L2 = tf.reduce_sum(tf.abs(grad_new[1]), axis=0)
    h_approx_L1 = tf.reduce_sum(tf.abs(L1_new), axis=0)
    h_approx_L2 = tf.reduce_sum(tf.abs(L2_new), axis=0)
    return h_approx_L1, h_approx_L2

def NAP(pruning_rate_per_layer):
    h_approx_L1, h_approx_L2 = approx_hessian_vector()
    adv_feat_new_1, adv_feat_new_2 = compare()
    adjusted_feat_1 = adv_feat_new_1 / h_approx_L1
    adjusted_feat_2 = adv_feat_new_2 / h_approx_L2
    mask = utils.mask_vec(adjusted_feat_2, pruning_rate_per_layer)
    inv_mask = (mask - 1) * -1
    #L1_new = L1 * utils.mask_vec(adjusted_feat_1, pruning_rate_per_layer)
    L1_new = L1
    L2_new_prev = tf.nn.relu(tf.matmul(L1_new, W2) + B2)
    L2_new = L2_new_prev * mask
    L2_new = L2_new + tf.nn.relu(B2 * inv_mask)
    #L2_new = L2_new_prev * utils.mask_vec(adv_feat_new_2 / tf.reduce_sum(tf.abs(L2), axis=0), pruning_rate_per_layer)
    model_new = tf.matmul(L2_new, W3) + B3
    return model_new

model_new = NAP(rate_place_holder)
'''''''''
'''''''''



init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 128
total_batch = int(mnist.train.num_examples / batch_size)
print(mnist.train.num_examples)

for epoch in range(30):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.8})
        total_cost += cost_val

    print('Epoch:', '%04d' % (epoch + 1),
          'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))

print('Training completed!')

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
is_correct_mb = tf.equal(tf.argmax(model_mb, 1), tf.argmax(Y, 1))
is_correct_dc = tf.equal(tf.argmax(model_af, 1), tf.argmax(Y, 1))
is_correct_rd = tf.equal(tf.argmax(model_rd, 1), tf.argmax(Y, 1))
is_correct_new = tf.equal(tf.argmax(model_new, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
accuracy_mb = tf.reduce_mean(tf.cast(is_correct_mb, tf.float32))
accuracy_dc = tf.reduce_mean(tf.cast(is_correct_dc, tf.float32))
accuracy_rd = tf.reduce_mean(tf.cast(is_correct_rd, tf.float32))
accuracy_new = tf.reduce_mean(tf.cast(is_correct_new, tf.float32))
X_comp, Y_comp = mnist.train.next_batch(55000)

print('Acc on legitimate:', sess.run(accuracy,
                        feed_dict={X: mnist.test.images,
                                   keep_prob: 1.,
                                   Y: mnist.test.labels}))
for i in range(20):
    print('Acc MBAP on legitimate, pruning rate: %d:'%(80+i), sess.run(accuracy_mb,
                            feed_dict={X: mnist.test.images,
                                       Y: mnist.test.labels,
                                       keep_prob: 1.,
                                       rate_place_holder: 80+i}))
for i in range(50):
    print('Acc AFD on legitimate, pruning rate: %d:'%(1+i), sess.run(accuracy_dc,
                            feed_dict={X: mnist.test.images,
                                       Y: mnist.test.labels,
                                       X_small: X_comp,
                                       Y_small: Y_comp,
                                       keep_prob: 1.,
                                       rate_place_holder: i+1}))
for i in range(50):
    print('Acc NAP on legitimate, pruning rate: %d:'%(1+i), sess.run(accuracy_new,
                            feed_dict={X: mnist.test.images,
                                       Y: mnist.test.labels,
                                       X_small: X_comp,
                                       Y_small: Y_comp,
                                       keep_prob: 1.,
                                       rate_place_holder: i+1}))
for i in range(1):
    print('Acc RAP on legitimate, pruning rate: %d:'%(10+i), sess.run(accuracy_rd,
                            feed_dict={X: mnist.test.images,
                                       Y: mnist.test.labels,
                                       keep_prob: 1.,
                                       rate_place_holder: 10+i}))
    
print('Make adversarial test sets')
XADV, YADV = sess.run([xadv, yadv], feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1.})
for i in range(len(XADV)):
    print('Acc on adversarial; no defense; epsilon = %.2f:'%epsilon[i], sess.run(accuracy,
                            feed_dict={X: XADV[i],
                                       keep_prob: 1.,
                                       Y: YADV}))
    for j in range(20):
        print('Acc on adversarial; normal activation pruning rate: %d; epsilon = %.2f:'%(80+j,epsilon[i]), sess.run(accuracy_mb,
                                feed_dict={X: XADV[i],
                                           Y: YADV,
                                           keep_prob: 1.,
                                           rate_place_holder: 80+j}))
    for j in range(50):
        print('Acc on adversarial; DeepCloak pruning rate: %d; epsilon = %.2f:'%(j+1,epsilon[i]), sess.run(accuracy_dc,
                                feed_dict={X: XADV[i],
                                           Y: YADV,
                                           X_small: X_comp,
                                           Y_small: Y_comp,
                                           keep_prob: 1.,
                                           rate_place_holder: j+1}))
    for j in range(50):
        print('Acc on adversarial; new pruning rate: %d; epsilon = %.2f:'%(j+1,epsilon[i]), sess.run(accuracy_new,
                                feed_dict={X: XADV[i],
                                           Y: YADV,
                                           X_small: X_comp,
                                           Y_small: Y_comp,
                                           keep_prob: 1.,
                                           rate_place_holder: j+1}))
    for j in range(1):
        print('Acc on adversarial; random activation pruning rate: %d; epsilon = %.2f:'%(10+j,epsilon[i]), sess.run(accuracy_rd,
                                feed_dict={X: XADV[i],
                                           Y: YADV,
                                           keep_prob: 1.0,
                                           rate_place_holder: 10+j}))

#import matplotlib
#matplotlib.use('tkagg')
#import matplotlib.pyplot as plt
#def gen_image(arr):
#    two_d = (np.reshape(arr, (28, 28)) * 255).astype(np.uint8)
#    plt.imshow(two_d, interpolation='nearest')
#    return plt
#
## Get a batch of two random images and show in a pop-up window.
#print(YADV[0])
#for i in range(len(XADV)):
#    gen_image(XADV[i][0]).show()
