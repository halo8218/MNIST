import tensorflow as tf
import numpy as np

#pruning smaller than a threshold
def prune(tensor, pruning_rate):
    tensor_abs = tf.abs(tensor)
    num_elements = tensor.shape[1] 
    num = num_elements * (100 - pruning_rate) / 100
    th_val, th_idx = tf.nn.top_k(tensor_abs, num)
    tmp_th = th_val[:, num-1]
    th = tmp_th[..., tf.newaxis]
    mask_bool = tensor_abs >= th
    mask_float = tf.cast(mask_bool, tf.float32) 
    pruned_tensor = tensor * mask_float
    return pruned_tensor, mask_float

#pruning larger than a threshold.
def mask_vec(vec, pruning_rate):
    vec_minus_abs = tf.abs(vec) * -1
    num_elements = vec.shape[0] 
    num = num_elements * (100 - pruning_rate) / 100
    th_val, th_idx = tf.nn.top_k(vec_minus_abs, num)
    mask_bool = vec_minus_abs >= th_val[num-1]
    mask_float = tf.cast(mask_bool, tf.float32) 
    return mask_float

def random_vector(*activations):
    ran_vecs = []
    for v in activations:
       ran_vec = tf.random.uniform([v.get_shape().as_list()[1]])
       ran_vecs.append(ran_vec)
    return ran_vecs 
