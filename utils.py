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

def mask_total(vecs, pruning_rate):
    lens = [vecs[i].get_shape().as_list()[0] for i in range(len(vecs))]
    max_len = max(lens)
    padded_vecs = [tf.pad(vecs[i], [[0, max_len - lens[i]]]) for i in range(len(vecs))]
    tmp_vec = tf.concat(padded_vecs, 0)
    vec_minus_abs = tf.abs(tmp_vec) * -1
    num_elements = sum(lens) 
    num_pad = tmp_vec.shape[0] - num_elements
    num = num_elements * (100 - pruning_rate) / 100
    th_val, th_idx = tf.nn.top_k(vec_minus_abs, num + num_pad )
    mask_bool = vec_minus_abs >= th_val[num + num_pad - 1]
    mask_float = tf.cast(mask_bool, tf.float32) 
    mask = [mask_float[i * max_len:i * max_len + lens[i]] for i in range(len(vecs))]
    return mask

def random_vector(*activations):
    ran_vecs = []
    for v in activations:
       ran_vec = tf.random.uniform([v.get_shape().as_list()[1]])
       ran_vecs.append(ran_vec)
    return ran_vecs 
