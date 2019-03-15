import tensorflow as tf
import numpy as np

#pruning smaller than a threshold
def prune(tensor, pruning_rate):
    tensor_abs = tf.abs(tensor)
    num_elements = tensor.get_shape().as_list()[1] 
    num = tf.cast(num_elements * (100 - pruning_rate) / 100, tf.int32)
    th_val, th_idx = tf.nn.top_k(tensor_abs, num)
    tmp_th = th_val[:, num-1]
    th = tmp_th[..., tf.newaxis]
    mask_bool = tensor_abs >= th
    mask_float = tf.cast(mask_bool, tf.float32) 
    pruned_tensor = tensor * mask_float
    return pruned_tensor, mask_float

def prune_conv_feature(tensor, pruning_rate):
    shape_as_list = tensor.get_shape().as_list()
    reshaped_tensor = tf.reshape(tensor, [-1, shape_as_list[1] * shape_as_list[2] * shape_as_list[3]])
    tmp_tensor, tmp_mask = prune(reshaped_tensor, pruning_rate)
    pruned_tensor = tf.reshape(tmp_tensor, [-1, shape_as_list[1], shape_as_list[2], shape_as_list[3]])
    pruned_mask = tf.reshape(tmp_mask, [-1, shape_as_list[1], shape_as_list[2], shape_as_list[3]])
    return pruned_tensor, pruned_mask

#pruning larger than a threshold.
def mask_vec(vec, pruning_rate):
    vec_minus_abs = tf.abs(vec) * -1
    num_elements = vec.get_shape().as_list()[0] 
    num = tf.cast(num_elements * (100 - pruning_rate) / 100, tf.int32)
    th_val, th_idx = tf.nn.top_k(vec_minus_abs, num)
    mask_bool = vec_minus_abs >= th_val[num-1]
    mask_float = tf.cast(mask_bool, tf.float32) 
    return mask_float

def mask_map(m, pruning_rate):
    shape = m.get_shape()
    tmp = tf.reshape(m, [-1])
    map_minus_abs = tf.abs(tmp) * -1
    num_elements = tmp.get_shape().as_list()[0] 
    num = tf.cast(num_elements * (100 - pruning_rate) / 100, tf.int32)
    th_val, th_idx = tf.nn.top_k(map_minus_abs, num)
    mask_bool = map_minus_abs >= th_val[num-1]
    mask_float = tf.cast(mask_bool, tf.float32) 
    mask_m = tf.reshape(mask_float, shape)
    return mask_m

def mask_total(vecs, pruning_rate):
    lens = [vecs[i].get_shape().as_list()[0] for i in range(len(vecs))]
    max_len = max(lens)
    padded_vecs = [tf.pad(vecs[i], [[0, max_len - lens[i]]]) for i in range(len(vecs))]
    tmp_vec = tf.concat(padded_vecs, 0)
    vec_minus_abs = tf.abs(tmp_vec) * -1
    num_elements = sum(lens) 
    num_pad = tmp_vec.shape[0] - num_elements
    num = tf.cast(num_elements * (100 - pruning_rate) / 100, tf.int32)
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

def random_map(*maps):
    ran_maps = []
    for m in maps:
        shape = m.get_shape()
        ran_map = tf.random.uniform(shape[1:])
        ran_maps.append(ran_map)
    return ran_maps 

#return numpy array
def avg_of_lists(list_of_lists):
    np_lists = np.array(list_of_lists)
    np_mean = np.mean(np_lists, 0)
    return np_mean

def shuffle_and_devide_into_batches(num, data, labels):
    db = {}
    db['data'] = []
    db['labels'] = []
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    for i in range(int(len(data) / num)):
        db['data'].append(data[idx[i*num:(i+1)*num]])
        db['labels'].append(labels[idx[i*num:(i+1)*num]])
    return db['data'], db['labels']
