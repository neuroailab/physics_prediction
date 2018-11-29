import os
import numpy as np
import sys
sys.path.append('../models')
import interaction_model as modelsource
import cPickle
import tensorflow as tf
import copy

# Define some constant values

def table_norot_grab_func(path, SEED=0):
    all_filenames = os.listdir(path)
    rng = np.random.RandomState(seed=SEED)
    rng.shuffle(all_filenames)
    print('got to file grabber!')
    return [os.path.join(path, fn) for fn in all_filenames \
            if fn.endswith('.tfrecords')]
            #and fn.startswith('2:')] \
            #and 'FAST_LIFT' in fn]

def return_outputs(inputs, outputs, targets, **kwargs):
    retval = {}
    for target in targets:
        retval[target] = outputs[target]
    return retval

def combine_interaction_data(
        data_paths, nums_examples, group_paths, 
        ):
    data = []
    curr_indx = 0
    for data_path, num_examples, group_path in \
            zip(data_paths, nums_examples, group_paths):
        if not group_path is None:
            group_data = {'which_dataset': curr_indx}
        else:
            group_data = {}
        data.append((data_path, num_examples, group_data))
        curr_indx +=1
    return data

def moving_filter_func(data, keys, is_moving='is_moving'):
    assert all(k in keys for k in [is_moving]), keys
    return data[is_moving]

def moving_not_acting_filter_func(data, keys, is_moving='is_moving'):
    assert all(k in keys for k in [is_moving, 'is_acting']), keys
    return tf.logical_and(moving_filter_func(data, keys, is_moving=is_moving),
            tf.logical_not(data['is_acting']))

def first_object_moving_and_there_filter_func(data, keys, is_moving='is_moving'):
    if len(data['is_object_in_view'].get_shape().as_list()) == 3:
        data['is_object_in_view'] = tf.expand_dims(data['is_object_in_view'],
                axis=2)
    assert all(k in keys for k in [is_moving, 'is_object_in_view']), keys
    return tf.logical_and(data[is_moving],
            data['is_object_in_view'][:, :, :, 0])

def first_moving_there_not_acting_filter_func(data, keys, is_moving='is_moving'):
    assert all(k in keys for k in [is_moving, 'is_object_in_view',
        'is_acting']), keys
    return tf.logical_and(
            first_object_moving_and_there_filter_func(data, keys,
                is_moving=is_moving),
            tf.logical_not(data['is_acting']))
