import numpy as np
import json
import tensorflow as tf
import os
import cPickle
import pdb
import sys
import copy

from utils import Filter

class SequenceNewDataProvider(object):
    '''
    Sequence data provider, outputs a sequence of data of the requested length.
    This data provider supports data filtering 
    This data provider uses new dataset interface in tensorflow
    '''
    def __init__(self,
            data,
            enqueue_batch_size,
            sources,
            sequence_len,
            buffer_size,
            is_training=True,
            delta_time=1,
            filter_rule=None,
            resize=None,
            augment=None,
            file_pattern='*.tfrecords',
            shuffle_seed=None,
            shuffle_queue=True,
            map_pcall_num=48,
            *args,
            **kwargs):
        self.data = data
        self.enqueue_batch_size = enqueue_batch_size
        self.sources = sources
        self.sequence_len = sequence_len
        self.delta_time = delta_time
        self.filter_rule = filter_rule
        self.resize = resize
        self.augment = augment
        self.map_pcall_num = map_pcall_num
        self.is_training = is_training
        self.shuffle_queue = shuffle_queue
        self.file_pattern = file_pattern
        self.shuffle_seed = shuffle_seed
        self.buffer_size = buffer_size
        self.all_sources = copy.deepcopy(sources)

        assert self.delta_time >= 1, \
                ('delta time has to be at least 1')
        assert self.sequence_len >= 1, \
                ('sequence length has to be at least 1')
        assert self.enqueue_batch_size >= self.sequence_len * self.delta_time, \
                ('batch size has to be at least equal to sequence length ' + \
                'times delta time')

    # make it each example wise, rather than batch wise
    def apply_filter(self, data):
        sequence_len = tf.constant(self.sequence_len, dtype = tf.int32)
        for f in self.filter.keys:
            data[f] = tf.cast(data[f], tf.bool)
            # Add the batch dimension
            data[f] = tf.expand_dims(data[f], axis=0)
        # combine filters according to specified filter rule
        master_filter = self.filter.eval(data)
        # check if ALL binary labels within sequence are not zero
        master_filter_sum = tf.reduce_sum(tf.cast(master_filter, tf.int32))
        # gather positive examples for each data entry
        return tf.equal(master_filter_sum, sequence_len)

    def enqueue_many_func(self, all_tensors):
        return tf.data.Dataset.zip(
                {key: tf.data.Dataset.from_tensor_slices(value) 
                    for key, value in all_tensors.items()})

    def postproc_each(self, str_loaded, source):
        keys_to_features = {
	        source: tf.FixedLenFeature((), tf.string, ''),} 
        parsed = tf.parse_single_example(str_loaded, keys_to_features)
        str_loaded = parsed[source]
        if str_loaded.dtype is tf.string:
            curr_meta = self.meta_dict[source][source]
            curr_data = tf.decode_raw(str_loaded, curr_meta['rawtype'])
            curr_data = tf.reshape(curr_data, curr_meta['rawshape'])
        else:
            curr_data = str_loaded

        if curr_data.dtype==tf.int16:
            curr_data = tf.cast(curr_data, tf.int32)

        return curr_data

    def postproc(self, all_keys):
        ret_dict = {}
        for source,str_loaded in all_keys.items():
            curr_data = self.postproc_each(str_loaded, source)
            ret_dict[source] = curr_data
        return ret_dict

    def get_tfr_filenames(self, folder_name, file_pattern='*.tfrecords'):
        # Get list of tfrecord filenames for given folder
        tfrecord_pattern = os.path.join(folder_name, file_pattern)
        datasource = tf.gfile.Glob(tfrecord_pattern)
        datasource.sort()

        return datasource

    def parse_standard_tfmeta(self, path_dict):
        meta_dict = {}
        for source in path_dict:
            path = path_dict[source]
            if isinstance(path, str):
                if path.startswith('meta') and path.endswith('.pkl'):
                    mpaths = [path]
                else:
                    assert os.path.isdir(path)
                    mpaths = filter(
                            lambda x: x.startswith('meta') \
                                    and x.endswith('.pkl'),
                            os.listdir(path))
                    mpaths = [os.path.join(path, mp) for mp in mpaths]
            else:
                # in this case, it's a list
                assert isinstance(path, list)
                mpaths = path
            d = {}
            for mpath in mpaths:
                d.update(cPickle.load(open(mpath)))
            meta_dict[source] = d
        return meta_dict

    def set_data_shape(self, data):
        shape = data.get_shape().as_list()
        shape[0] = self.enqueue_batch_size
        for s in shape:
            assert s is not None, ("Unknown shape", shape)
        data.set_shape(shape)
        return data

    def create_data_sequence(self, data):
        if self.delta_time==1:
            long_len = self.sequence_len
            min_len = self.sequence_len
            data = tf.expand_dims(data, 1)
            data_shape = data.get_shape().as_list()
            data_type = data.dtype
            data_augmented = tf.concat(
                    [data, 
                     tf.zeros(
                         [long_len - min_len] + data_shape[1:], 
                         dtype=data_type)], 
                    axis = 0)
            shift_len = self.enqueue_batch_size - (min_len - 1)
            shifts = [data_augmented[i : i + shift_len] \
                    for i in range(long_len)]
            return tf.concat(shifts, 1)
        else:
            data = tf.expand_dims(data, 0)
            sequences = [data[:, i : i+self.sequence_len*self.delta_time : \
                    self.delta_time] for i in \
                    range(self.enqueue_batch_size - (self.sequence_len - 1) * \
                        self.delta_time)]
            return tf.concat(sequences, axis = 0)

    def build_one_dataset(self, curr_data):
        # Unpack the data related info, num_examples is not used
        curr_data_path, _, extra_tensors = curr_data

        # Dictionary with keys being source, and values being directories
        self.source_paths = { 
                source: os.path.join(curr_data_path, source) \
                for source in self.sources }

        # load filters, add that to source_paths
        if self.filter_rule:
            self.filter = Filter(self.filter_rule)
            for f in self.filter.keys:
                self.source_paths[f] = os.path.join(curr_data_path, f)
                if f not in self.all_sources:
                    self.all_sources.append(f)
        else:
            self.filter = None

        # load metas
        self.meta_dict = self.parse_standard_tfmeta(self.source_paths)

        # Get tfr filenames
        source_lists = {
                source: self.get_tfr_filenames(
                    self.source_paths[source], 
                    file_pattern=self.file_pattern) \
                for source in self.source_paths}

        # This shuffle needs to be False to keep the order of every attribute
        # the same
        file_datasets = {
                source: tf.data.Dataset.list_files(curr_files, shuffle=False) \
                for source, curr_files in source_lists.items()}

        if self.is_training:
            # Shuffle file names using the same shuffle_seed
            file_datasets = {
                    source: curr_dataset.shuffle(
                        buffer_size=len(source_lists.values()[0]), 
                        seed=self.shuffle_seed).repeat() \
                    for source,curr_dataset in file_datasets.items()}

        # Create dataset for both
	def _fetch_dataset(filename):
	    buffer_size = 8 * 1024 * 1024     # 8 MiB per file
	    dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
	    return dataset

        each_dataset = {
                source: curr_dataset.apply(
                    tf.contrib.data.parallel_interleave(
                        _fetch_dataset, 
                        cycle_length=1, 
                        sloppy=False)) \
                for source,curr_dataset in file_datasets.items()
                }

        # Decode raw first before zip
        each_dataset = {
                source: curr_dataset.map(
                    lambda x: self.postproc_each(x, source),
                    num_parallel_calls=self.map_pcall_num,
                    ) \
                for source, curr_dataset in each_dataset.items()
                }

        # Zip, repeat, batch
        zip_dataset = tf.data.Dataset.zip(each_dataset)
        zip_dataset = zip_dataset.repeat()
        zip_dataset = zip_dataset.batch(self.enqueue_batch_size)

        # Set shape (first dimension to be batchsize)
        zip_dataset = zip_dataset.map(
                lambda x: {
                    key: self.set_data_shape(value) 
                    for key,value in x.items()}, 
                num_parallel_calls=self.map_pcall_num)

        # Create sequence for each dataset
        zip_dataset = zip_dataset.map(
                lambda x: {
                    key: self.create_data_sequence(value) 
                    for key, value in x.items()}, 
                num_parallel_calls=self.map_pcall_num)

        # Add extra tensors
        def add_extra_tensors(value):
            for extra_key, extra_tensor in extra_tensors.items():
                assert extra_key not in value
                batch_size = value[value.keys()[0]].get_shape().as_list()[0]
                time = value[value.keys()[0]].get_shape().as_list()[1]
                extra_tensor = tf.constant(extra_tensor, dtype=tf.float32)
                extra_shape = extra_tensor.get_shape().as_list()
                value[extra_key] = tf.tile(
                        tf.reshape(
                            extra_tensor,
                            [1, 1] + extra_shape),
                        [batch_size, time] + [1] * len(extra_shape))
                if extra_key not in self.all_sources:
                    self.all_sources.append(extra_key)
            return value
        zip_dataset = zip_dataset.map(
                add_extra_tensors,
                num_parallel_calls=self.map_pcall_num)

        return zip_dataset

    def get_max_shapes(self, zip_datasets):
        max_shapes = dict([(source, [0]) for source in self.all_sources])

        for each_dataset in zip_datasets:
            curr_shapes = each_dataset.output_shapes
            for source, curr_shape in curr_shapes.items():
                curr_shape = curr_shape.as_list()
                while len(max_shapes[source]) < len(curr_shape):
                    max_shapes[source].append(0)

                max_shapes[source] = list(np.maximum( \
                        max_shapes[source], \
                        curr_shape))

        return max_shapes

    def pad_tensors(self, zip_datasets):
        max_shapes = self.get_max_shapes(zip_datasets)

        def _pad_up_to_using_0(tensor, max_shape):
            shape = tensor.get_shape().as_list()
            paddings = [[0, m - shape[i]] if m is not None else [0, 0] \
                    for (i, m) in enumerate(max_shape)]
            return tf.pad(
                    tensor, paddings, 'CONSTANT', \
                    constant_values=0)

        def _pad_to_max_shapes(value):
            for source, max_shape in max_shapes.items():
                mask_key = source + '_mask'
                assert mask_key not in value
                value[mask_key] = _pad_up_to_using_0(
                        tf.ones(tf.shape(value[source]), dtype=tf.bool),
                        max_shape)
                value[mask_key].set_shape(max_shape)
                value[source] = _pad_up_to_using_0(value[source], max_shape)
                value[source].set_shape(max_shape)

                if mask_key not in self.all_sources:
                    self.all_sources.append(mask_key)
            return value
        
        for idx in range(len(zip_datasets)):
            zip_datasets[idx] = zip_datasets[idx].map(
                    _pad_to_max_shapes,
                    num_parallel_calls=self.map_pcall_num)
        return zip_datasets

    def concate_datasets(self, zip_datasets):
        zip_dataset = tf.data.Dataset.zip(tuple(zip_datasets))

        def _concate(*value):
            new_value = {}
            for source in self.all_sources:
                new_value[source] = []
                for _each_value in value:
                    new_value[source].append(_each_value[source])
                new_value[source] = tf.concat(new_value[source], axis=0)
            return new_value
        zip_dataset = zip_dataset.map(
                _concate,
                num_parallel_calls=self.map_pcall_num)
        return zip_dataset

    def build_datasets(self):
        # Build dataset for every data path
        zip_datasets = [
                self.build_one_dataset(curr_data)\
                for curr_data in self.data]

        # Pad and concatenate
        zip_datasets = self.pad_tensors(zip_datasets)
        zip_dataset = self.concate_datasets(zip_datasets)

        # "Enqueue_many" it, shuffle it
        zip_dataset = zip_dataset.flat_map(self.enqueue_many_func)
        # Apply filters
        if self.filter:
            zip_dataset = zip_dataset.filter(self.apply_filter)
        if self.is_training and self.shuffle_queue:
            # Shuffle it
            zip_dataset = zip_dataset.shuffle(
                    buffer_size=self.buffer_size, 
                    seed=None,
                    )
        # Batch it again
        zip_dataset = zip_dataset.apply(
                tf.contrib.data.batch_and_drop_remainder(self.model_batch_size))
        zip_dataset = zip_dataset.prefetch(2)

        return zip_dataset

    # entry point for TFUtils
    def input_fn(self, batch_size, params=None, **kwargs):
        self.model_batch_size = batch_size
        zip_dataset = self.build_datasets()
        zip_iter = zip_dataset.make_one_shot_iterator()
        input_dict = zip_iter.get_next()
        return input_dict
