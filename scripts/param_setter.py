import pdb
import numpy as np
import os
import tensorflow as tf
import sys
import copy
import cPickle as pickle

import utils2 as utils

sys.path.append('../data')
sys.path.append('../models')

from new_data import SequenceNewDataProvider
import interaction_model as modelsource
import interaction_config as modelconfig


class ParameterSetter(object):
    """
    Process the command-line arguments, set up the parameters for training
    """
    EXAMPLES_PER_FILE = 1024

    def __init__(self, args):
        self.args = args
        self.exp_id = args.expId

        self._set_loss_base_params()
        self._set_multi_data()

    def _set_loss_base_params(self):
        args = self.args

        self.loss_base_params = {
                'alpha': args.alpha,
                'preserve_distance_radius': args.pdr,
                'use_running_mean': args.use_running_mean==1,
                'sl': args.seq_len,
                'add_gloss': args.add_gloss,
                'avd_obj_mask': args.avd_obj_mask,
                }

    def _set_multi_data(self):
        args = self.args

        # Get DATA_PATH from args
        data_paths = []
        group_file = []
        static_files = []

        dataset_split = args.dataset.split(',')
        group_split = args.group_file.split(',')

        assert len(group_split)==len(dataset_split), \
                "Length of group files should be the same as dataset"
        
        for each_group, dataset in zip(group_split, dataset_split):
            if not os.path.isdir(dataset) and args.dataset_search_dir:
                dataset = os.path.join(args.dataset_search_dir, dataset)
            _data_path = os.path.join(dataset, 'new_tfdata')
            assert os.path.isdir(_data_path), \
                    "Dataset %s not existing!" % dataset 
            data_paths.append(_data_path)

            if not os.path.exists(each_group):
                each_group = os.path.join(dataset, each_group)
            assert os.path.exists(each_group), \
                    "Group file %s not existing!" % each_group
            group_file.append(each_group)

            _static_file = os.path.join(dataset, 'static_particles.pkl')
            assert os.path.exists(_static_file), \
                    "Static file %s not existing!" % _static_file
            static_files.append(_static_file)

        self.data_paths = data_paths
        self.group_file = group_file
        self.static_files = static_files

    def _get_model_cfg(self):
        args = self.args

        n_classes = 3
        ctrl_dist = 4
        model_cfg = getattr(modelconfig, args.network_func)
        model_cfg = model_cfg(n_classes, ctrl_dist, nonlin=args.network_nonlin)
        return model_cfg

    def get_model_params(self):
        args = self.args

        model = modelsource.hrn
        model_cfg = self._get_model_cfg()

        model_params = {
            'func': model,
            'debug': args.debug,
            'OB1': args.OB1,
            'OB2': args.OB2,
            'cfg': model_cfg,
            'number_of_kNN': args.number_of_kNN,
            'room_center': args.room_center,
            'group_file': self.group_file,
            'use_collisions': args.with_coll,
            'use_actions': args.with_act,
            'max_collision_distance': args.max_collision_distance,
            'gravity_term': args.gravity_term,
            'add_dist': args.add_dist,
            'use_static': args.with_static,
            'use_self': args.with_self,
            'static_path': self.static_files,
            'vary_grav': args.vary_grav,
            'vary_stiff': args.vary_stiff,
            'both_stiff_vary': args.both_stiff_vary,
            'is_fluid': args.is_fluid,
            'is_moving': args.is_moving,
        }
        model_params.update(self.loss_base_params)
        return model_params

    def get_learning_rate_params(self):
        args = self.args

        lr_boundary = [100000, 200000, 300000]
        if args.lr_boundary is not None:
            lr_boundary = [int(x) for x in args.lr_boundary.split(',')]

        all_lrs = np.asarray([1e-3, 5e-4, 1e-4, 5e-5])

        learning_rate_params = {
            'func':lambda global_step, boundaries, values:\
                    tf.train.piecewise_constant(
                        x=global_step,
                        boundaries=boundaries, values=values),
            'values':list( all_lrs * args.init_lr_mult ),
            'boundaries':lr_boundary,
        }
        return learning_rate_params

    def get_optimizer_params(self):
        args = self.args

        # About optimizer params
        optimizer_params = {
                'optimizer_class':tf.train.MomentumOptimizer,
                'momentum':.9
            }

        if args.whichopt==1:
            optimizer_params = {
                'optimizer_class':tf.train.AdamOptimizer,
                'epsilon':args.adameps,
                'beta1':args.adambeta1,
                'beta2':args.adambeta2,
            }

        if args.whichopt==2:
            optimizer_params = {
                'optimizer_class':tf.train.AdagradOptimizer,
            }

        if args.whichopt==3:
            optimizer_params = {
                    'optimizer_class':tf.train.AdagradDAOptimizer,
                    'momentum':.9,
                    'use_nesterov':True
                }

        if args.whichopt==4:
            optimizer_params = {
                'optimizer_class':tf.train.AdadeltaOptimizer,
            }

        if args.whichopt==5:
            optimizer_params = {
                'optimizer_class':tf.train.RMSPropOptimizer,
            }

        optimizer_params['func'] = optimizer_params.pop('optimizer_class')
        return optimizer_params

    def get_loss_params(self):
        args = self.args
        
        loss_params = {
            'pred_targets': [],
            'loss_func': modelsource.flex_l2_particle_loss,
            'loss_func_kwargs':{
                'debug':args.debug}, 
        }
        loss_params['loss_func_kwargs'].update(self.loss_base_params)
        return loss_params

    def get_train_params(self):
        args = self.args

        TRAIN_TARGETS = [
                'mean_gt', 'mean_pred', 
                'velocity_loss'
                ]
        if not args.alpha == 1:
            TRAIN_TARGETS.append('preserve_distance_loss')
        TRAIN_TARGETS.append('un_velocity_loss')

        DATA_BATCH_SIZE = 256
        base_sources = ['full_particles']
        if args.with_coll==1:
            base_sources.append('collision')
        if args.with_static==1:
            base_sources.append('static_collision')
        if args.with_self==1:
            base_sources.append('self_collision')
        if args.vary_grav==1:
            base_sources.append('gravity')
        if args.vary_stiff==1:
            base_sources.append('stiffness')

        filter_rule = (
                utils.moving_not_acting_filter_func, 
                [args.is_moving, 'is_acting'],
                {'is_moving': args.is_moving},
                )
        if args.with_act==1:
            filter_rule = (
                    utils.moving_filter_func, 
                    [args.is_moving],
                    {'is_moving': args.is_moving},
                    )

        base_data_params = {
                'enqueue_batch_size':DATA_BATCH_SIZE,
                'sources': base_sources,
                'sequence_len': args.seq_len,
                'delta_time': 1,
                'filter_rule': filter_rule,
                'resize': None,
                'augment': None,
                'shuffle': True,
                'shuffle_seed': args.train_seed,
                'file_grab_func': utils.table_norot_grab_func,
                'buffer_size': DATA_BATCH_SIZE*20,
                }

        NS_TRAIN_EXAMPLES = [14*256*4] * len(self.data_paths)
        train_data_params = {
                'data': utils.combine_interaction_data(
                    self.data_paths, 
                    NS_TRAIN_EXAMPLES, 
                    self.group_file, 
                    ),
                'is_training':True,
                }

        train_data_params.update(base_data_params)

        def _data_input_fn_wrapper(batch_size, **kwargs):
            data_provider_cls = SequenceNewDataProvider(**kwargs)
            return data_provider_cls.input_fn(batch_size)

        train_data_params['func'] = _data_input_fn_wrapper
        train_data_params['batch_size'] = args.batchsize

        train_params = {
            'num_steps': 460000,
            'data_params': train_data_params,
            'queue_params': None,
            'validate_first':False,
            'thres_loss':float('inf'),
            'targets':{
                'func': utils.return_outputs,
                'targets':TRAIN_TARGETS,
            },
        }
        return train_params
