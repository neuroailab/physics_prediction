import os
import tensorflow as tf
import sys
from tfutils import base

import train
from param_setter import ParameterSetter
import utils2

from test_classes.qual_test import retrieve_qualitative_examples
from test_classes.quan_test import retrieve_quantitative_results_parallel

sys.path.append('../data')
sys.path.append('../models')

from new_data import SequenceNewDataProvider

global TEST_BATCH_SIZE
TEST_BATCH_SIZE = 1 #1


def get_params_from_arg(args):
    STEP = None
    dataset = args.dataset

    assert ',' not in dataset, "Only one dataset in test is supported"
    DATA_PATH = dataset + '/new_tfdata'
    VALDATA_PATH = dataset + '/new_tfvaldata'
    STATIC_FILE = dataset + '/static_particles.pkl'

    DATA_BATCH_SIZE = 256
    MODEL_BATCH_SIZE = args.MODEL_BATCH_SIZE #256
    EXP_ID = 'full5231_no_gas_ctl' 
    n_classes = 3
    ctrl_dist = 4

    save_params = {
        'host' : 'localhost',
        'port' : 24444,
        'dbname' : 'physics_obj_dyn',
        'collname' : 'flex',
        'exp_id' : EXP_ID + '_val',
        'save_initial_filters' : False,
        'save_to_gfs' : []
    }

    load_params = {
        'host' : 'localhost',
        'port' : 24444,
        'dbname' : 'physics_obj_dyn',
        'collname': 'flex',
        'exp_id' : EXP_ID,
        'do_restore': True,
        'query': {'step': STEP} if STEP is not None else None,
    }

    model_params = ParameterSetter(args).get_model_params()
    model_params = model_params

    model_params['my_test'] = True
    model_params['test_batch_size'] = TEST_BATCH_SIZE

    val_data_path = VALDATA_PATH
    if args.on_val==0:
        val_data_path = DATA_PATH

    shuffle_flag = args.fancy_test==0

    sources = [
            'full_particles',
            args.is_moving,
            'is_acting']
    if args.with_coll==1:
        sources.append('collision')
    if args.with_self==1:
        sources.append('self_collision')
    if args.with_static==1:
        sources.append('static_collision')
    if args.vary_grav==1:
        sources.append('gravity')
    if args.vary_stiff==1:
        sources.append('stiffness')

    data_init_params = {
            'data': utils2.combine_interaction_data(
                [val_data_path],
                [2*256*4],
                [args.group_file],
                ),
            'is_training': shuffle_flag,
            'enqueue_batch_size': DATA_BATCH_SIZE,
            'sources': sources,
            'sequence_len': args.seq_len,
            'delta_time': 1,
            'filter_rule': None,
            'shuffle_seed' : args.test_seed,
            'special_delta': 1,
            'buffer_size': DATA_BATCH_SIZE*2,
            'shuffle_queue': False,
            'num_cores': 1,
            }
    validation_params = {
        'valid0' : {
            'data_params': {
                'func': SequenceNewDataProvider(**data_init_params).input_fn,
                'batch_size': MODEL_BATCH_SIZE,
                },
            'queue_params': None,
            'targets': {
                'func': lambda inputs, outputs: outputs,
                },
            'agg_func': lambda step_results: step_results[0],
            'num_steps': 1, 
        },
    }

    save_params['exp_id'] = EXP_ID + '_val'
    load_params['exp_id'] = EXP_ID

    params = {
        'save_params' : save_params,
        'load_params' : load_params,
        'model_params' : model_params,
        'validation_params' : validation_params,
        'dont_run': True,
        'skip_check': True,
    }

    return params

def init():
    global TEST_BATCH_SIZE

    parser = train.get_parser()
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    params = get_params_from_arg(args)
    if args.quant==1:
        TEST_BATCH_SIZE = args.TEST_BATCH_SIZE
    else:
        TEST_BATCH_SIZE = 1
    params['model_params']['test_batch_size'] = TEST_BATCH_SIZE

    # get session and data
    print('Creating Graph...')
    test_api = base.test_from_params(**params)

    dbinterface = test_api['dbinterface'][0]
    valid_targets_dict = test_api['validation_targets'][0]
    sess = dbinterface.sess
    return sess, valid_targets_dict, args


if __name__ == '__main__':
    sess, valid_targets_dict, args = init()
    if args.quant==1:
        n_validation_examples = 32
        retrieve_quantitative_results_parallel(
                sess, valid_targets_dict,
                n_validation_examples, args)
    else:
        retrieve_qualitative_examples(sess, valid_targets_dict, args)
