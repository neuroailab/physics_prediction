import argparse


def get_general_setting(parser):
    # General settings
    parser.add_argument(
            '--debug', default=0, type=int, action='store', 
            help='Whether to use debug print outs or not')
    parser.add_argument(
            '--expId', default="interact30both", type=str, action='store', 
            help='Name of experiment id')
    parser.add_argument(
            '--cacheDirPrefix', default=None, type=str, action='store', 
            help='Prefix of cache directory')
    parser.add_argument(
            '--batchsize', default=256, type=int, action='store', 
            help='Batch size')

    parser.add_argument(
            '--fre_save', default=4000, type=int, action='store', 
            help='Saving frequency')
    parser.add_argument(
            '--restore_path', default=None, type=str, action='store', 
            help='Ckpt path for restoring')

    # GPU related
    parser.add_argument(
            '--gpu', default = '0', type = str, action = 'store', 
            help = 'Availabel GPUs')

    return parser


def get_optimizer_setting(parser):
    # Optimizer related
    parser.add_argument(
            '--whichopt', default=1, type=int, action='store',
            help='Choice of the optimizer, 0 means momentum, 1 means Adam')
    parser.add_argument(
            '--adameps', default=1e-08, type=float, action='store',
            help='Epsilon for adam, only used when whichopt is 1')
    parser.add_argument(
            '--adambeta1', default=0.9, type=float, action='store',
            help='Beta1 for adam, only used when whichopt is 1')
    parser.add_argument(
            '--adambeta2', default=0.999, type=float, action='store',
            help='Beta2 for adam, only used when whichopt is 1')
    parser.add_argument(
            '--init_lr_mult', default=1, type=float, action='store',
            help='Init learning rate multiply')
    parser.add_argument(
            '--lr_boundary', default=None, type=str, action='store',
            help='Learning rate boundaries, split by ",", have 3 numbers')

    return parser


def get_train_setting(parser):
    # Training related
    parser.add_argument(
            '--num_steps', default=float('inf'), type=float, action='store', 
            help='Number of training steps')
    parser.add_argument(
            '--train_seed', default=0, type=int, action='store', 
            help='Seed for training')

    ## Model related
    parser.add_argument(
            '--alpha', default=0.01, type=float, action='store', 
            help='Loss ratio for velocity loss.')
    parser.add_argument(
            '--number_of_kNN', default=7, type=int, action='store', 
            help='Number of nearest nodes used in relations')
    parser.add_argument(
            '--OB1', default=22, type=int, action='store', 
            help='OB1 for model params')
    parser.add_argument(
            '--OB2', default=23, type=int, action='store', 
            help='OB2 for model params')
    parser.add_argument(
            '--room_center', default='10,0.2,10', type=str, 
            action='store', help='Center of the room')
    parser.add_argument(
            '--gravity_term', default=9.81, type=float, action='store', 
            help='Gravity constant')
    parser.add_argument(
            '--network_func', default='interaction_cfg', type=str, 
            action='store', help='Function name for network')
    parser.add_argument(
            '--network_nonlin', default='relu', type=str, 
            action='store', help='Nonlinearity for network')
    parser.add_argument(
            '--seq_len', default=1, type=int, action='store', 
            help='How many timesteps to use as input')
    parser.add_argument(
            '--add_dist', default=0, type=int, action='store', 
            help='Whether adding ground truth distance, default is 0 (no)')
    parser.add_argument(
            '--pdr', default=100000, type=int, action='store', 
            help='Preserver distance radius')
    parser.add_argument(
            '--use_running_mean', default=0, type=int, action='store',
            help='Whether to use running mean for local loss normalization')
    parser.add_argument(
            '--add_gloss', default=0, type=float, action='store', 
            help='Whether adding global loss to the final loss')
    parser.add_argument(
            '--avd_obj_mask', default=0, type=int, action='store', 
            help='Whether avoiding masking using object idxs, default is 0 (no)')
    parser.add_argument(
            '--is_fluid', default=0, type=int, action='store', 
            help='Whether it is a fluid or not')
    parser.add_argument(
            '--is_moving', default='is_moving', type=str, action='store', 
            help='Which is_moving to use')
    return parser


def get_dataset_setting(parser):
    ## Group related
    parser.add_argument(
            '--group_file', 
            default=None, required=True,
            type=str, action='store', 
            help='Path to the generated group files')

    ## Dataset related
    parser.add_argument(
            '--dataset', default="14_world_dataset", type=str, 
            action='store', help='Name of the dataset')
    parser.add_argument(
            '--dataset_search_dir', 
            default=None, type=str,
            action='store', help='Search directories of the dataset')
    parser.add_argument(
            '--with_coll', default=0, type=int, action='store', 
            help='Whether include collision or not')
    parser.add_argument(
            '--with_act', default=0, type=int, action='store',
            help='Whether include action or not')
    parser.add_argument(
            '--with_static', default=0, type=int, action='store',
            help='Whether the used dataset contains static flex objects or not')
    parser.add_argument(
            '--with_self', default=0, type=int, action='store',
            help='Whether the used dataset contains self collisions or not')

    parser.add_argument(
            '--max_collision_distance', default=0.5, type=float, 
            action='store', 
            help='Max distance for adding the collision relationship')
    parser.add_argument(
            '--vary_grav', default=0, type=int, action='store', 
            help='Whether use gravity from input')
    parser.add_argument(
            '--vary_stiff', default=0, type=int, action='store', 
            help='Whether use stiffness from input')
    parser.add_argument(
            '--both_stiff_vary', default=0, type=int, action='store', 
            help='Whether two stiffnesses will both vary')

    return parser


def get_test_setting(parser):
    # Parameter for test
    parser.add_argument(
            '--SAVE_DIR', default=None, type=str, action='store', 
            help='Save directory')
    parser.add_argument(
            '--on_val', default=1, type=int, action='store', help='Whether on validation set')
    parser.add_argument(
            '--MODEL_BATCH_SIZE', default=64, type=int, action='store', help='Batch size')
    parser.add_argument(
            '--save_suf', default=None, type=str, action='store', help='Suffix of saving')
    parser.add_argument(
            '--quant', default=1, type=int, action='store', 
            help='Whether to compute quantitative metrics or not')
    parser.add_argument(
            '--unroll_length', default=None, type=int, action='store', 
            help='Unrolling length of prediction during evaluation')
    parser.add_argument(
            '--pred_unroll_length', default=None, type=int, action='store', 
            help='Unrolling length of prediction during evaluation')
    parser.add_argument(
            '--coll_online', default=1, type=int, action='store', 
            help='Whether compute collision relations online')
    parser.add_argument(
            '--test_seed', default=8, type=int, action='store', 
            help='Random seed for testing')
    parser.add_argument(
            '--TEST_BATCH_SIZE', default=64, type=int, action='store', 
            help='test batch size, for quant==1')
    parser.add_argument(
            '--test_n_pulls', default=4, type=int, action='store', 
            help='Number of pulls, for quant==1')
    parser.add_argument(
            '--fancy_test', default=0, type=int, action='store', 
            help='If set to 1, will pull the first part of the test data')
    parser.add_argument(
            '--only_first', default=0, type=int, action='store', 
            help='If set to bigger than 1, will only use the first several ranges')
    parser.add_argument(
            '--manual_range', default=None, type=str, action='store', 
            help='If set, will use this range')
    parser.add_argument(
            '--qual_random_pull', 
            default=16, type=int, action='store', 
            help='Maximal of random pulls, for qual')
    return parser


def get_parser():
    parser = argparse.ArgumentParser(
            description='Solve the physics!')

    parser = get_general_setting(parser)
    parser = get_optimizer_setting(parser)
    parser = get_train_setting(parser)
    parser = get_dataset_setting(parser)
    parser = get_test_setting(parser)

    return parser
