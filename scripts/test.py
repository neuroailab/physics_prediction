import os
import tensorflow as tf
import sys
import train
from param_setter import ParameterSetter
import utils2

from test_classes.qual_test import retrieve_qualitative_examples
from test_classes.quan_test import retrieve_quantitative_results_parallel
sys.path.append('../data')
from new_data import SequenceNewDataProvider


class TestFramework(object):
    def __init__(self, args):
        self.args = args
        self.param_setter = ParameterSetter(args)

    def build_inputs(self):
        args = self.args

        dataset = args.dataset
        assert ',' not in dataset, "Only one dataset in test is supported"
        DATA_PATH = dataset + '/new_tfdata'
        VALDATA_PATH = dataset + '/new_tfvaldata'

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
                'enqueue_batch_size': 256,
                'sources': sources,
                'sequence_len': args.seq_len,
                'delta_time': 1,
                'filter_rule': None,
                'shuffle_seed' : args.test_seed,
                'special_delta': 1,
                'buffer_size': 512,
                'shuffle_queue': False,
                'num_cores': 1,
                }

        data_provider = SequenceNewDataProvider(**data_init_params)
        self.inputs = data_provider.input_fn(args.MODEL_BATCH_SIZE)

    def build_model(self):
        print('Creating Graph...')
        model_params = ParameterSetter(self.args).get_model_params()
        model_params['my_test'] = True
        if self.args.quant == 0:
            model_params['test_batch_size'] = 1
        else:
            model_params['test_batch_size'] = self.args.TEST_BATCH_SIZE

        func = model_params.pop('func')
        self.outputs, _ = func(inputs=self.inputs, **model_params)

    def build_sess_and_saver(self):
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                gpu_options=gpu_options,
                ))
        self.sess = sess
        self.saver = tf.train.Saver()

    def init_and_restore(self):
        init_op_global = tf.global_variables_initializer()
        self.sess.run(init_op_global)
        init_op_local = tf.local_variables_initializer()
        self.sess.run(init_op_local)

        assert self.args.restore_path, "Must provide a model to restore"
        print('Restore from %s' % self.args.restore_path)
        self.saver.restore(self.sess, self.args.restore_path)

    def prepare_for_test(self):
        self.build_inputs()
        self.build_model()
        self.build_sess_and_saver()
        self.init_and_restore()

    def run_quantitative_test(self):
        n_validation_examples = 32
        retrieve_quantitative_results_parallel(
                self.sess, self.outputs,
                n_validation_examples, self.args)

    def run_qualitative_test(self):
        retrieve_qualitative_examples(
                self.sess, self.outputs, self.args)


def main():
    parser = train.get_parser()
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.SAVE_DIR is None:
        args.SAVE_DIR = os.environ['HOME']

    test_class = TestFramework(args)
    test_class.prepare_for_test()
    if args.quant == 1:
        test_class.run_quantitative_test()
    else:
        test_class.run_qualitative_test()


if __name__ == '__main__':
    main()
