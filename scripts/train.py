import os
from cmd_parser import get_parser
from param_setter import ParameterSetter
import tensorflow as tf
import time


class TrainFramework(object):
    def __init__(self, args):
        self.args = args
        self.param_setter = ParameterSetter(args)

        # Set cache directory
        cache_prefix = self.args.cacheDirPrefix
        if not cache_prefix:
            cache_prefix = os.path.join(os.environ['HOME'], '.model_cache')
            
        self.cache_dir = os.path.join(cache_prefix, args.expId)
        os.system('mkdir -p %s' % self.cache_dir)

        self.log_file_path = os.path.join(self.cache_dir, 'log.txt')
        if not self.args.restore_path:
            self.log_writer = open(self.log_file_path, 'w')
        else:
            self.log_writer = open(self.log_file_path, 'a+')

    def build_inputs(self):
        self.train_params = self.param_setter.get_train_params()
        data_params = self.train_params['data_params']
        func = data_params.pop('func')
        self.inputs = func(**data_params)

    def build_network(self):
        model_params = self.param_setter.get_model_params()
        func = model_params.pop('func')
        self.outputs, _ = func(inputs=self.inputs, **model_params)

    def build_train_op(self):
        loss_params = self.param_setter.get_loss_params()
        input_targets = [self.inputs[key] \
                for key in loss_params['pred_targets']]
        func = loss_params['loss_func']
        self.loss_retval = func(
                self.outputs, 
                *input_targets, 
                **loss_params['loss_func_kwargs'])

        self.global_step = tf.get_variable(
                'global_step', [],
                dtype=tf.int64, trainable=False,
                initializer=tf.constant_initializer(0))
        lr_rate_params = self.param_setter.get_learning_rate_params()
        func = lr_rate_params.pop('func')
        learning_rate = func(self.global_step, **lr_rate_params)

        opt_params = self.param_setter.get_optimizer_params()
        func = opt_params.pop('func')
        opt = func(learning_rate=learning_rate, **opt_params)

        self.train_op = opt.minimize(
                self.loss_retval, 
                global_step=self.global_step)

    def build_train_targets(self):

        extra_targets_params = self.train_params['targets']
        func = extra_targets_params.pop('func')
        train_targets = func(self.inputs, self.outputs, **extra_targets_params)

        train_targets['train_op'] = self.train_op
        train_targets['loss'] = self.loss_retval

        self.train_targets = train_targets

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

        if self.args.restore_path:
            print('Restore from %s' % self.args.restore_path)
            self.saver.restore(self.sess, self.args.restore_path)

    def run_train_loop(self):
        start_step = self.sess.run(self.global_step)

        for curr_step in range(start_step, self.train_params['num_steps']):
            self.start_time = time.time()
            train_res = self.sess.run(self.train_targets)
            duration = time.time() - self.start_time

            message = 'Step {} ({:.0f} ms) -- '\
                    .format(curr_step, 1000 * duration)
            rep_msg = ['{}: {:.4f}'.format(k, v) \
                    for k, v in train_res.items()
                    if k != 'train_op']
            message += ', '.join(rep_msg)
            print(message)

            if curr_step % self.args.fre_save == 0:
                print('Saving model...')
                self.saver.save(
                        self.sess, 
                        os.path.join(
                            self.cache_dir,
                            'model.ckpt'), 
                        global_step=curr_step)
            self.log_writer.write(message + '\n')
            if curr_step % 200 == 0:
                self.log_writer.close()
                self.log_writer = open(self.log_file_path, 'a+')

    def train(self):
        self.build_inputs()
        self.build_network()
        self.build_train_op()
        self.build_train_targets()

        self.build_sess_and_saver()
        self.init_and_restore()

        self.run_train_loop()


def main():
    parser = get_parser()
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    train_framework = TrainFramework(args)
    train_framework.train()


if __name__ == '__main__':
    main()
