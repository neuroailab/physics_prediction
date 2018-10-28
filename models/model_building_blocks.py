import sys
import tensorflow as tf
from collections import OrderedDict
import pdb


class ConvNetBase(object):
    def __init__(self, seed=None, **kwargs):
        self.seed = seed
        self.output = None
        self._params = OrderedDict()

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        name = tf.get_variable_scope().name
        if name not in self._params:
            self._params[name] = OrderedDict()
        self._params[name][value['type']] = value

    @property
    def graph(self):
        return tf.get_default_graph().as_graph_def()

    def initializer(
            self, kind='xavier', stddev=0.01, init_file=None, init_keys=None):
        if kind == 'xavier':
            init = tf.contrib.layers.xavier_initializer(seed=self.seed)
        elif kind == 'trunc_norm':
            init = tf.truncated_normal_initializer(
                    mean=0, stddev=stddev, seed=self.seed)
        elif kind == 'from_file':
            params = np.load(init_file)
            init = {}
            init['weight'] = params[init_keys['weight']]
            init['bias'] = params[init_keys['bias']]
        else:
            raise ValueError('Please provide an appropriate initialization '
                             'method: xavier or trunc_norm')
        return init


class ConvNet(ConvNetBase):
    def __init__(self, debug=0, seed=None, **kwargs):
        self.debug = debug
        super(ConvNet, self).__init__(seed=seed, **kwargs)

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        name = tf.get_variable_scope().name
        if name not in self._params:
            self._params[name] = OrderedDict()
        if value['type'] in self._params[name]:
            self._params[name][value['type']]['input'] \
                    = self._params[name][value['type']]['input'] \
                    + ',' + value['input']
        else:
            self._params[name][value['type']] = value

    def activation(self, kind='relu', alpha=0.01, in_layer=None):
        if in_layer is None:
                in_layer = self.output
        last_axis = len(in_layer.get_shape().as_list()) - 1
        if type(kind) != list:
                kind = [kind]
        for_out = []
        for k in kind:
            if self.debug:
                print('activation: ' + k)
            if k == 'relu':
                for_out.append(tf.nn.relu(in_layer, name='relu'))
            elif k == 'elu':
                for_out.append(tf.nn.elu(in_layer, name='elu'))
            elif k == 'crelu':
                for_out.append(tf.nn.crelu(in_layer, name='crelu'))
            elif k == 'leaky_relu':
                for_out.append(tf.maximum(in_layer, alpha * in_layer))
            elif k == 'tanh':
                for_out.append(tf.tanh(in_layer, name = 'tanh'))
            elif k == 'square':
                if self.debug:
                    print('square nonlinearity!')
                for_out.append(in_layer * in_layer)
            elif k == 'safe_square':
                my_tanh = tf.tanh(in_layer)
                for_out.append(my_tanh * my_tanh)
            elif k == 'neg_relu':
                if self.debug:
                    print('neg relu')
                for_out.append(tf.nn.relu(-in_layer, name = 'neg_relu'))
            elif k == 'square_relu':
                if self.debug:
                    print('square relu')
                rel_inlayer = tf.nn.relu(in_layer)
                for_out.append(rel_inlayer)
            elif k == 'square_relu_neg':
                if self.debug:
                    print('square relu neg')
                rel_in = tf.nn.relu(-in_layer)
                for_out.append(rel_in * rel_in)
            elif k == 'square_crelu':
                crel = tf.nn.crelu(in_layer, name = 'crelu')
                for_out.append(crel * crel)
            elif k == 'identity':
                if self.debug:
                    print('no nonlinearity!')
                for_out.append(in_layer)
            elif k == 'zeroflat':
                zthres = tf.get_variable(
                        initializer=tf.constant_initializer(0.1),
                        shape=[1],
                        dtype=tf.float32,
                        name='zero_threshold')
                rel_in = tf.nn.relu(in_layer - tf.abs(zthres)) - \
                        tf.nn.relu(-in_layer - tf.abs(zthres))
                for_out.append(rel_in)
            else:
                raise ValueError("Activation %s not defined" % (k))
        self.output = tf.concat(for_out, last_axis)
        return self.output

    @tf.contrib.framework.add_arg_scope
    def fc(self,
           out_shape,
           init='xavier',
           stddev=.01,
           bias=1,
           activation='relu',
           dropout=.5,
           in_layer=None,
           init_file=None,
           init_layer_keys=None,
           train=True,
           bn_name='',
           trainable=True):

        if not isinstance(out_shape, list):
            out_shape = [out_shape]

        if in_layer is None:
            in_layer = self.output
        if self.debug:
            print(in_layer)
            print([in_layer.get_shape().as_list()[0], -1])

        #let's assume things are flattened, ok?
        resh = in_layer
        resh_shape = resh.get_shape().as_list()
        in_shape = resh_shape[:-2] + [resh_shape[-1]]
        if init != 'from_file':
            kernel = tf.get_variable(
                    initializer=self.initializer(init, stddev=stddev),
                    shape=in_shape + out_shape,
                    dtype=tf.float32,
                    name='weights',
                    trainable=trainable)
            biases = tf.get_variable(
                    initializer=tf.constant_initializer(bias),
                    shape=out_shape,
                    dtype=tf.float32,
                    name='bias',
                    trainable=trainable)
        else:
            init_dict = self.initializer(init,
                                         init_file=init_file,
                                         init_keys=init_layer_keys)
            kernel = tf.get_variable(initializer=init_dict['weight'],
                                     dtype=tf.float32,
                                     name='weights',
                                     trainable=trainable)
            biases = tf.get_variable(initializer=init_dict['bias'],
                                     dtype=tf.float32,
                                     name='bias',
                                     trainable=trainable)

        fcm = tf.matmul(resh, kernel)
        self.output = tf.nn.bias_add(fcm, biases, name='fc')

        if activation is not None:
            self.activation(kind=activation)
        if dropout is not None:
            if self.debug:
                print('Dropout!')
            self.output = tf.nn.dropout(
                    self.output, 
                    dropout, 
                    seed=self.seed, 
                    name='dropout') 

        self.params = {'input': in_layer.name,
                       'type': 'fc',
                       'num_filters': out_shape,
                       'init': init,
                       'bias': bias,
                       'stddev': stddev,
                       'activation': activation,
                       'dropout': dropout,
                       'seed': self.seed}
        return self.output


def hidden_mlp(
        input_node,
        m,
        cfg,
        desc,
        bn_name='',
        hidden_name='',
        stddev=.01,
        reuse_weights=False,
        activation='relu',
        train=False,
        debug=0):
    hidden_depth = cfg[desc + '_depth']
    m.output = input_node
    if debug:
        print('in hidden loop ' + desc)
        print(m.output)
    share_vars = cfg[desc].get('share_vars')
    for i in range(1, hidden_depth + 1):
        if share_vars is not None and share_vars is True:
            var_scope = desc
        else:
            var_scope = desc + str(i)
        with tf.variable_scope('hidden_' + var_scope + hidden_name, \
            reuse=tf.AUTO_REUSE) as scope:
            if reuse_weights:
                scope.reuse_variables()

            nf = cfg[desc][i]['num_features']
            my_activation = cfg[desc][i].get('activation')
            if my_activation is None:
                my_activation = activation
            if train:
                my_dropout = cfg[desc][i].get('dropout')
            else:
                my_dropout = None
            bias = cfg[desc][i].get('bias', .01)

            m.fc(nf, init='xavier', 
                 activation=my_activation,
                 bias=bias, stddev=stddev, 
                 dropout=my_dropout,
                 train=train, bn_name=bn_name)

            if debug:
                print(m.output)
    return m.output
