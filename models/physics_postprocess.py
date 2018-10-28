import copy
import tensorflow as tf
import numpy as np
from itertools import product
import cPickle


class PhysicsPostprocess(object):
    def __init__(self,
            inputs,
            static_path = None,
            use_static=0,
            which_dataset = None,
            seed=0,
            room_center='10,0.2,10',
            rot_amp=1.,
            mass_amp=None,
            mass_adjust=None,
            *args,  **kwargs):
        self.mass_amp = mass_amp
        self.mass_adjust = mass_adjust 
        self.mass_change_value = None
        self.use_static = use_static
        self.room_center = room_center
        self.which_dataset = which_dataset

        inputs['sparse_particles'] = inputs['full_particles']

        inputs['sparse_particles'] = \
                self.adjust_mass_for_particles(inputs['sparse_particles'])

        if use_static==1:
            assert static_path, "Must input static_path if use static!"
            inputs = self.load_static(static_path, inputs)

        inputs = self.subtract_room_center(inputs)

        self.inputs = inputs

    def subtract_room_center(self, inputs):
        normalize_vec = [float(x) for x in self.room_center.split(',')]

        def _add_extra_dim(big_tensor):
            extra_dim = big_tensor.get_shape().as_list()[-1] - 3
            new_norm_vec = normalize_vec + [0] * extra_dim
            return new_norm_vec

        inputs['sparse_particles'] \
                -= np.asarray(_add_extra_dim(inputs['sparse_particles']))
        inputs['full_particles'] \
                -= np.asarray(_add_extra_dim(inputs['full_particles']))

        if self.use_static==1:
            inputs['static_particles'] \
                    -= np.asarray(_add_extra_dim(inputs['static_particles']))
        return inputs

    def load_static(self, static_path, inputs):
        bs, ts = inputs['sparse_particles'].get_shape().as_list()[0:2]

        static_particles = []
        if not isinstance(static_path, list):
            static_path = [static_path]
        max_static_shape = 0
        for sp in static_path:
            static_particles.append(cPickle.load(open(sp, 'rb'))[0])
            max_static_shape = max(
                    static_particles[-1].shape[0], 
                    max_static_shape)

        for sp in xrange(len(static_particles)):
            if static_particles[sp].shape[0]<max_static_shape:
                left_dim = max_static_shape - static_particles[sp].shape[0]
                static_particles[sp] = np.concatenate([
                    static_particles[sp], 
                    np.zeros(
                        [left_dim, static_particles[sp].shape[1]], 
                        dtype=np.float32)],
                    axis=0)
                 
        inputs['static_particles'] = tf.stack(static_particles)
        if self.which_dataset is not None: # multiple datasets
            inputs['static_particles'] = tf.tile(
                    tf.expand_dims(
                        tf.gather(
                            inputs['static_particles'], 
                            self.which_dataset), 
                        1), 
                    [1, ts, 1, 1])
        else:
            assert inputs['static_particles'].shape[0]==1
            inputs['static_particles'] = tf.tile(
                    tf.expand_dims(inputs['static_particles'], 0), 
                    [bs, ts, 1, 1])

        if self.mass_amp is not None:
            # mass_adjust will not change static particle mass
            tmp_sp = inputs['static_particles']
            tmp_sp = tf.concat([
                tmp_sp[:,:,:,:3],
                tmp_sp[:,:,:,3:4]*self.mass_change_value,
                tmp_sp[:,:,:,4:],
                ],
                axis=-1)
            inputs['static_particles'] = tmp_sp
        return inputs

    def adjust_mass_for_particles(self, particles):
        # Adjust mass which influences both mass and forces randomly or 
        # according to one number
        if self.mass_change_value is None:
            self.mass_change_value = 1
            if self.mass_adjust:
                self.mass_change_value = self.mass_adjust

        # for mass
        changed_states = tf.concat([
            particles[:,:,:,0:3],
            particles[:,:,:,3:4] * self.mass_change_value, 
            particles[:,:,:,4:7]], 
            axis=-1,)
        # for forces, and torques
        changed_actions = particles[:,:,:,7:13] * self.mass_change_value

        particles = tf.concat([
            changed_states, 
            changed_actions, 
            particles[:,:,:,13:19]], axis=-1)
        return particles
