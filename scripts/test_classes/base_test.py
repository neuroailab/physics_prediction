import numpy as np
import os, sys
import tensorflow as tf
import h5py
import pdb


class BaseTest(object):
    """
    Base class for testing
    """
    def __init__(
            self,
            sess, 
            outputs, 
            args,
            seed,
            ):
        self.sess = sess
        self.args = args
        self.seed = seed
        self.trivial_update_keys = []
        
        self.unpack_ops(outputs, args)

    def unpack_ops(self, outputs, args):
        args = self.args

        input_ts_keys = [
                'full_particles', 
                args.is_moving, 
                'is_acting']
        placeholders_keys = ['particles']

        if args.with_coll == 1:
            input_ts_keys.append('collision')
            placeholders_keys.append('collision')
        if args.with_self == 1:
            input_ts_keys.append('self_collision')
            placeholders_keys.append('self_collision')
        if args.with_static == 1:
            input_ts_keys.append('static_collision')
            input_ts_keys.append('static_particles')
            placeholders_keys.append('static_collision')
        if args.vary_grav == 1:
            input_ts_keys.append('gravity')
            placeholders_keys.append('gravity')
            self.trivial_update_keys.append('gravity')
        if args.vary_stiff == 1:
            input_ts_keys.append('stiffness')
            placeholders_keys.append('stiffness')
            self.trivial_update_keys.append('stiffness')

        input_tensor_dict = {}
        for each_key in input_ts_keys:
            input_tensor_dict[each_key] = outputs[each_key]

        placeholders = {}
        for each_key in placeholders_keys:
            placeholders[each_key] = outputs['%s_placeholder' % each_key]

        # run model ops
        predict_velocity_ts = outputs['pred_particle_velocity']

        self.input_tensor_dict = input_tensor_dict
        self.placeholders = placeholders
        self.predict_velocity_ts = predict_velocity_ts

    def pull_batch_random(self, n_random_pulls=2):
        args = self.args

        exact_number = args.fancy_test
        rng = np.random.RandomState(seed=self.seed)
        sess = self.sess

        if exact_number == 0:
            for i in range(max(rng.randint(n_random_pulls), 1)):
                batch = sess.run(self.input_tensor_dict)
        else:
            # Otherwise, just pull exact_number times
            for _ in range(exact_number):
                batch = sess.run(self.input_tensor_dict)

        for k in batch:
            batch[k] = np.squeeze(batch[k])
        return batch


    def get_use_frame(self, batch):
        args = self.args

        if args.with_act:
            use_frame = batch[args.is_moving]
        else:
            use_frame = batch[args.is_moving] * (1 - batch['is_acting'])

        use_frame = np.sum(use_frame, axis=1) == args.seq_len
        if args.fancy_test>=1:
            use_frame = np.ones(use_frame.shape, dtype=use_frame.dtype)

        self.use_frame = use_frame
        return use_frame

    def get_ranges(self, batch):
        args = self.args

        self.get_use_frame(batch)
        use_frame = self.use_frame

        unroll_length = args.unroll_length
        isone = np.concatenate(([0], np.equal(use_frame, 1).view(np.int8), [0]))
        absdiff = np.abs(np.diff(isone))
        ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
        if unroll_length is not None:
            new_ranges = []
            ranges = list(ranges)
            for i, r in enumerate(ranges):
                while r[1] - r[0] > unroll_length:
                    new_ranges.append([r[0], r[0] + unroll_length])
                    r[0] = r[0] + unroll_length
            ranges = np.array(new_ranges)

        if args.fancy_test>=1:
            if args.only_first>=1:
                ranges = ranges[:args.only_first]
            print(ranges)

        if args.manual_range is not None:
            splt_range = args.manual_range.split(':')
            ranges = [[int(x) for x in _tmp.split(',')] for _tmp in splt_range]
            print(ranges)
        self.ranges = ranges
        return ranges

    def _change_shape(self, particle):
        raise NotImplementedError

    def _get_copy(self, frame, copy_tensor):
        raise NotImplementedError

    def _move_by(self, move_tensor):
        raise NotImplementedError

    def _add_by(
            self,
            orig_tensor, add_tensor, 
            val, part_indx):
        raise NotImplementedError

    def _replace_by(
            self,
            orig_tensor, replace_tensor, 
            val, part_indx):
        raise NotImplementedError

    def _coll_online(
            self,
            obj_pair, particle, collision, 
            static_particles=None, initial_distances=None):
        raise NotImplementedError

    def get_init_prtcl_dist(self, dataset):
        h5_path = '/mnt/fs1/datasets/' + dataset + '/dataset0.hdf5'
        assert os.path.exists(h5_path), \
                "Need original hdf5 file at %s!" % h5_path
        f = h5py.File(h5_path, 'r')

        # initial undeformed particles, right after teleport at t = 10 + 1
        initial_particles = np.reshape(f['particles'][11], [-1, 7])
        pos = initial_particles[:,0:3]
        r = np.sum(pos*pos, axis=-1)
        r = np.reshape(r, [-1, 1])
        distance = r - 2*np.matmul(pos, pos.transpose()) + r.transpose()
        return np.sqrt(np.abs(distance))

    def update_with_gt(
            self,
            batch,
            frame):
        # First take the last time point
        _slice_full = np.take(
                batch['full_particles'], 
                indices=-1,
                axis=-3)
        # Then take the future velocity
        _slice_pv = np.take(
                _slice_full, 
                indices=np.arange(15,18),
                axis=-1)

        self.predicted_velocity = self._get_copy(frame, _slice_pv)

    def update_with_run(self):
        args = self.args
        
        all_inputs = {'particles': self.particle}
        if args.with_coll == 1:
            all_inputs['collision'] = self.collision
        if args.with_self == 1:
            all_inputs['self_collision'] = self.self_collision
        if args.with_static == 1:
            all_inputs['static_collision'] = self.static_collision
        for key in self.more_inp:
            all_inputs[key] = self.more_inp[key]

        feed_dict = {}
        for input_name, input_value in all_inputs.items():
            now_place = self.placeholders[input_name]
            feed_dict[now_place] = self._change_shape(input_value)

        predicted_velocity = self.sess.run(
                [self.predict_velocity_ts],
                feed_dict=feed_dict)
        self.predicted_velocity = np.squeeze(predicted_velocity[0])

    def unroll_init(
            self,
            frame, 
            batch, 
            use_ground_truth_for_unroll,
            ):
        args = self.args
        self.particle = self._get_copy(frame, batch['full_particles'])

        self.collision = None
        if args.with_coll==1:
            self.collision = self._get_copy(frame, batch['collision'])

        self.self_collision = None
        if args.with_self==1:
            self.self_collision =self._get_copy(frame, batch['self_collision'])

        self.static_collision = None
        self.static_particles = None
        if args.with_static==1:
            # TODO: fix the static particles problem (not important though)
            self.static_collision \
                    = self._get_copy(frame, batch['static_collision'])
            self.static_particles \
                    = self._get_copy(frame, batch['static_particles'])

        # More input tensors
        self.more_inp = {}
        for key in self.trivial_update_keys:
            self.more_inp[key] = self._get_copy(frame, batch[key])

        if use_ground_truth_for_unroll:
            self.update_with_gt(batch, frame)
        else:
            # predict next velocity
            self.update_with_run()

        _ap_particle = np.take(self.particle, indices=-1, axis=-3)
        self.predicted_particles.append(_ap_particle)

        _ap_full_particle = np.take(
                batch['full_particles'],
                indices=-1,
                axis=-3)
        self.true_particles.append(self._get_copy(frame, _ap_full_particle))

        return self.particle

    def _get_obj_pair(self, particle_at_0):
        args = self.args

        all_obj_idx = np.unique(particle_at_0[:, 14])
        self.all_obj_idx = all_obj_idx

        if args.with_coll==1 and self.obj_pair is None:
            for indx_0 in range(len(all_obj_idx)):
                for indx_1 in range(indx_0+1, len(all_obj_idx)):
                    _obj0_indx = []
                    _obj1_indx = []
                    for _indx in range(particle_at_0.shape[0]):
                        if particle_at_0[_indx, 14]==all_obj_idx[indx_0]:
                            _obj0_indx.append(_indx)
                        elif particle_at_0[_indx, 14]==all_obj_idx[indx_1]:
                            _obj1_indx.append(_indx)
                    
                    _obj_pair = np.transpose([
                        np.tile(_obj0_indx, len(_obj1_indx)),
                        np.repeat(_obj1_indx, len(_obj0_indx)),])
                    if self.obj_pair is None:
                        self.obj_pair = _obj_pair
                    else:
                        self.obj_pair = np.concatenate(
                                [self.obj_pair, _obj_pair], 
                                axis=0)

    def _get_self_pair(self, particle_at_0):
        args = self.args
        objs = [(particle_at_0[:,14:15] == obj_idx).astype(np.float64) \
                for obj_idx in self.all_obj_idx]
        for indx_0 in range(len(objs)):
            mask = np.matmul(objs[indx_0], objs[indx_0].transpose())
            mask = np.triu(mask)
            _obj_pair = np.where(mask)
            _obj_pair = np.array([
                _obj_pair[0],
                _obj_pair[1]]).transpose()
            if self.same_part_pair is None:
                self.same_part_pair = _obj_pair
            else:
                self.same_part_pair = np.concatenate(
                        [self.same_part_pair, _obj_pair], 
                        axis=0)

    def _get_sta_obj_pair(self, particle_at_0):
        args = self.args
        if args.with_static==1:
            _obj0_indx = range(particle_at_0.shape[0])
            _obj1_indx = range(self.static_particles.shape[0])

            self.sta_obj_pair = np.transpose([
                np.tile(_obj0_indx, len(_obj1_indx)),
                np.repeat(_obj1_indx, len(_obj0_indx)),
                ])

    def _update_inputs(self, frame, batch):
        args = self.args
        
        self.particle = self._move_by(self.particle)
        if args.with_coll == 1:
            self.collision = self._move_by(self.collision)
        if args.with_self == 1:
            self.self_collision = self._move_by(self.self_collision)
        if args.with_static == 1:
            self.static_collision = self._move_by(self.static_collision)

        _particle_obj_idx = np.take(
                np.take(self.particle, indices=-1, axis=-3), 
                indices=14,
                axis=-1)
        val = np.squeeze(_particle_obj_idx.nonzero())

        # Update particle
        self.particle = self.particle.copy()
        self.particle = self._add_by(
                self.particle, self.predicted_velocity, 
                val, (0, 3))
        self.particle = self._replace_by(
                self.particle, self.predicted_velocity, 
                val, (4, 7))

        # Update forces
        def _update_sta_end(_sta=7, _end=10, inp=None):
            _take_indx = range(_sta, _end)
            if inp is None:
                _tmp_particle = np.take(
                        np.take(batch['full_particles'], indices=-1, axis=-3),
                        indices=_take_indx,
                        axis=-1)
                _tmp_particle = self._get_copy(frame, _tmp_particle)
            else:
                _tmp_particle = np.take(
                        inp,
                        indices=_take_indx,
                        axis=-1)
            self.particle = self._replace_by(
                    self.particle, _tmp_particle,
                    val, (_sta, _end))

        if frame < len(batch['full_particles']):
            _update_sta_end()
        else:
            # set external actions to 0
            _update_sta_end(inp=np.zeros(particle.shape))

        # update things in more_inp
        for key in self.more_inp:
            self.more_inp[key] = self._get_copy(frame, batch[key])

    def _update_collisions(self, frame, batch):
        args = self.args
        
        if args.with_coll==1:
            if args.coll_online==1:
                self.collision = self._coll_online(
                        self.obj_pair, 
                        self.particle, 
                        self.collision)
            else:
                assert frame < args.MODEL_BATCH_SIZE, \
                        'Must use coll_online=1 if doing long unrolling'
                self.collision = self._get_copy(frame, batch['collision'])

        if args.with_self==1:
            if args.coll_online==1:
                self.self_collision = self._coll_online(
                        self.same_object_pair, 
                        self.particle, self.self_collision, 
                        initial_distances=self.initial_distances)
            else:
                assert frame < args.MODEL_BATCH_SIZE, \
                        'Must use coll_online=1 if doing long unrolling'
                self.self_collision = self._get_copy(
                        frame, 
                        batch['self_collision'])

        if args.with_static==1:
            if args.coll_online==1:
                assert self.static_particles is not None
                self.static_collision = self._coll_online(
                        self.sta_obj_pair, self.particle, 
                        self.static_collision, 
                        static_particles=self.static_particles)
            else:
                assert frame < args.MODEL_BATCH_SIZE, \
                        'Must use coll_online=1 if doing long unrolling'
                self.static_collision = self._get_copy(
                        frame, 
                        batch['static_collision'])
                self.static_particles = self._get_copy(
                        frame, 
                        batch['static_particles'])

    def unroll_noninit(
            self,
            frame, batch,
            use_ground_truth_for_unroll,
            ):
        self._update_inputs(frame, batch)

        _ap_particle = np.take(
                self.particle,
                indices=-1,
                axis=-3)
        self.predicted_particles.append(_ap_particle)

        _ap_full_particle = np.take(
                batch['full_particles'],
                indices=-1,
                axis=-3)
        if frame < len(batch['full_particles']):
            self.true_particles.append(self._get_copy(frame, _ap_full_particle))

        if use_ground_truth_for_unroll:
            self.update_with_gt(batch, frame)
        else:
            self._update_collisions(frame, batch)
            # predict next velocity
            self.update_with_run()
