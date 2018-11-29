from base_test import BaseTest
from tqdm import tqdm, trange
import numpy as np
import os
import cPickle
import pdb

from multiprocessing import Pool


def pad_and_mask_along_axis(data, target_length, axis=0):
    data = np.array(data)
    assert axis < len(data.shape), (axis, len(data.shape))
    data_mask = np.ones(data.shape)
    if target_length <= data.shape[axis]:
        return data, data_mask

    data_padding = np.zeros((len(data.shape), 2)).astype(np.int32)
    data_padding[axis, 1] = target_length - data.shape[axis]
    data = np.pad(data, data_padding, 'constant')
    data_mask = np.pad(data_mask, data_padding, 'constant')
    return data, data_mask


def pad_to_same_size(x, y):
    assert len(x.shape) == len(y.shape), \
            'x and y must have the same rank'
    x_pad = np.zeros((len(x.shape), 2), np.int32)
    y_pad = np.zeros((len(y.shape), 2), np.int32)

    for i in range(len(x.shape)):
        if x.shape[i] > y.shape[i]:
            y_pad[i] = (0, x.shape[i] - y.shape[i])
        elif x.shape[i] < y.shape[i]:
            x_pad[i] = (0, y.shape[i] - x.shape[i])

    y = np.pad(y, y_pad, 'constant')
    x = np.pad(x, x_pad, 'constant')

    assert y.shape == x.shape, (y.shape, x.shape)
    return x, y


def compute_error(results, total_error, preserve_error, total_n):
    _obj_idx_0 = results[0]['same_part_pair'][:,0]
    _obj_idx_1 = results[0]['same_part_pair'][:,1]
    for r in results:
        error = (r['predicted_particles'] - r['true_particles']) ** 2
        total_error, error = pad_to_same_size(total_error, error)
        total_error += error

        # To get the preserve error
        dist_0 = np.sqrt(np.sum((r['predicted_particles'][:, _obj_idx_0, :3] \
                - r['predicted_particles'][:, _obj_idx_1, :3])**2, axis=-1))
        dist_1 = np.sqrt(np.sum((r['true_particles'][:, _obj_idx_0, :3] \
                - r['true_particles'][:, _obj_idx_1, :3])**2, axis=-1))
        _pres_error = (dist_0 - dist_1) ** 2
        preserve_error, _pres_error = pad_to_same_size(preserve_error, _pres_error)
        preserve_error += _pres_error

        if len(error) > len(total_n):
            total_n = np.pad(total_n, 
                    (0, len(error) - len(total_n)),
                    'constant')
        total_n[:len(error)] += 1

    num_nodes = np.max(_obj_idx_0)+1
    ret_pre_res = np.zeros([preserve_error.shape[0], num_nodes])
    for indx_n in xrange(num_nodes):
        ret_pre_res[:, indx_n] = np.sum(
                preserve_error[:, _obj_idx_0==indx_n], 
                axis=-1)

    return total_error, preserve_error, total_n, ret_pre_res


def _cal_collision_one(
        _bindx, obj_pair, particle, 
        collision, static_particles, initial_distances, use_mult_thread=1):
    if use_mult_thread==0:
        curr_particle = particle[_bindx, :, :]
        if static_particles is None:
            curr_static_particles = None
        else:
            curr_static_particles = static_particles[_bindx, :, :]
    else:
        curr_particle = particle
        curr_static_particles = static_particles

    if static_particles is not None:
        assert initial_distances is None
        _all_coll_dist = np.linalg.norm(
                curr_particle[-1, obj_pair[:,0], 0:3] -\
                        curr_static_particles[obj_pair[:,1], 0:3],
                        axis=-1)
    elif initial_distances is not None:
        assert static_particles is None
        _all_coll_dist = np.linalg.norm(
                curr_particle[-1, obj_pair[:,0], 0:3] -\
                        curr_static_particles[obj_pair[:,1], 0:3],
                        axis=-1)
    else:
        _all_coll_dist = np.linalg.norm(
                curr_particle[-1, obj_pair[:,0], 0:3] -\
                        curr_particle[-1, obj_pair[:,1], 0:3],
                        axis=-1)
    _all_coll_dist = np.concatenate(
            [obj_pair, _all_coll_dist[:, np.newaxis]], 
            axis=-1)
    if initial_distances is not None:
        _all_init_dist = initial_distances[obj_pair[:,0], obj_pair[:,1]]
        _all_coll_dist = np.concatenate([_all_coll_dist, \
                _all_init_dist[:, np.newaxis]], axis=-1)
    _all_coll_dist = sorted(_all_coll_dist, key=lambda x:x[2])
    _all_coll_dist = np.asarray(_all_coll_dist[:collision.shape[2]])
    if use_mult_thread==0:
        if initial_distances is not None:
            collision[_bindx, -1, :, :] = _all_coll_dist
        else:
            collision[_bindx, -1, :, :] = _all_coll_dist
    else:
        return _all_coll_dist


def _cal_collision_one_wrapper(args):
    return _cal_collision_one(*args)


class QuanTest(BaseTest):
    """
    Class for qualitative testing
    """
    def __init__(self, *args, **kwargs):
        super(QuanTest, self).__init__(*args, **kwargs)
        self.thread_p = Pool(10)

    def _change_shape(self, particle):
        return particle

    def _get_copy(self, frame, copy_tensor):
        return copy_tensor[:, frame].copy()

    def _move_by(self, move_tensor):
        move_tensor[:, :-1] = move_tensor[:, 1:]
        return move_tensor

    def _add_by(
            self, 
            orig_tensor, add_tensor, 
            val, part_indx):
        orig_tensor[val[0], -1, val[1], part_indx[0]:part_indx[1]] \
                += add_tensor[val[0], val[1]]
        return orig_tensor

    def _replace_by(
            self,
            orig_tensor, replace_tensor, 
            val, part_indx):
        orig_tensor[val[0], -1, val[1], part_indx[0]:part_indx[1]] \
                = replace_tensor[val[0], val[1]]
        return orig_tensor

    def _get_sta_obj_pair(self, particle_at_0):
        args = self.args
        if args.with_static==1:
            _obj0_indx = range(particle_at_0.shape[0])
            _obj1_indx = range(self.static_particles.shape[2])

            self.sta_obj_pair = np.transpose([
                np.tile(_obj0_indx, len(_obj1_indx)),
                np.repeat(_obj1_indx, len(_obj0_indx)),
                ])

    def _coll_online(
            self,
            obj_pair, particle, collision, 
            static_particles=None,
            initial_distances=None):
        _bs = particle.shape[0]

        all_arg = []
        for _bindx in range(_bs):
            if static_particles is None:
                curr_static_particles = None
            else:
                curr_static_particles = static_particles[_bindx, -1]

            if initial_distances is None:
                curr_initial_distances = None
            else:
                curr_initial_distances = initial_distances[_bindx]

            all_arg.append(
                    (_bindx, obj_pair, 
                        particle[_bindx], 
                        collision[_bindx:_bindx+1], 
                        curr_static_particles, 
                        curr_initial_distances
                        ))

        mult_res = self.thread_p.map(_cal_collision_one_wrapper, all_arg)
        collision[:, -1, :, :] = np.stack(mult_res)
        return collision

    def unroll_prediction_parallel(
            self, 
            batch,
            use_ground_truth_for_unroll):
        args = self.args

        predicted_sequences = []

        self.predicted_particles = []
        self.true_particles = []

        self.obj_pair = None
        self.sta_obj_pair = None
        self.same_part_pair = None
        if args.with_self == 1:
            self.initial_distances = self.get_init_prtcl_dist(args.dataset)

        init = True
        ranges = self.ranges
        max_range = max(ranges[:,1] - ranges[:,0])
        for frame in range(0, max_range):
            if init:
                init = False
                particle = self.unroll_init(
                        frame, 
                        batch, 
                        use_ground_truth_for_unroll,
                        )

                if args.coll_online == 1:
                    particle_at_0 = particle[0, 0]
                    self._get_obj_pair(particle_at_0)
                    self._get_self_pair(particle_at_0)
                    self._get_sta_obj_pair(particle_at_0)
            else:
                self.unroll_noninit(
                        frame, batch, 
                        use_ground_truth_for_unroll)

        def remove_padding(data, ranges, stack=True):
            if stack:
                data = np.stack(data, axis=1)
            data = data[:len(ranges)]
            data = [data[i,:(r[1]-r[0])] for i, r in enumerate(ranges)]
            return data
        self.predicted_particles = remove_padding(
                self.predicted_particles, 
                ranges)
        self.true_particles = remove_padding(
                self.true_particles, 
                ranges)
        batch['input_particles'] = remove_padding(
                batch['input_particles'], 
                ranges,
                stack=False)

        for pred, true, inp \
                in zip(
                        self.predicted_particles, 
                        self.true_particles, 
                        batch['input_particles']):
            predicted_sequences.append({
                'predicted_particles': pred,
                'true_particles': true,
                'input_particles': inp,
                'same_part_pair': self.same_part_pair,
                })
        return predicted_sequences


def save_and_report_results(results, args):
    predicted_error = results['pred_err_sq']
    preserve_error = results['preserve_error']
    final_pre_err = results['final_pre_err']
    predicted_n = results['pred_n']

    # Position
    results['pred_pos_mse'] = np.sum(predicted_error[:,:,0:3]) \
            / np.sum(predicted_n * predicted_error.shape[1] * 3)
    print('=========================')
    print('===== POSITION MSE ======')
    print('=========================')
    print('Prediction: %f' % (results['pred_pos_mse']))
    results['pred_pos_mse_per_t'] = np.sum(predicted_error[:,:,0:3], axis=(1,2)) \
            / (predicted_n * predicted_error.shape[1] * 3)
    print('=========================')
    print('= POSITION MSE PER TIME =')
    print('=========================')
    for i, r in enumerate(results['pred_pos_mse_per_t']):
        print('t =%2d  |  %.6f' % (i, r))

    # Preserve MSE
    results['preserve_mse'] = np.sum(final_pre_err) \
            / np.sum(predicted_n * final_pre_err.shape[1])
    print('=========================')
    print('===== PRESERVE MSE ======')
    print('=========================')
    print('Prediction: %f' % (results['preserve_mse']))
    results['preserve_mse_per_t'] = np.sum(final_pre_err, axis=1) \
            / (predicted_n * final_pre_err.shape[1])
    print('=========================')
    print('= PRESERVE MSE PER TIME =')
    print('=========================')
    for i, r in enumerate(results['preserve_mse_per_t']):
        print('t =%2d  |  %.6f' % (i, r))

    # Delta postion report
    results['pred_dpos_mse'] = np.sum(predicted_error[:,:,4:7]) \
            / np.sum(predicted_n * predicted_error.shape[1] * 3)
    print('=========================')
    print('==== D_POSITION MSE =====')
    print('=========================')
    print('Prediction: %f' % (results['pred_dpos_mse']))
    results['pred_dpos_mse_per_t'] = np.sum(predicted_error[:,:,4:7], axis=(1,2)) \
            / (predicted_n * predicted_error.shape[1] * 3)
    print('=========================')
    print(' D_POSITION MSE PER TIME ')
    print('=========================')
    for i, r in enumerate(results['pred_dpos_mse_per_t']):
        print('t =%2d  |  %.6f' % (i, r))

    # Store results in .pkl file
    saving_suffix = ''
    if not args.save_suf is None:
        saving_suffix = args.save_suf
    results_file = os.path.join(args.SAVE_DIR, 'quant_results_' + args.expId + \
            '_' + str(args.test_seed) + saving_suffix + '.pkl')
    with open(results_file, 'w') as f:
        cPickle.dump(results, f)
    print('=========================')
    print('Results stored in ' + results_file)


def retrieve_quantitative_results_parallel(
        sess, outputs, 
        n_validation_examples, args):
    quan_class = QuanTest(sess, outputs, args, args.test_seed)

    print('=========================')
    print('Evaluating %d validation examples...' % (n_validation_examples))

    # evaluate whole validation set
    predicted_error = np.zeros((1,1,1))
    preserve_error = np.zeros((1,1))
    predicted_n = np.zeros((1))
    n_pulls = args.test_n_pulls

    for i in trange(n_validation_examples / n_pulls, desc='Batch'):
        # for faster processing sequences of multiple data pulls are batched
        super_batch = []
        super_ranges = []
        for p in range(n_pulls):
            # pull batch
            batch = quan_class.pull_batch_random(args.qual_random_pull)
            ranges = quan_class.get_ranges(batch)

            super_batch.append(batch)
            if len(ranges) > 0:
                super_ranges.append(ranges)

        # pad and stack sequences for batch processing
        ranges = np.concatenate(super_ranges, axis=0)
        assert len(ranges) <= args.TEST_BATCH_SIZE, \
                ('Increase TEST_BATCH_SIZE=%d < len(ranges)=%d' % \
                (args.TEST_BATCH_SIZE, len(ranges)))
        max_range = max(ranges[:, 1] - ranges[:, 0])
        ranges_batch = {}
        for batch, ranges in zip(super_batch, super_ranges):
            for r in ranges:
                for k in batch:
                    if k not in ranges_batch:
                        ranges_batch[k] = []
                    data = batch[k][r[0]:r[1]]
                    data, _ = pad_and_mask_along_axis(data, max_range, axis=0)
                    ranges_batch[k].append(data)

        # pad to constant test batch size
        for k in ranges_batch:
            ranges_batch[k] = np.stack(ranges_batch[k], axis=0)
            ranges_batch[k], _ = pad_and_mask_along_axis(
                    ranges_batch[k], 
                    args.TEST_BATCH_SIZE, axis=0)

        batch = ranges_batch
        batch['input_particles'] = batch['full_particles'][:,:,:,0:4].copy()
        ranges = np.concatenate(super_ranges, axis=0)

        # UNROLL EVALUATION ACROSS TIME
        # run model and reuse output as input
        # once for ground truth and once for prediction
        predicted_sequences = quan_class.unroll_prediction_parallel(
                batch,
                False)

        predicted_error, preserve_error, predicted_n, final_pre_err \
                = compute_error(
                        predicted_sequences, 
                        predicted_error, 
                        preserve_error, 
                        predicted_n)
    results = {
            'pred_err_sq': predicted_error, 
            'preserve_error':preserve_error, 
            'final_pre_err':final_pre_err,
            'pred_n': predicted_n}

    save_and_report_results(results, args)
