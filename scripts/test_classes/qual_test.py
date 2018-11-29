from base_test import BaseTest
from tqdm import tqdm
import numpy as np
import os
import cPickle


class QualTest(BaseTest):
    """
    Class for qualitative testing
    """
    def __init__(self, *args, **kwargs):
        super(QualTest, self).__init__(*args, **kwargs)

    def _change_shape(self, particle):
        feed_particle = particle[np.newaxis]
        return feed_particle

    def _get_copy(self, frame, copy_tensor):
        if frame < copy_tensor.shape[0]:
            return copy_tensor[frame].copy()
        else:
            return copy_tensor[-1].copy()

    def _move_by(self, move_tensor):
        move_tensor[:-1, :, :] = move_tensor[1:, :, :]
        return move_tensor

    def _add_by(
            self,
            orig_tensor, add_tensor, 
            val, part_indx):
        orig_tensor[-1, val, part_indx[0]:part_indx[1]] += add_tensor[val]
        return orig_tensor

    def _replace_by(
            self,
            orig_tensor, replace_tensor, 
            val, part_indx):
        orig_tensor[-1, val, part_indx[0]:part_indx[1]] = replace_tensor[val]
        return orig_tensor

    def _coll_online(
            self,
            obj_pair, particle, collision, 
            static_particles=None, initial_distances=None):
        pair_0_pos = particle[-1, obj_pair[:,0], 0:3]
        if static_particles is not None: 
            assert initial_distances is None
            # just use first timestep since it doesn't vary
            pair_1_pos = static_particles[obj_pair[:,1], 0:3]
        else:
            pair_1_pos = particle[-1, obj_pair[:,1], 0:3]

        _all_coll_dist = np.linalg.norm(
                pair_0_pos - pair_1_pos,
                axis=-1)
        _all_coll_dist = np.concatenate(
                [obj_pair, _all_coll_dist[:, np.newaxis]], 
                axis=-1)

        if initial_distances is not None:
            _all_init_dist = initial_distances[obj_pair[:,0], obj_pair[:,1]]
            _all_coll_dist = np.concatenate(
                    [_all_coll_dist, _all_init_dist[:, np.newaxis]], 
                    axis=-1)

        nc_real = collision.shape[1]
        _all_coll_dist = sorted(_all_coll_dist, key=lambda x:x[2])
        _all_coll_dist = np.asarray(_all_coll_dist[:nc_real])

        nc_curr = _all_coll_dist.shape[0]
        if nc_curr < nc_real:
            remain_len = nc_real - nc_curr
            if initial_distances:
                padding = np.zeros((remain_len, 4))
            else:
                padding = np.zeros((remain_len, 3))
            padding[:,2] = 1000000
            _all_coll_dist = np.concatenate([_all_coll_dist, padding], axis=0)

        collision[-1, :, :] = _all_coll_dist
        return collision

    def unroll_prediction(
            self,
            batch,
            use_ground_truth_for_unroll):
        args = self.args

        predicted_sequences = []

        self.obj_pair = None
        self.same_part_pair = None
        self.sta_obj_pair = None
        self.initial_distances = None
        if args.with_self == 1:
            self.initial_distances = self.get_init_prtcl_dist(args.dataset)

        for i, r in tqdm(enumerate(self.ranges)):
            init = True
            # unroll beyond ground truth
            if args.pred_unroll_length is not None:
                if args.pred_unroll_length > r[1] - r[0] \
                        and not use_ground_truth_for_unroll:
                    r = r.copy()
                    r[1] = r[0] + args.pred_unroll_length

            self.predicted_particles = []
            self.true_particles = [] 
            for frame in range(r[0], r[1]):
                if frame >= args.MODEL_BATCH_SIZE \
                        and use_ground_truth_for_unroll:
                    break

                if init:
                    init = False
                    particle = self.unroll_init(
                            frame, 
                            batch, 
                            use_ground_truth_for_unroll,
                            )

                    # Only use the first times step to compute collision pairs
                    if args.coll_online == 1:
                        particle_at_0 = particle[0]
                        self._get_obj_pair(particle_at_0)
                        self._get_self_pair(particle_at_0)
                        self._get_sta_obj_pair(particle_at_0)

                else:
                    self.unroll_noninit(
                            frame, batch, 
                            use_ground_truth_for_unroll)

            predicted_sequences.append({
                'predicted_particles': np.stack(
                    self.predicted_particles, 
                    axis=0),
                'true_particles': np.stack(self.true_particles, axis=0),
                'input_particles': batch['input_particles']})

        return predicted_sequences


def save_results_pickle(predicted_sequences, use_ground_truth_for_unroll, args):
    # Store results and ground truth in .pkl file
    saving_suffix = ''
    if not args.save_suf is None:
        saving_suffix = args.save_suf
    if use_ground_truth_for_unroll:
        results_file = os.path.join(args.SAVE_DIR, 'true_results_' + args.expId + \
                '_' + str(args.test_seed) + saving_suffix + '.pkl')
        with open(results_file, 'w') as f:
            cPickle.dump(predicted_sequences, f)
    else:
        results_file = os.path.join(args.SAVE_DIR, 'results_' + args.expId + \
                '_' + str(args.test_seed) + saving_suffix + '.pkl')
        with open(results_file, 'w') as f:
            cPickle.dump(predicted_sequences, f)
    print('=========================')
    print('Results stored in ' + results_file)


def retrieve_qualitative_examples(sess, outputs, args):
    qual_class = QualTest(sess, outputs, args, args.test_seed)

    # start at a random batch and pull data
    batch = qual_class.pull_batch_random(args.qual_random_pull)
    batch['input_particles'] = batch['full_particles'].copy()

    # determine range of test sequences
    ranges = qual_class.get_ranges(batch)

    print('=========================')
    print('Creating %d prediction sequences for seed %d...' \
            % (len(ranges), args.test_seed))

    # UNROLL EVALUATION ACROSS TIME
    # run model and reuse output as input
    # once for ground truth and once for prediction
    for use_ground_truth_for_unroll in [True, False]:
        predicted_sequences = qual_class.unroll_prediction(
                batch,
                use_ground_truth_for_unroll,
                )
        save_results_pickle(predicted_sequences, use_ground_truth_for_unroll, args)
