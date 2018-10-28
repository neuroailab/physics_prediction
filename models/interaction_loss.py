import tensorflow as tf
import numpy as np
import cPickle
import pdb
import copy
from collections import OrderedDict


def get_velocity_loss(outputs,
        norm_loss=1,
        pred_particle_velocity=None,
        gt_particle_velocity=None,
        str_type="",
        sl=1,
        use_running_mean=False,
        ):
    if pred_particle_velocity is None:
        pred_particle_velocity = outputs['pred_particle_velocity']
    sparse_particles = outputs['sparse_particles']
    if gt_particle_velocity is None:
        gt_particle_velocity \
                = sparse_particles[:, -1, :, 15 : 18]

    valid_particles_mask = tf.not_equal(
            sparse_particles[:, -1, :, 14], 
            0)
    gt_next_vel_pts = tf.gather_nd(
            gt_particle_velocity,
            tf.where(valid_particles_mask))
    pred_next_vel_pts = tf.gather_nd(
            pred_particle_velocity,
            tf.where(valid_particles_mask))

    velocity_loss = (pred_next_vel_pts - gt_next_vel_pts) ** 2
    if norm_loss==1:
        depth_list = outputs['depth_list']
        max_depth = outputs['max_depth']

        all_vel_loss = []

        running_sum = None
        running_N = None
        if use_running_mean:
            with tf.variable_scope(
                    tf.get_variable_scope(), 
                    reuse=tf.AUTO_REUSE) as vscope:
                running_sum = tf.get_variable(
                        initializer=tf.constant_initializer(1.0),
                        shape=[max_depth],
                        dtype=tf.float64,
                        name='running_sum',
                        trainable=False)
                running_N = tf.get_variable(
                        initializer=tf.constant_initializer(1),
                        shape=[max_depth],
                        dtype=tf.int64,
                        name='running_N',
                        trainable=False)

        for curr_depth in xrange(max_depth):
            curr_mask = tf.equal(depth_list, curr_depth)
            curr_mask = tf.logical_and(valid_particles_mask, curr_mask)

            curr_gt_pos_pts = tf.gather_nd( \
                    tf.matmul(outputs['mult_mat_ts'][:,0], \
                    tf.matmul(outputs['mult_mat_rev_ts'] * \
                    tf.cast(tf.expand_dims(outputs['leaf_flag'], \
                    axis=1), tf.float32), \
                    sparse_particles[:,-1,:,0:3])),
                    tf.where(curr_mask))

            curr_gt_next_vel_pts = tf.gather_nd(
                    sparse_particles[:,-1,:,15:18],
                    tf.where(curr_mask))
            curr_pred_next_vel_pts = tf.gather_nd(
                    pred_particle_velocity,
                    tf.where(curr_mask))

            # 0 divistion fix
            #eps = 1e-1
            #mean_l2norm_vel += tf.cast(tf.less(mean_l2norm_vel, eps),
            #        tf.float32) * eps

            if use_running_mean:
                # calc running mean
                curr_sum = tf.reduce_sum(
                    curr_gt_pos_pts ** 2, axis=1)
                curr_sum0 = tf.cast(tf.equal(curr_sum, 0.0), tf.float32)
                curr_sum = tf.sqrt(curr_sum + curr_sum0) - curr_sum0
                running_sum = tf.assign(running_sum, \
                        running_sum + tf.cast( \
                        tf.one_hot(curr_depth, max_depth) * \
                        tf.reduce_sum(curr_sum, axis=0), tf.float64))
                running_N = tf.assign(running_N, \
                        running_N + tf.cast(
                        tf.one_hot(curr_depth, max_depth), tf.int64) * \
                        tf.cast(tf.shape(curr_sum)[0], tf.int64))
                running_mean = running_sum / tf.cast(running_N, tf.float64)
                mean_l2norm_vel = tf.cast(running_mean[curr_depth], tf.float32)
            else:
                mean_l2norm_vel = tf.reduce_mean(
                        tf.sqrt(
                            tf.reduce_sum(
                                curr_gt_next_vel_pts**2, 
                                axis=1)
                            ), 
                        axis=0,
                        )
            curr_vel_loss = ((curr_gt_next_vel_pts - curr_pred_next_vel_pts) \
                    / mean_l2norm_vel
                    ) ** 2
            all_vel_loss.append(curr_vel_loss)
        velocity_loss = tf.concat(all_vel_loss, axis=0)

    return velocity_loss, gt_next_vel_pts, pred_next_vel_pts


def get_preserve_distance_loss(outputs, 
        pred_particle_velocity=None,
        pos=None, sl=1,
        avd_obj_mask=0,
        ):
    if pos is None:
        pos = outputs['sparse_particles'][:,-1:,0:3]

    pred_next_vel = outputs['pred_particle_velocity']
    if pred_particle_velocity is not None:
        pred_next_vel = pred_particle_velocity
    pred_pos = pos + pred_next_vel

    if avd_obj_mask==0:
        obj1_mask = tf.cast(tf.equal(
            outputs['sparse_particles'][:,-1,:,14:15],
            outputs['OB1']), tf.float32)
        obj2_mask = tf.cast(tf.equal(
            outputs['sparse_particles'][:,-1,:,14:15],
            outputs['OB2']), tf.float32)
    else:
        # Avoiding masks, useful for more than 2 objects 
        # as the kNN relations already includes the object mask
        obj1_mask = tf.ones(
            outputs['sparse_particles'][:,-1,:,14:15].shape,
            dtype=tf.float32)
        obj2_mask = tf.ones(
            outputs['sparse_particles'][:,-1,:,14:15].shape,
            dtype=tf.float32)

    kNN_idx = tf.cast(outputs['kNN'][:,0,:,1:], tf.int32)
    batch_size, n_particles, n_neighbors = \
            kNN_idx.get_shape().as_list()
    batch_coordinates = tf.reshape(tf.tile(tf.expand_dims(
        tf.range(batch_size), axis=-1), \
                [1, n_particles * n_neighbors]), \
                [batch_size, n_particles, n_neighbors])
    kNN_idx = tf.stack([batch_coordinates, kNN_idx], axis=3)
    # (BS, N, NN, 3) with NN << N
    kNN_t0 = tf.gather_nd(pos, kNN_idx)
    kNN_t1 = tf.gather_nd(pred_pos, kNN_idx)

    if 'kNN_mask' in outputs:
        org_shape = kNN_t0.get_shape().as_list()
        kNN_t0 = tf.reshape(kNN_t0, [-1,org_shape[-1]])
        kNN_t1 = tf.reshape(kNN_t1, [-1,org_shape[-1]])
        pos = tf.reshape(tf.tile(tf.expand_dims(pos, axis=2), \
                [1,1,org_shape[2],1]), [-1,org_shape[-1]])
        pred_pos = tf.reshape(tf.tile(tf.expand_dims(pred_pos, axis=2), \
                [1,1,org_shape[2],  1]), [-1,org_shape[-1]])

        kNN_mask = tf.reshape(outputs['kNN_mask'][:,0,:,1:], [-1])

        kNN_t0 = tf.boolean_mask(kNN_t0, kNN_mask)
        kNN_t1 = tf.boolean_mask(kNN_t1, kNN_mask)
        pos = tf.boolean_mask(pos, kNN_mask)
        pred_pos = tf.boolean_mask(pred_pos, kNN_mask)
        distance_t0 = tf.reduce_sum((pos - kNN_t0) ** 2, axis=-1)
        distance0 = tf.cast(tf.equal(distance_t0, 0.0), tf.float32)
        distance_t0 = tf.sqrt(distance_t0 + distance0) - distance0
        distance_t1 = tf.reduce_sum((pred_pos - kNN_t1) ** 2, axis=-1)
        distance0 = tf.cast(tf.equal(distance_t1, 0.0), tf.float32)
        distance_t1 = tf.sqrt(distance_t1 + distance0) - distance0

        obj1_mask = tf.reshape(tf.tile(obj1_mask, [1,1,org_shape[2]]), [-1])
        obj2_mask = tf.reshape(tf.tile(obj2_mask, [1,1,org_shape[2]]), [-1])
        obj1_mask = tf.boolean_mask(obj1_mask, kNN_mask)
        obj2_mask = tf.boolean_mask(obj2_mask, kNN_mask)
    else:
        distance_t0 = tf.reduce_sum(
            (tf.expand_dims(pos, axis=2) - kNN_t0) ** 2, axis=-1)
        distance0 = tf.cast(tf.equal(distance_t0, 0.0), tf.float32)
        distance_t0 = tf.sqrt(distance_t0 + distance0) - distance0
        distance_t1 = tf.reduce_sum(
            (tf.expand_dims(pred_pos, axis=2) - kNN_t1) ** 2, axis=-1)
        distance0 = tf.cast(tf.equal(distance_t1, 0.0), tf.float32)
        distance_t1 = tf.sqrt(distance_t1 + distance0) - distance0

    distance_mask = tf.cast(tf.less(distance_t0, \
            outputs['preserve_distance_radius']), tf.float32)
    preserve_distance_loss = (distance_t1 - distance_t0) ** 2 * distance_mask
    # preserve_distance_loss = tf.abs(distance_t1 - distance_t0) * distance_mask

    preserve_distance_loss = \
            preserve_distance_loss * obj1_mask + \
            preserve_distance_loss * obj2_mask
    return preserve_distance_loss


def flex_l2_particle_loss(
        outputs, labels=None,
        alpha=0.5,
        preserve_distance_radius=0,
        use_running_mean=False,
        separate_return=False,
        load_kNN=True,
        debug=0,
        sl=1,
        add_gloss=0,
        avd_obj_mask=0,
        **kwargs):

    if 'final_loss' in outputs and not separate_return:
        return outputs['final_loss']

    # For concatenated things due to multiple gpus, just take the first iteam
    for k, v in outputs.items():
        if isinstance(v, list) and len(v) > 0:
            outputs[k] = v[0]

    outputs['preserve_distance_radius'] = preserve_distance_radius
    velocity_loss, mean_gt, mean_pred = get_velocity_loss(
            outputs, 
            norm_loss=1,
            sl=sl, 
            use_running_mean=use_running_mean)

    if add_gloss>0:
        # Get the global loss, note here norm_loss is always 0
        velocity_loss_global, mean_gt_global, mean_pred_global \
                = get_velocity_loss(
                        outputs, 
                        norm_loss=0, sl=sl,
                        pred_particle_velocity=outputs['g_pred_v'],
                        gt_particle_velocity=outputs['g_gt_v'],
                        )
        # Time it by 20 as that's the ratio between final values
        velocity_loss = tf.concat(
                [velocity_loss, 20*add_gloss*velocity_loss_global], 
                axis=0)
        mean_gt = tf.concat([mean_gt, mean_gt_global], axis=0)
        mean_pred = tf.concat([mean_pred, mean_pred_global], axis=0)

    retval = {
            'mean_gt': tf.reduce_mean(mean_gt),
            'mean_pred': tf.reduce_mean(mean_pred),
            'velocity_loss': tf.reduce_mean(velocity_loss),
            }
    retval['un_velocity_loss'] = tf.reduce_mean((mean_gt - mean_pred)**2)

    if preserve_distance_radius > 0:
        preserve_distance_loss = get_preserve_distance_loss(
                outputs,
                pos=outputs['sparse_particles'][:,-1,:,0:3],
                pred_particle_velocity=outputs['pred_particle_velocity'][:,:,:3],
                sl=sl,
                avd_obj_mask=avd_obj_mask,
                )
        retval['preserve_distance_loss'] = tf.reduce_mean(preserve_distance_loss)

        if not alpha==1:
            retval['final_loss'] \
                    = alpha * tf.reduce_mean(velocity_loss) \
                      + (1 - alpha) * tf.reduce_mean(preserve_distance_loss)
        else:
            retval['final_loss'] = tf.reduce_mean(velocity_loss)

    return retval
