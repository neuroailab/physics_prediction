import tensorflow as tf
import numpy as np
import cPickle
import pdb


def update_cached_dict(ret_dict, key, value, update_type='append'):
    if key not in ret_dict:
	if update_type=='update':
	    ret_dict[key] = value
	else:
	    ret_dict[key] = [value]
    else:
	if update_type=='update':
	    ret_dict[key] = max(ret_dict[key], value)
	else:
	    ret_dict[key].append(value)

    return ret_dict


def pad_to_tensors(ret_dict,
        dtype_dict={
            'kNN_list':tf.int16,
            'kNN_valid_flag':tf.bool,
            'new_particle':tf.bool,
            'no_wo_group_flag':tf.bool,
            'depth_list':tf.int32,
            'drt_father_list':tf.int32,
            'father_list':tf.int32}):

    new_ret_dict = {}

    all_num_p_rep = ret_dict['num_p_rep']
    ret_dict['new_particle'] = [np.ones(each_num_p_rep.shape) \
            for each_num_p_rep in all_num_p_rep]

    for key, value in ret_dict.items():
        if not isinstance(value, list):
            new_ret_dict[key] = value
            continue

        max_shape = list(value[0].shape)
        for curr_value in value[1:]:
            curr_shape = curr_value.shape
            for indx_shape in xrange(len(max_shape)):
                max_shape[indx_shape] = max(max_shape[indx_shape], \
                        curr_shape[indx_shape])

        new_value = []
        value_mask = []
        for curr_value in value:
            curr_shape = curr_value.shape
            pad_shape = [(0, m_shape - c_shape) for m_shape, c_shape in zip(max_shape,   curr_shape)]
            new_curr_value = np.pad(curr_value, pad_shape, mode='constant')
            new_value.append(new_curr_value)

            curr_mask = np.ones(curr_value.shape)
            curr_mask = np.pad(curr_mask, pad_shape, mode='constant')
            value_mask.append(curr_mask.astype(np.bool))

        new_value = np.asarray(new_value)
        value_mask = np.asarray(value_mask)
        #print(key, new_value.shape, value_mask.shape)

        curr_dtype = dtype_dict.get(key, tf.float32)
        new_ret_dict[key] = tf.constant(new_value, dtype=curr_dtype)
        new_ret_dict['%s_mask' % key] = tf.constant(value_mask, dtype=tf.bool)

    return new_ret_dict


def get_drt_father_list(all_data):
    drt_father_list = all_data['father_list']

    for each_p in xrange(len(drt_father_list)):
        if drt_father_list[each_p] is None:
            drt_father_list[each_p] = each_p

    drt_father_list = np.asarray(drt_father_list)
    return drt_father_list


def get_group_cached_ts(group_file):

    ret_dict = {}

    have_axis = None
    have_dist = None
    have_ms = None

    for each_group_file in group_file:
        print('Loading group file %s' % each_group_file)
        all_data = cPickle.load(open(each_group_file, 'r'))

        max_kNN_len = all_data['max_kNN_len']
        ret_dict = update_cached_dict(ret_dict, 'max_kNN_len',
                max_kNN_len, update_type='update')

        kNN_list = all_data['kNN_list']
        kNN_list = np.asarray(kNN_list)
        ret_dict = update_cached_dict(ret_dict, 'kNN_list', kNN_list)

        kNN_valid_flag = all_data['kNN_valid_flag']
        kNN_valid_flag = np.asarray(kNN_valid_flag)
        ret_dict = update_cached_dict(ret_dict, 'kNN_valid_flag', kNN_valid_flag)

        ret_dict = update_cached_dict(ret_dict, 'mult_mat', all_data['mult_mat'])
        ret_dict = update_cached_dict(ret_dict, 'mult_mat_space', \
                all_data['mult_mat_space'])
        ret_dict = update_cached_dict(ret_dict, 'mult_mat_rev', \
                all_data['mult_mat_rev'])
        ret_dict = update_cached_dict(ret_dict, 'num_p_rep', all_data['num_p_rep'])
        ret_dict = update_cached_dict(ret_dict, 'grav_flag', all_data['grav_flag'])

        depth_list, all_father_list = get_depth_father_list(all_data)
        max_depth = np.max(depth_list)
        ret_dict = update_cached_dict(ret_dict, 'max_depth',
                max_depth, update_type='update')
        ret_dict = update_cached_dict(ret_dict, 'depth_list', depth_list)
        ret_dict = update_cached_dict(ret_dict, 'father_list', all_father_list)

        drt_father_list = get_drt_father_list(all_data)
        ret_dict = update_cached_dict(ret_dict, 'drt_father_list', drt_father_list)

        no, no_wo_group = all_data['mult_mat'].shape
        no_wo_group_flag = np.arange(no)<no_wo_group
        ret_dict = update_cached_dict(ret_dict, 'no_wo_group_flag', no_wo_group_flag)

        if have_axis:
            assert 'all_axis' in all_data, \
                    "All group files should include all_axis! %s" % each_group_file
        if 'all_axis' in all_data:
            have_axis = True
            ret_dict = update_cached_dict(ret_dict, 'all_axis', \
                    all_data['all_axis'])
        else:
            have_axis = False

        if have_dist:
            assert 'all_dist' in all_data, \
                    "All group files should include all_dist! %s" % each_group_file
        if 'all_dist' in all_data:
            have_dist = True
            ret_dict = update_cached_dict(ret_dict, 'all_dist', \
                    all_data['all_dist'])
        else:
            have_dist = False

        if have_ms:
            assert 'L2H_attribute' in all_data, \
                    "All group files should include super node! %s" % each_group_file
        if 'L2H_attribute' in all_data:
            have_ms = True
            for key_now in ['L2H_attribute', 'WG_attribute', 'H2L_attribute', \
                    'L2H_division', 'WG_division']:
                ret_dict = update_cached_dict(ret_dict, key_now, \
                        all_data[key_now])
        else:
            have_dist = False

    ret_dict = pad_to_tensors(ret_dict)

    return ret_dict


def get_new_particles(
        sparse_particles, 
        mult_mat_space_ts,
        mult_mat_ts,
        new_num_p,
        temp_ones,
        ):
    len_of_state = sparse_particles.get_shape().as_list()[-1]
    # if len_of_state is 22, 
    # then the state also includes actual velocity (and that of next frame)
    include_velocity = len_of_state==22

    old_pos = sparse_particles[:,:,:,:3]
    new_pos = tf.matmul(mult_mat_space_ts, old_pos)
    old_vel = sparse_particles[:,:,:,4:7]
    new_global_vel = tf.matmul(mult_mat_space_ts, old_vel)
    new_vel = tf.matmul(mult_mat_ts, old_vel)

    old_f_t = sparse_particles[:,:,:,7:10]
    new_f_t = tf.matmul(mult_mat_space_ts, old_f_t)
    if not include_velocity:
        # Torque, not really used
        old_t_t = sparse_particles[:,:,:,10:13]
        new_t_t = tf.matmul(mult_mat_space_ts, old_t_t)
    else:
        # current actual velocity
        old_avel = sparse_particles[:,:,:,10:13]
        new_avel = tf.matmul(mult_mat_ts, old_avel)
        old_fu_avel = sparse_particles[:,:,:,19:22]
        new_fu_avel = tf.matmul(mult_mat_ts, old_fu_avel)
    old_fu_vel = sparse_particles[:,:,:,15:18]
    new_fu_vel = tf.matmul(mult_mat_ts, old_fu_vel)
    new_mass = tf.matmul(mult_mat_space_ts, sparse_particles[:,:,:,3:4]) \
            / new_num_p

    #new_ids = sparse_particles[0,0,0,13]*temp_ones
    new_ids = tf.matmul(mult_mat_space_ts, sparse_particles[:,:,:,13:14])
    #new_nids = sparse_particles[0,0,0,14]*temp_ones
    new_nids = tf.matmul(mult_mat_space_ts, sparse_particles[:,:,:,14:15])
    if not include_velocity:
        new_sparse_particles = tf.concat([
            new_pos,
            new_mass,
            new_vel,
            new_f_t,
            new_t_t,
            new_ids,
            new_nids,
            new_fu_vel,
            new_num_p], axis=3)
    else:
        new_sparse_particles = tf.concat([
            new_pos,
            new_mass,
            new_vel,
            new_f_t,
            new_avel,
            new_ids,
            new_nids,
            new_fu_vel,
            new_num_p,
            new_fu_avel], axis=3)
    return new_sparse_particles, new_global_vel


def multi_group_process(ts, sparse_particles, which_dataset, cached_ts_dict):
    max_kNN_len = cached_ts_dict['max_kNN_len']
    kNN_list = cached_ts_dict['kNN_list']
    kNN_list = tf.gather(kNN_list, which_dataset)
    kNN_valid_flag = cached_ts_dict['kNN_valid_flag']
    kNN_valid_flag = tf.gather(kNN_valid_flag, which_dataset)
    kNN_valid_flag = tf.expand_dims(kNN_valid_flag, axis=1)
    kNN_valid_flag = tf.tile(kNN_valid_flag, [1,ts,1,1])
    mult_mat = cached_ts_dict['mult_mat']
    mult_mat = tf.gather(mult_mat, which_dataset)
    mult_mat_space = cached_ts_dict['mult_mat_space']
    mult_mat_space = tf.gather(mult_mat_space, which_dataset)
    mult_mat_rev = cached_ts_dict['mult_mat_rev']
    mult_mat_rev = tf.gather(mult_mat_rev, which_dataset)
    num_p_rep = cached_ts_dict['num_p_rep']
    num_p_rep = tf.gather(num_p_rep, which_dataset)
    bs = num_p_rep.get_shape().as_list()[0]
    num_p = num_p_rep.get_shape().as_list()[1]
    grav_flag = cached_ts_dict['grav_flag']
    grav_flag = tf.gather(grav_flag, which_dataset)
    new_particle = tf.cast(cached_ts_dict['new_particle'], tf.float32)
    new_particle = tf.gather(new_particle, which_dataset)

    # Get the new kNN
    new_kNN = tf.expand_dims(kNN_list, axis=1)
    new_kNN = tf.tile(new_kNN, [1,ts,1,1])

    # Get the tensor representations
    mult_mat_space_ts = tf.expand_dims(mult_mat_space, axis=1)
    mult_mat_space_ts = tf.tile(mult_mat_space_ts, [1,ts,1,1])
    mult_mat_ts = tf.expand_dims(mult_mat, axis=1)
    mult_mat_ts = tf.tile(mult_mat_ts, [1,ts,1,1])
    new_num_p = tf.expand_dims(tf.expand_dims(num_p_rep, axis=1), axis=3)
    new_num_p = tf.tile(new_num_p, [1,ts,1,1])
    new_particle = tf.expand_dims(tf.expand_dims(new_particle, axis=1), axis=3)
    new_particle = tf.tile(new_particle, [1,ts,1,1])

    # Get the new sparse_particles
    new_sparse_particles, new_global_vel = get_new_particles(
            sparse_particles, 
            mult_mat_space_ts,
            mult_mat_ts,
            new_num_p,
            new_particle,
            )

    return max_kNN_len, kNN_valid_flag, new_kNN, new_sparse_particles, \
            grav_flag, mult_mat_rev, new_global_vel, mult_mat_space, mult_mat_ts


def get_depth_father_list(all_data):
    father_list = all_data['father_list']

    def get_particle_depth(start_idx):
        if father_list[start_idx] is None:
            return 0, [start_idx]
        else:
            curr_father = father_list[start_idx]
            temp_depth, temp_father_list = get_particle_depth(curr_father)
            return 1 + temp_depth, temp_father_list + [start_idx]

    all_father_list = []
    depth_list = []
    for each_p in xrange(len(father_list)):
        temp_depth, temp_father_list = get_particle_depth(each_p)
        depth_list.append(temp_depth)
        all_father_list.append(temp_father_list)

    depth_list = np.asarray(depth_list)
    max_depth = np.max(depth_list)

    for each_p in xrange(len(father_list)):
        for _ in xrange(len(all_father_list[each_p]), max_depth+1):
            all_father_list[each_p].append(each_p)

    all_father_list = np.asarray(all_father_list)

    return depth_list, all_father_list
