import tensorflow as tf
import numpy as np
import cPickle
import pdb
import copy
from collections import OrderedDict

from physics_postprocess import PhysicsPostprocess
from model_building_blocks import ConvNet, hidden_mlp
import interaction_utils as utils
from interaction_loss import flex_l2_particle_loss


def add_batch_index(x):
    shape = x.get_shape().as_list()
    batch_index = tf.tile(
            tf.reshape(
                tf.range(shape[0]),
                [shape[0]] + [1] * (len(shape) - 1)
                ),
            [1] + shape[1:]
            )
    x = tf.stack([batch_index, tf.cast(x, tf.int32)], axis = len(shape))
    return x


def one_hot_column(x, ndims, axis):
    assert axis < ndims, 'axis must be smaller than the vector dimension'
    one_hot = np.zeros(ndims).astype(np.float32)
    one_hot[axis] = 1
    one_hot = tf.tile(tf.reshape(one_hot, [1, ndims]), [tf.shape(x)[0], 1])
    return one_hot


def sum_effects(E, Rr, bs, no):
    de = E.get_shape().as_list()[1]
    RrSum = tf.reshape(Rr, [-1, Rr.get_shape().as_list()[-1]])
    cum = tf.cumprod([bs, no], axis=-1, reverse=True, exclusive=True)
    RrSum = tf.matmul(RrSum, tf.expand_dims(cum, axis=-1))
    RrSum = tf.reshape(RrSum, [-1])

    E = tf.unsorted_segment_sum(E, RrSum, bs*no)
    E = tf.reshape(E, [bs, no, de])
    return E


class HRNModel(object):
    def __init__(
            self, 
            inputs, OB1, OB2, 
            batch_size=None, cfg=None, my_test=False,
            test_batch_size=1, alpha=0.5, 
            number_of_kNN=7, room_center='10,0.2,10',
            use_collisions=0, use_actions=0, group_file=None,
            max_collision_distance=0.5,
            sl=1, use_static=0, use_self=0, static_path=None,
            mass_adjust=None,
            preserve_distance_radius=100000,
            vary_grav=0, vary_stiff=0, both_stiff_vary=0,
            add_dist=0, add_gloss=0,
            gravity_term=9.81, debug=0, avd_obj_mask=0, 
            is_fluid = False, 
            is_moving = 'is_moving',
            use_running_mean=False,
            **kwargs):

        self.inputs = inputs
        self.OB1 = OB1
        self.OB2 = OB2
        self.batch_size = batch_size
        self.cfg = cfg
        self.my_test = my_test
        self.test_batch_size = test_batch_size
        self.alpha = alpha
        self.room_center = room_center
        self.use_collisions = use_collisions
        self.use_actions = use_actions
        self.group_file = group_file
        self.max_collision_distance = max_collision_distance
        self.sl = sl
        self.use_static = use_static
        self.use_self = use_self
        self.static_path = static_path
        self.mass_adjust = mass_adjust
        self.preserve_distance_radius = preserve_distance_radius
        self.vary_grav = vary_grav
        self.vary_stiff = vary_stiff
        self.both_stiff_vary = both_stiff_vary
        self.add_dist = add_dist
        self.add_gloss = add_gloss
        self.gravity_term = gravity_term
        self.debug = debug
        self.avd_obj_mask = avd_obj_mask
        self.is_fluid = is_fluid
        self.is_moving = is_moving
        self.use_running_mean = use_running_mean

        assert self.OB1 < self.OB2, (self.OB1, self.OB2)
        assert self.group_file != None, \
                "Please specify group file"

        self.nk = number_of_kNN

        # Initialize base network
        self.model_builder = ConvNet(debug=debug, **kwargs)

    def _get_group_and_which_dataset(self):
        """
        which_dataset is used for indexing multiple datasets within one batch:
            the shape should be [batch_size], each number is 0 to N_datasets
        """
        inputs = self.inputs

        self.cached_ts_dict = utils.get_group_cached_ts(self.group_file)

        which_dataset = inputs['which_dataset'][:,-1]
        which_dataset = tf.reshape(which_dataset, [-1])
        which_dataset = tf.cast(which_dataset, tf.int32)
        if self.my_test:
            # Assuming only one dataset
            which_dataset = tf.zeros([self.test_batch_size], tf.int32)
        org_which_dataset = which_dataset

        self.which_dataset = which_dataset
        # org_which_dataset is before potential randomizations
        self.org_which_dataset = org_which_dataset

    def _post_process_data(self):
        """
        Read in and do necessary post-processing to inputs
        including subtracting the room center, potential random mass,
        """
        post_proc = PhysicsPostprocess(
                self.inputs,                 
                static_path=self.static_path,
                use_static=self.use_static,
                which_dataset=self.org_which_dataset,
                room_center=self.room_center,
                sl=self.sl,
                mass_adjust=self.mass_adjust,
                )
        self.post_proc = post_proc
        self.org_input_ts = post_proc.inputs
        self.inputs = copy.copy(post_proc.inputs)
        self.sparse_particles = self.inputs['sparse_particles']

    def _set_test_placeholders(self):
        """
        As test is done by feed_dict, we need to set the placeholders here
        """
        num_particles = self.sparse_particles.get_shape().as_list()[2]
        self.full_particles_place = tf.placeholder(
                tf.float32,
                [self.test_batch_size, self.sl, num_particles, 22], 
                'particle_input')
        self.sparse_particles = self.post_proc.adjust_mass_for_particles(
                self.full_particles_place)

        self.placeholders = {}
        place_keys = []

        if self.use_collisions==1:
            place_keys.append('collision')
        if self.use_self==1:
            place_keys.append('self_collision')
        if self.use_static==1:
            place_keys.append('static_collision')
        if self.vary_grav==1:
            place_keys.append('gravity')
        if self.vary_stiff==1:
            place_keys.append('stiffness')

        for key in place_keys:
            org_inpt_ts = self.inputs[key]
            org_shape = org_inpt_ts.shape.as_list()
            place_shape = [self.test_batch_size] + org_shape[1:]
            curr_placeholder = tf.placeholder(
                    org_inpt_ts.dtype,
                    place_shape,
                    '%s_input' % key
                    )
            self.placeholders[key] = curr_placeholder
            self.inputs[key] = curr_placeholder
            self.inputs['%s_mask' % key] = tf.ones_like(
                    curr_placeholder, 
                    dtype=tf.bool)

    def _set_attributes_from_inputs(self):
        inputs = self.inputs

        if self.use_collisions == 1:
            assert 'collision' in inputs, "There needs to be collision in input"
            self.collision = inputs['collision'] # (BS, sl, NC, 3)
            self.nc = self.collision.get_shape().as_list()[2]
            self.collision_mask = inputs['collision_mask']
        if self.use_self == 1:
            assert 'self_collision' in inputs, \
                    'There needs to be self collision in input'
            self.self_collision = inputs['self_collision'] # (BS, sl, NCC, 3)
            self.ncc = self.self_collision.get_shape().as_list()[2]
            self.self_collision_mask = inputs['self_collision_mask']
        if self.use_static == 1:
            assert 'static_particles' in inputs, \
                    'There needs to be static particles in input'
            self.static_particles = inputs['static_particles']
            self.static_collision = inputs['static_collision'] # (BS, sl, NSC, 3)
            self.nsc = self.static_collision.get_shape().as_list()[2]
            self.static_collision_mask = inputs['static_collision_mask']

        if self.vary_grav == 1:
            self.inp_gravity = inputs['gravity']

        self.inp_stiff = None
        if self.vary_stiff == 1:
            self.inp_stiff = inputs['stiffness']
            self.num_stiff_values = self.inp_stiff.shape[-1]
            if self.both_stiff_vary == 1:
                assert self.num_stiff_values > 1, \
                        "Should have multiple stiffness values"

    def _set_bs_ts_no(self):
        self.bs = self.sparse_particles.get_shape().as_list()[0]
        self.ts = self.sparse_particles.get_shape().as_list()[1]
        self.no = self.sparse_particles.get_shape().as_list()[2]

    def _apply_group(self):
        # Update raw information according to grouping results
        self.father_list = None
        self.max_kNN_len, self.kNN_valid_flag, \
                self.kNN, self.sparse_particles, \
                self.grav_flag, self.mult_mat_rev_ts, \
                self.new_global_vel, \
                self.mult_mat_space_ts, self.mult_mat_ts \
                = utils.multi_group_process(
                        self.ts, 
                        self.sparse_particles, 
                        self.which_dataset, 
                        self.cached_ts_dict, 
                        )
        self.kNN_mask = self.kNN_valid_flag
        self.depth_list = self.cached_ts_dict['depth_list'] # (Num_of_dataset, max_num_particles)
        self.depth_list = tf.gather(self.depth_list, self.which_dataset) # (BS, max_num_particles)
        self.max_depth = self.cached_ts_dict['max_depth']

        self.leaf_flag = self.cached_ts_dict['no_wo_group_flag']
        self.leaf_flag = tf.gather(
                self.leaf_flag, 
                self.which_dataset)
        
        self.father_list = self.cached_ts_dict['father_list']
        self.father_list = tf.gather(
                self.father_list, 
                self.which_dataset) # (BS, max_n_particles, max_depth)

        # Get the ground truth distance between particles if needed
        if self.add_dist==1:
            assert 'all_dist' in cached_ts_dict, \
                    "Grouping files must include all distance matrix"
            all_dist = cached_ts_dict['all_dist']
            # (BS, max_num_particles, max_num_particles)
            all_dist_ts = tf.gather(all_dist, which_dataset) 

            self.all_dist_ts = all_dist_ts

        assert self.max_kNN_len > self.nk, \
                "kNN must be smaller than maximal number %i" % self.max_kNN_len

    def _get_all_states(self):
        # Unpacking the states, transposing it so that time is dimension 2
        self.state_pos = tf.transpose(
                self.sparse_particles[:,:,:,0:3], 
                [0,2,1,3])
        self.state_mass = tf.transpose(
                self.sparse_particles[:,:,:,3:4],
                [0,2,1,3])
        self.state_local_delta_pos = tf.transpose(
                self.sparse_particles[:,:,:,4:7],
                [0,2,1,3])
        self.state_global_delta_pos = tf.transpose(
                self.new_global_vel,
                [0,2,1,3])

    def _set_short_names_for_dims(self):
        # DR: Dimension Relation
        self.dr = 1

        # DX: Dimension eXternal effect
        self.dx = 3
        if self.vary_grav==1:
            self.dx = self.dx * self.sl

        # DP: Dimension Prediction (next delta position)
        self.dp = 3

        # NR: Number of Relations for one example
        self.nr = self.no * self.nk

        # DS: Dimension State
        self.ds = 7 * self.sl

    def _set_gravity(self):
        ''''
        Create gravity effect matrix
        (BS, NO, DX)
        There is one thing special about gravity:
            as we are predicting local motions, we only apply gravity to the 
            root particle for each object.
        '''
        ## Adding time dimension temporarly if it's perhaps varying every time
        ## otherwise, gravity_term will only be one float number
        if self.vary_grav==1:
            self.gravity_term = tf.tile(
                    tf.expand_dims(self.inp_gravity, axis=2), 
                    [1,1,no,1]
                    )

        state_gravity \
                = tf.reshape(
                        self.grav_flag, 
                        [self.bs, 1, self.no, 1]
                        ) \
                * self.gravity_term

        # Create gravity force vector [x,y,z] = [0,g,0]
        grav_mult = tf.concat(
                [tf.zeros(tf.shape(state_gravity)),
                 tf.ones(tf.shape(state_gravity)), 
                 tf.zeros(tf.shape(state_gravity))], 
                axis=-1
                )
        state_gravity = state_gravity * grav_mult # (bs,sl,no,3)

        state_gravity = tf.transpose(state_gravity, [0,2,1,3])
        if self.vary_grav==0:
            state_gravity = state_gravity[:,:,-1,:] # (bs, no, 3)
        else:
            state_gravity = tf.reshape(
                    state_gravity, 
                    [self.bs, self.no, -1]
                    ) # (bs,no,3*sl)
        self.state_gravity = state_gravity

    def ___combine_pos_mass_delta_pos(
            self, 
            all_state_pos_mass, 
            all_state_delta_pos):
        # For legacy purpose, put all delta pos in the last of states
        all_state_pos_mass = tf.reshape(all_state_pos_mass, [-1, 4 * self.sl])
        all_state_delta_pos = tf.reshape(all_state_delta_pos, [-1, 3 * self.sl])

        all_state = tf.concat(
                [all_state_pos_mass, all_state_delta_pos], 
                axis=-1)
        all_state = tf.reshape(all_state, [-1, self.ds])
        return all_state

    def __get_static_states(self, particle_indexes):
        SO = self.static_particles
        SO = tf.transpose(SO, [0, 2, 1, 3])
        all_state_pos_mass = tf.gather_nd(SO[:, :, :, 0:4], particle_indexes)
        all_state_delta_pos = tf.gather_nd(SO[:, :, :, 4:7], particle_indexes)
        all_state = self.___combine_pos_mass_delta_pos(
                all_state_pos_mass,
                all_state_delta_pos)
        return all_state

    def __get_pos_mass_states(self, particle_indexes):
        all_state_pos = tf.gather_nd(self.state_pos, particle_indexes)
        all_state_mass = tf.gather_nd(self.state_mass, particle_indexes)
        all_state_pos_mass = tf.concat([all_state_pos, all_state_mass], axis=-1)
        return all_state_pos_mass

    def __get_states(
            self, 
            particle_indexes,
            local_delta_pos=True):
        all_state_pos_mass = self.__get_pos_mass_states(particle_indexes)

        if local_delta_pos:
            all_state_delta_pos = tf.gather_nd(
                    self.state_local_delta_pos,
                    particle_indexes)
        else:
            all_state_delta_pos = tf.gather_nd(
                    self.state_global_delta_pos,
                    particle_indexes)

        all_state = self.___combine_pos_mass_delta_pos(
                all_state_pos_mass,
                all_state_delta_pos)
        return all_state

    def __prepare_for_collisions(self):
        # Prepare collision data
        self.ds_c = 3
        # Receiver index for collisions
        self.Rr_C = tf.zeros([0, 2], dtype=tf.int32)
        # Receiver states for collisions
        self.C_Rr = tf.zeros([0, self.ds], dtype=tf.float32)
        # Sender states for collisions
        self.C_Rs = tf.zeros([0, self.ds], dtype=tf.float32)

        # Additional states, including one hot vector for time
        self.C_Ra = tf.zeros([0, self.sl], dtype=tf.float32)

        # Placeholder for results (in case no collision happened)
        cfg = self.cfg
        output_dim_phiC = cfg['phiC'][cfg['phiC_depth']]['num_features']
        self.PhiC = tf.zeros(
                [0, output_dim_phiC], 
                dtype=tf.float32,
                )

    def __add_one_time_collisions(
            self, collision, 
            which_time, valid_col_mask,
            static_coll=False):
        # Dynamic graph (between object relations)
        # (BS, NC, 2, 2) 
        # psender_preceiver_tuple, batch_idx_particle_idx
        col = collision[:, :, 0:2]
        col = add_batch_index(col)

        # Filter out relationships with too large distance
        # and invalid relationships caused by padding if needed
        col = tf.gather_nd(col, tf.where(valid_col_mask)) # (NO_valid_coll, 2, 2)

        # Create collision sender and receiver matrices
        col = tf.cast(col, tf.int32)

        if not static_coll:
            Rs_col = col[:, 0] # (NO_valid_coll, 2)
            Rr_col = col[:, 1] # (NO_valid_coll, 2)
            # As collision relations are bidirectional
            # but precomputed collision pairs are directional
            Rr_col_2ways = tf.concat([Rr_col, Rs_col], axis=0) # (NO_valid_coll*2, 2)
            Rs_col_2ways = tf.concat([Rs_col, Rr_col], axis=0) # (NO_valid_coll*2, 2)

            # Use global velocity in the collision state
            _C_Rr = self.__get_states(Rr_col_2ways, local_delta_pos=False)
            _C_Rs = self.__get_states(Rs_col_2ways, local_delta_pos=False)

            _C_Ra = tf.reshape(
                    one_hot_column(_C_Rr, self.sl, which_time), 
                    [-1, self.sl])

            self.Rr_C = tf.concat([self.Rr_C, Rr_col_2ways], axis=0)
        else:
            Rr_col = col[:, 0] # (NO_valid_coll, 2)
            Rs_col = col[:, 1] # (NO_valid_coll, 2)

            _C_Rr = self.__get_states(Rr_col, local_delta_pos=False)
            _C_Rs = self.__get_static_states(Rs_col)

            _C_Ra = tf.reshape(
                    one_hot_column(_C_Rr, self.sl, which_time), 
                    [-1, self.sl])

            self.Rr_C = tf.concat([self.Rr_C, Rr_col], axis=0)

        self.C_Rr = tf.concat([self.C_Rr,_C_Rr], axis=0)
        self.C_Rs = tf.concat([self.C_Rs,_C_Rs], axis=0)
        self.C_Ra = tf.concat([self.C_Ra,_C_Ra], axis=0)

    def __add_collisions(self):
        for which_time in range(self.sl):
            curr_coll = self.collision[:, which_time, :, :] # (BS, NC, 3)
            valid_col_mask = tf.less(
                    curr_coll[:, :, 2],
                    self.max_collision_distance
                    )
            valid_col_mask = tf.logical_and(
                    valid_col_mask,
                    self.collision_mask[:, which_time, :, 2]
                    )
            self.__add_one_time_collisions(
                    curr_coll, which_time, 
                    valid_col_mask)

    def __add_self_collisions(self):
        for which_time in range(self.sl):
            curr_coll = self.self_collision[:, which_time, :, :] # (BS, NCC, 3)
            if self.is_fluid:
                valid_col_mask = tf.less(
                        curr_coll[:,:,2],
                        self.max_collision_distance)
            else:
                valid_col_mask = tf.logical_and(
                        tf.less(curr_coll[:,:,2],
                                self.max_collision_distance),
                        tf.less(curr_coll[:,:,2],
                                curr_coll[:,:,3]))
            valid_col_mask = tf.logical_and(
                    valid_col_mask,
                    self.self_collision_mask[:, which_time, :, 2]
                    )
            self.__add_one_time_collisions(
                    curr_coll, which_time, 
                    valid_col_mask)

    def __add_static_collisions(self):
        for which_time in range(self.sl):
            curr_coll = self.static_collision[:, which_time, :, :] # (BS, NSC, 3)
            valid_col_mask = tf.less(
                    curr_coll[:, :, 2],
                    self.max_collision_distance
                    )
            valid_col_mask = tf.logical_and(
                    valid_col_mask,
                    self.static_collision_mask[:, which_time, :, 2]
                    )
            self.__add_one_time_collisions(
                    curr_coll, which_time, 
                    valid_col_mask,
                    static_coll=True)

    def _get_phiC(self):
        self.__prepare_for_collisions()

        if self.use_collisions==1:
            self.__add_collisions()

        if self.use_self==1:
            self.__add_self_collisions()

        if self.use_static==1:
            self.__add_static_collisions()

        if self.use_collisions==1 \
                or self.use_static==1 \
                or self.use_self==1:
            C = tf.concat([self.C_Rr, self.C_Rs, self.C_Ra], axis = 1)

            self.PhiC = hidden_mlp(
                    C, self.model_builder, 
                    self.cfg, 'phiC',
                    reuse_weights=False, train=not self.my_test, 
                    debug=self.debug)

    def _get_phiH(self):
        # Process single particle history information
        # For legacy issue, it's also named as phiS
        assert 'phiS' in self.cfg, "Please set the network shape in cfg!"
        Rr_S = tf.cast(tf.where(self.leaf_flag), tf.int32)
        S_S = self.__get_states(Rr_S, local_delta_pos=True)
        S_Ra = tf.reshape(
                one_hot_column(S_S, 4, 0), 
                [-1, 4])
        S_S = tf.concat([S_S, S_Ra], axis=1)
        self.PhiH = hidden_mlp(
                S_S, self.model_builder, 
                self.cfg, 'phiS',
                reuse_weights=False, train=not self.my_test, 
                debug=self.debug)
        self.Rr_S = Rr_S

    def __prepare_for_phiF(self):
        # Rr_X is just empty
        Rr_X = tf.zeros([0, 2], tf.int32)
        F_Rr = self.__get_states(Rr_X, local_delta_pos=True)
        F_E = tf.zeros([0, self.dx], tf.float32)
        F_Ra = tf.reshape(one_hot_column(F_E, 1, 0), [-1, 1])

        self.Rr_X = Rr_X
        self.F_Rr = F_Rr
        self.F_E = F_E
        self.F_Ra = F_Ra

    def __add_actions(self):
        # Get actions and create action receiver matrix
        # Adding previous force to the PhiF network
        A = self.sparse_particles[:, :, :, 7:10]
        A = tf.transpose(A, [0, 2, 1, 3])
        A = tf.reshape(A, [self.bs, self.no, -1]) # (bs, no, 3*sl)

        # Get particles with forces applied
        Rr_act_raw = tf.greater(tf.reduce_sum(tf.abs(A), axis=2), 0)
        # Only apply forces to leaf particles
        Rr_act_raw = tf.logical_and(Rr_act_raw, self.leaf_flag)
        Rr_act_raw = tf.where(Rr_act_raw)
        Rr_act = tf.cast(Rr_act_raw, tf.int32)

        # Force network translates external forces into effects
        A_Rr = self.__get_states(Rr_act, local_delta_pos=True)
        A_E = tf.reshape(
                tf.gather_nd(A, Rr_act), 
                [-1, A.get_shape().as_list()[-1]])
        A_Ra = tf.reshape(one_hot_column(A_E, 1, 0), [-1, 1])

        # Concatenate actions and external forces
        self.Rr_X = Rr_act
        self.F_Rr = A_Rr
        self.F_E = A_E
        self.F_Ra = A_Ra

    def _get_phiF(self):
        # Process external forces and gravity
        self.__prepare_for_phiF()

        if self.use_actions==1:
            self.__add_actions()

        F = tf.concat([self.F_Rr, self.F_E, self.F_Ra], axis = 1)
        self.PhiF = hidden_mlp(
                F, self.model_builder, 
                self.cfg, 'phiF',
                reuse_weights=False, train=not self.my_test, 
                debug=self.debug)

    def _combine_phiC_phiH_phiF(self):
        # Sum up all external effects
        raw_E_all_X = tf.concat([self.PhiF, self.PhiC, self.PhiH], axis=0)
        Rr_all_X = tf.concat([self.Rr_X, self.Rr_C, self.Rr_S], axis=0)
        E_all_X = sum_effects(
                raw_E_all_X, Rr_all_X, 
                self.bs, self.no) # (BS, NO, DE)

        self.E_all_X = E_all_X
        self.de = E_all_X.get_shape().as_list()[2]

    def __set_inp_stiff(self):
        # Do some processing to inp_stiff
        def _expand_and_tile(inp_stiff):
            inp_stiff = tf.expand_dims(inp_stiff, axis=1)
            inp_stiff = tf.tile(inp_stiff, [1, self.no, 1])
            return inp_stiff

        inp_stiff = self.inp_stiff
        if self.both_stiff_vary==0:
            inp_stiff = tf.squeeze(inp_stiff, axis=-1)
            inp_stiff = _expand_and_tile(inp_stiff)
        elif self.both_stiff_vary==1:
            #TODO: support more than two objects here?
            obj1_mask = tf.cast(
                    tf.equal(self.sparse_particles[:,0,:,14:15], self.OB1), 
                    tf.float32)
            obj2_mask = tf.cast(
                    tf.equal(self.sparse_particles[:,0,:,14:15], self.OB2), 
                    tf.float32)

            inp_stiff_obj1 = inp_stiff[:,:,0]
            inp_stiff_obj1 = _expand_and_tile(inp_stiff_obj1)
            inp_stiff_obj2 = inp_stiff[:,:,1]
            inp_stiff_obj2 = _expand_and_tile(inp_stiff_obj2)

            inp_stiff = inp_stiff_obj1*obj1_mask + inp_stiff_obj2*obj2_mask
        else:
            #TODO: support rigid and soft collision here?
            raise NotImplementedError(
                    "both_stiff_vary of other values not implemented!")
        self.inp_stiff = inp_stiff

    def _prepare_for_HRN(self):
        self.just_one_mat = tf.ones([self.bs, self.no, 1], dtype=tf.float32)

        dshape = self.depth_list.get_shape().as_list()
        # As father_list is from closer ancestors to further ancestors
        depth_father_list = tf.tile(
                tf.reshape(
                    tf.range(self.max_depth, dtype=tf.int32),
                    [1, 1, self.max_depth]), 
                [dshape[0], dshape[1], 1])
        depth_mask = tf.less(
                depth_father_list,
                tf.tile(
                    tf.reshape(
                        self.depth_list, 
                        [dshape[0], dshape[1], 1]), 
                    [1, 1, self.max_depth]))
        self.depth_mask = depth_mask

        # Create list of keys for unshared PhiR
        self.phiR_list = ['phiR', 'phiR', 'phiR']
        self.reuse_list = [False, True, True]

        # Create father-leaf pairs (TODO: father should be ancestor!)
        Rr_father = self.father_list[:,:,:-1]
        Rr_father = add_batch_index(Rr_father)
        Rs_leaf = tf.tile(
                tf.reshape(tf.range(self.no), [1, self.no, 1]), 
                [self.bs, 1, Rr_father.get_shape().as_list()[2]])
        Rs_leaf = add_batch_index(Rs_leaf)

        self.Rr_father = Rr_father
        self.Rs_leaf = Rs_leaf

        if self.vary_stiff==1:
            self.__set_inp_stiff()

    def _get_L2H_receiver_sender(self):
        depth_mask_L2H = tf.logical_and(
                self.depth_mask, 
                tf.tile(
                    tf.expand_dims(self.leaf_flag, axis=2), 
                    [1, 1, self.max_depth]))
        depth_mask_indx_L2H = tf.where(depth_mask_L2H)

        Rr_father_L2H = tf.gather_nd(self.Rr_father, depth_mask_indx_L2H)
        Rs_leaf_L2H = tf.gather_nd(self.Rs_leaf, depth_mask_indx_L2H)
        return Rr_father_L2H, Rs_leaf_L2H

    def _get_L2H_states(self, Rr_father_L2H, Rs_leaf_L2H):
        L2H_Rr_poss_mass = self.__get_pos_mass_states(Rr_father_L2H)
        L2H_Rs_poss_mass = self.__get_pos_mass_states(Rs_leaf_L2H)
        L2H_Rr_poss_mass = tf.reshape(L2H_Rr_poss_mass, [-1, 4 * self.sl])
        L2H_Rs_poss_mass = tf.reshape(L2H_Rs_poss_mass, [-1, 4 * self.sl])

        L2H_Rr_delta_pos = tf.gather_nd(
                self.state_global_delta_pos,
                Rr_father_L2H)
        L2H_Rr_delta_pos = tf.reshape(L2H_Rr_delta_pos, [-1, 3 * self.sl])
        L2H_Rs_delta_pos = tf.gather_nd(
                self.state_global_delta_pos,
                Rs_leaf_L2H)
        L2H_Rs_delta_pos = tf.reshape(L2H_Rs_delta_pos, [-1, 3 * self.sl])

        L2H_Rs = tf.concat(
                [L2H_Rs_poss_mass, L2H_Rs_delta_pos - L2H_Rr_delta_pos], 
                axis=-1)
        L2H_Rr = tf.concat(
                [L2H_Rr_poss_mass, \
                        tf.zeros_like(L2H_Rr_delta_pos, dtype=tf.float32)],
                axis=-1)
        return L2H_Rr, L2H_Rs

    def _get_WS_receiver_sender(self):
        # Create receiver index matrix
        # (BS, NR, 2)
        # This index is used later in tf.gather_nd function
        Rr = tf.tile(
                tf.reshape(
                    tf.tile(
                        tf.expand_dims(
                            tf.range(self.no), 
                            axis=1),
                        [1, self.nk],
                        ), 
                    [1, -1]
                    ), 
                [self.bs, 1]
                )
        Rr = add_batch_index(Rr)

        # Create sender index matrix
        # (BS, NR, 2)
        # This index is used later in tf.gather_nd function
        kNN_idx = tf.cast(self.kNN[:, 0, :, 1 : self.nk+1], tf.int32)
        Rs = tf.reshape(kNN_idx, [self.bs, self.nr])
        Rs = add_batch_index(Rs)

        kNN_mask_flat = tf.reshape(
                self.kNN_mask[:, 0, :, 1 : self.nk+1], 
                [self.bs, self.nr]
                )
        Rr = tf.boolean_mask(Rr, kNN_mask_flat)
        Rs = tf.boolean_mask(Rs, kNN_mask_flat)
        
        return Rr, Rs

    def _get_WS_states(self, Rr, Rs):
        state_Rr = self.__get_states(Rr, local_delta_pos=True)
        state_Rs = self.__get_states(Rs, local_delta_pos=True)
        return state_Rr, state_Rs

    def _get_H2L_receiver_sender(self):
        # Relation network computes effects of higher on lower particles
        depth_mask_indx_H2L = tf.where(self.depth_mask)
        Rs_father_H2L = tf.gather_nd(self.Rr_father, depth_mask_indx_H2L)
        Rr_leaf_H2L = tf.gather_nd(self.Rs_leaf, depth_mask_indx_H2L)
        return Rr_leaf_H2L, Rs_father_H2L

    def _get_H2L_states(self, Rr, Rs):
        # Here Rs is ancestor, Rr is children
        state_Rs, state_Rr = self._get_L2H_states(Rs, Rr)
        return state_Rr, state_Rs

    def build_L2H_or_WS_or_H2L(self, which_module, prev_effect):
        '''
        L2H - Lower to higher hierarchy propagation (L2A in the paper) 
        WG - Within group (WS in the paper)
        H2L - Higher hierarchy to lower propagation (A2D in the paper)
        '''
        if which_module=='L2H':
            func_get_receiver_sender = self._get_L2H_receiver_sender
            func_get_states = self._get_L2H_states
            which_one_hot = 0
        elif which_module=='WG':
            func_get_receiver_sender = self._get_WS_receiver_sender
            func_get_states = self._get_WS_states
            which_one_hot = 1
        elif which_module=='H2L':
            func_get_receiver_sender = self._get_H2L_receiver_sender
            func_get_states = self._get_H2L_states
            which_one_hot = 2

        Rr, Rs = func_get_receiver_sender()
        state_Rr, state_Rs = func_get_states(Rr, Rs)

        effect_Rs = tf.reshape(
                tf.gather_nd(prev_effect, Rs), 
                [-1, self.de])
        extra_attribute = tf.reshape(
                one_hot_column(effect_Rs, ndims=3, axis=which_one_hot), 
                [-1, 3])

        # Add stiffness to Ra if needed
        if self.vary_stiff==1:
            curr_stiff = tf.reshape(
                    tf.gather_nd(self.inp_stiff, Rs), 
                    [-1, self.sl])
            extra_attribute = tf.concat([extra_attribute, curr_stiff], axis=-1)

        # Add ground truth distance to Ra if needed
        if self.add_dist==1:
            # Get the new index tensor
            indx_dist = tf.concat([Rr, Rs[:, 1:]], axis=-1)
            curr_dist = tf.reshape(
                    tf.gather_nd(self.all_dist_ts, indx_dist), 
                    [-1, 1])
            extra_attribute = tf.concat([extra_attribute, curr_dist], axis=-1)

        mlp_input = tf.concat(
                [state_Rr, state_Rs, effect_Rs, extra_attribute], 
                axis=1)

        raw_effects = hidden_mlp(
                mlp_input, self.model_builder, 
                self.cfg, self.phiR_list[which_one_hot],
                reuse_weights=self.reuse_list[which_one_hot], 
                train=not self.my_test, debug=self.debug)

        summed_effects = sum_effects(
                raw_effects, Rr, 
                self.bs, self.no) # (BS, NO, DE)

        return summed_effects

    def get_effects(self):

        self._get_phiC()
        self._get_phiH()
        self._get_phiF()
        self._combine_phiC_phiH_phiF()

        self._prepare_for_HRN()
        E_L2H = self.build_L2H_or_WS_or_H2L(
                which_module='L2H', 
                prev_effect=self.E_all_X,
                )
        E_WG = self.build_L2H_or_WS_or_H2L(
                which_module='WG', 
                prev_effect=E_L2H,
                )
        E_H2L = self.build_L2H_or_WS_or_H2L(
                which_module='H2L', 
                prev_effect=E_L2H + E_WG,
                )

        # Sum all particles effects and compute next velocity
        E = E_L2H + E_WG + E_H2L
        return E

    def build_hrn_network(self):
        cfg = self.cfg
        self.de = cfg['phiR'][cfg['phiR_depth']]['num_features']
        E = self.get_effects()
        return E

    def get_predictions(self, E):
        # O stands for all states
        state_pos_mass = tf.concat(
                [self.state_pos, self.state_mass],
                axis=-1)
        state_pos_mass = tf.reshape(state_pos_mass, [self.bs, self.no, -1])
        state_delta_pos = tf.reshape(
                self.state_local_delta_pos,
                [self.bs, self.no, -1])
        O = tf.concat([state_pos_mass, state_delta_pos], axis=-1)

        C = tf.concat([O, self.state_gravity, E], axis=2)
        C = tf.reshape(
                C, 
                [self.bs * self.no, self.ds + self.de + self.dx])

        PhiO = hidden_mlp(
                C, self.model_builder, 
                self.cfg, 'phiO',
                reuse_weights=False, train=not self.my_test, 
                debug=self.debug)
        P = tf.reshape(PhiO, [self.bs, self.no, self.dp])
        return P

    def set_retval(self, pred_particle_velocity):
        inputs = self.inputs

        retval = {
                'pred_particle_velocity': pred_particle_velocity,
                'sparse_particles': inputs['sparse_particles'], # room center subtracted
                'OB1': self.OB1,
                'OB2': self.OB2,
                'my_test': self.my_test,
                'depth_list': self.depth_list,
                'max_depth': self.max_depth + 1,
                }

        if self.add_gloss>0:
            g_pred_v = tf.matmul(
                    self.mult_mat_space_ts, 
                    tf.matmul(
                        self.mult_mat_rev_ts, 
                        pred_particle_velocity
                        )
                    )
            g_gt_v = tf.matmul(
                    self.mult_mat_space_ts, 
                    tf.matmul(
                        self.mult_mat_rev_ts, 
                        self.sparse_particles[:, -1, :, 15:18],
                        )
                    )
            retval['g_pred_v'] = g_pred_v
            retval['g_gt_v'] = g_gt_v

        retval['kNN'] = self.kNN
        retval['sparse_particles'] = self.sparse_particles
        retval['kNN_mask'] = self.kNN_mask
        retval['mult_mat_ts'] = self.mult_mat_ts
        retval['mult_mat_rev_ts'] = self.mult_mat_rev_ts
        retval['leaf_flag'] = self.leaf_flag

        retval.update(
                flex_l2_particle_loss(
                    retval, 
                    alpha=self.alpha, 
                    preserve_distance_radius=self.preserve_distance_radius,
                    use_running_mean=self.use_running_mean,
                    separate_return=True,
                    debug=self.debug, sl=self.sl,
                    add_gloss=self.add_gloss,
                    avd_obj_mask=self.avd_obj_mask,
                    )
                )

        if self.my_test:
            if self.debug:
                print('------Recover velocity for test-----')

            for key, placeholder in self.placeholders.items():
                retval['%s_placeholder' % key] = placeholder
            retval['particles_placeholder'] = self.full_particles_place

            retval.update(self.org_input_ts)

            pred_particle_velocity = tf.matmul(
                    self.mult_mat_rev_ts, 
                    pred_particle_velocity)
            retval['pred_particle_velocity'] = pred_particle_velocity

        if self.debug:
            print('------NETWORK END-----')

        return retval

    def prepare_for_building(self):
        if self.debug:
            print('------NETWORK START-----')

        self._get_group_and_which_dataset()
        self._post_process_data()
        # Define placeholders for test script that feeds in its own predictions
        if self.my_test:
            self._set_test_placeholders()
        self._set_attributes_from_inputs()

        self._set_bs_ts_no()
        self._apply_group()
        self._set_bs_ts_no()

        self._get_all_states()
        self._set_short_names_for_dims()
        self._set_gravity()


def hrn(**kwargs):
    hrn_model = HRNModel(**kwargs)

    hrn_model.prepare_for_building()
    E = hrn_model.build_hrn_network()
    pred_particle_velocity = hrn_model.get_predictions(E)
    retval = hrn_model.set_retval(pred_particle_velocity)

    return retval, hrn_model.model_builder.params
