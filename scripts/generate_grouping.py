import os, sys
import numpy as np

import tensorflow as tf
import cPickle

import json
import copy
import argparse
import pdb

import sklearn.cluster as skcluster

def load_tfrs(args, type_dict={'kNN':np.int16,'full_particles':np.float32}):

    main_key = 'full_particles'
    ret_res = []

    fp_dir = os.path.join(args.datapath, main_key)
    fp_meta_path = os.path.join(fp_dir, 'meta.pkl')
    fp_meta = cPickle.load(open(fp_meta_path, 'r'))

    all_tfrs_path = os.listdir(fp_dir)
    all_tfrs_path = filter(lambda x:'tfrecords' in x, all_tfrs_path)
    all_tfrs_path.sort()
    all_tfrs_path = [\
            os.path.join(fp_dir, each_tfr) \
            for each_tfr in all_tfrs_path]

    for tfr_path in all_tfrs_path:
        record_iterator = tf.python_io.tf_record_iterator(path=tfr_path)
        get_num = 0

        for string_record in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(string_record)
            fp_string = (example.features.feature[main_key]
                                          .bytes_list
                                          .value[0])
            fp_array = np.fromstring(fp_string, dtype=type_dict[main_key])
            fp_array = fp_array.reshape(fp_meta[main_key]['rawshape'])
            get_num = get_num+1
            if get_num >= args.skip_frame:
                if args.num_particle_filter:
                    now_num = np.sum(fp_array[:, 14] > 0)
                    if now_num != args.num_particle_filter:
                        continue
                ret_res.append(fp_array)
                break
        if len(ret_res) >= 1:
            break

    return ret_res[0]

def get_parser():
    parser = argparse.ArgumentParser(
            description='The script to get the grouping index')
    parser.add_argument(
            '--datapath', 
            default='/data2/mrowca/datasets/18_world_dataset/new_tfdata', 
            type=str, 
            action='store', help='Load directory of the flex dataset')
    parser.add_argument(
            '--num_per_level', default=10, type=int, 
            action='store', help='Number of particles for each level')
    parser.add_argument(
            '--savepath', 
            default='/data2/chengxuz/Dataset/18_world_dataset/group_result.pkl', 
            type=str, action='store', help='Save path of the grouping results')
    parser.add_argument(
            '--cluster_alg', default=None, type=str,
            action='store', required=True,
            help='Algorithm for clustering')
    parser.add_argument(
            '--cluster_kwargs', default=None, type=str,
            action='store', help='Kwargs for clustering algorithm')
    parser.add_argument(
            '--dyn_div', default=0, type=int, action='store', 
            help='Whether dynamically dividing particles')
    parser.add_argument(
            '--add_sparse', default=0, type=int, action='store', 
            help='Whether adding sparse representations for the matrixes')
    parser.add_argument(
            '--add_axis', default=0, type=int, action='store', 
            help='Whether adding group longest axis, default is 0 (no)')
    parser.add_argument(
            '--rand_seed', default=None, type=int, action='store', 
            help='Whether fixing rand seed, default is None!')
    parser.add_argument(
            '--add_dist', default=0, type=int, action='store', 
            help='Whether adding ground truth distance, default is 0 (no)')
    parser.add_argument(
            '--remove_top', default=0, type=int, action='store', 
            help='Whether removing the top, default is 0 (No)!')
    parser.add_argument(
            '--add_local_mat', default=0, type=int, action='store', 
            help='Whether adding local motion matrix, default is 0 (No)!')
    parser.add_argument(
            '--mult_obj_special', default=0, type=int, action='store', 
            help='Adding multiple object special support, default is 0 (No)!')
    parser.add_argument(
            '--reverse_division', default=0, type=int, action='store', 
            help='When adding multiple object special support, '\
                    + 'setting this to 1 will reverse the division')
    parser.add_argument(
            '--ignore_obj_ids', default=0, type=int, action='store', 
            help='Whether ignoring object ids saved, if yes, '\
                    + 'all particles are treated as in the same object')
    parser.add_argument(
            '--skip_frame', default=11, type=int, action='store', 
            help='Skipping first several frames for first file')
    parser.add_argument(
            '--num_particle_filter', default=None, type=int, action='store', 
            help='Only requiring this number of particles')

    return parser

def get_longest_axis(full_pos, node_list=None):
    max_dist = 0
    num_p = len(full_pos)
    if node_list is None:
        node_list = range(num_p)
    for pn_indx, p_indx in enumerate(node_list):
        p_pos = full_pos[p_indx]
        for p2_indx in node_list[pn_indx+1:]:
            p2_pos = full_pos[p2_indx]

            curr_dist = np.linalg.norm(p2_pos-p_pos)
            if curr_dist > max_dist:
                max_dist = curr_dist

    return max_dist

def get_axis_for_each_group(full_pos, father_list):
    num_prt = len(father_list)
    all_axis = np.zeros(num_prt)

    for indx in xrange(num_prt):
        son_nodes = [\
                each_n for each_n in xrange(indx) \
                if father_list[each_n]==indx]
        if len(son_nodes)>0:
            curr_axis = get_longest_axis(full_pos, son_nodes)
            all_axis[indx] = curr_axis
    return all_axis

def get_dist_mat(full_pos):
    num_prt = len(full_pos)

    dist_mat = np.zeros([num_prt, num_prt], dtype=np.float32)

    for indx_0 in xrange(num_prt):
        for indx_1 in xrange(num_prt):
            _curr_dist = np.linalg.norm(full_pos[indx_0] - full_pos[indx_1])
            dist_mat[indx_0, indx_1] = _curr_dist

    return dist_mat

def get_hier_mask(father_list):

    def get_particle_depth(start_idx):
        if father_list[start_idx] is None:
            return 0
        else:
            curr_father = father_list[start_idx]
            return 1 + get_particle_depth(curr_father)

    depth_list = [\
            get_particle_depth(each_p) \
            for each_p in xrange(len(father_list))]
    depth_list = np.asarray(depth_list)

    return depth_list

def try_group_alg(
        args,
        full_pos,
        node_list,
        ):
    '''
    The function to use specified algorithm to group particles
    '''
    new_nodes = []
    # Build the cluster
    cluster_func = getattr(skcluster, args.cluster_alg)
    cluster_kwargs = json.loads(args.cluster_kwargs)
    if 'n_clusters' in cluster_kwargs:
        cluster_kwargs['n_clusters'] = args.num_per_level
    # Dynamically choose the number of clusters
    if args.dyn_div==1:
        curr_div = min(
                int(len(node_list)/args.num_per_level), 
                args.num_per_level)
        if curr_div==1:
            return [[each_node] for each_node in node_list]
        if 'n_clusters' in cluster_kwargs:
            cluster_kwargs['n_clusters'] = curr_div
    # Actually running the clustering algorithm
    cluster = cluster_func(**cluster_kwargs)
    pos_arr = np.asarray([full_pos[each_node] for each_node in node_list])
    cluster_res = cluster.fit_predict(pos_arr)
    # Put each group together
    largest_group = np.max(cluster_res) + 1
    for group_indx in xrange(largest_group):
        new_nodes.append(
                [node_list[curr_indx] \
                        for curr_indx,curr_i in enumerate(cluster_res) \
                        if curr_i==group_indx])

    # Test codes: Count the number of nodes in each group
    #len_nodes = [len(each_group) for each_group in new_nodes]
    #if 1 in len_nodes:
    #    print(len_nodes, len(node_list))
    return new_nodes

def add_one_node(
        son_nodes,
        pos_list, 
        kNN_list,
        new_node_list,
        groupflag_list,
        father_list,
        ):
    # create a new particle
    new_indx = len(pos_list)

    ## Add new position
    new_pos = pos_list[son_nodes[0]]
    for pos_indx in xrange(1,len(son_nodes)):
        new_pos = new_pos + pos_list[son_nodes[pos_indx]]
    new_pos = new_pos/len(son_nodes)
    pos_list.append(new_pos)

    ## Add new kNN (empty for new node)
    ## and change the old kNN to be connected with each other
    kNN_list.append([new_indx])
    for son_indx in son_nodes:
        assert len(kNN_list[son_indx])==1, "Must be empty lists for sons"
        kNN_list[son_indx] = sorted(son_nodes, 
                key=lambda i: np.linalg.norm(pos_list[i]-pos_list[son_indx]))
        #for each_node in son_nodes:
        #    if each_node>son_indx:
        #        kNN_list[son_indx].append(each_node)
        #for each_node in son_nodes:
        #    if each_node<son_indx:
        #        kNN_list[son_indx].append(each_node)

    ## Add new_node_flag
    new_node_list.append(1)

    ## Add new group flag and change old group flag
    groupflag_list.append(0)
    for son_indx in son_nodes:
        assert groupflag_list[son_indx]==0, "Must be ungrouped for sons"
        groupflag_list[son_indx] = 1

    ## Add new member to father list and change son's father flag
    father_list.append(None)
    for son_indx in son_nodes:
        assert father_list[son_indx] is None, \
                "Father flag must be None for sons!"
        father_list[son_indx] = new_indx

    return new_indx

def group_particles(
        args,
        node_to_group,
        pos_list, 
        kNN_list,
        new_node_list,
        groupflag_list,
        father_list,
        remove_top=0,
        ):

    ret_res = [pos_list,
            kNN_list,
            new_node_list,
            groupflag_list,
            father_list]

    num_to_group = len(node_to_group)
    if num_to_group==1:
        return node_to_group+ret_res
    if num_to_group<=args.num_per_level:
        new_indx = add_one_node(node_to_group, *ret_res)
        return [new_indx]+ret_res
    
    # Do hard work, divide and conquer
    if args.cluster_alg is None:
        raise NotImplementedError, "Cluster alg needs to be specified"
    else:
        new_nodes = try_group_alg(args,pos_list,node_to_group)

    ## Add each new node and add the overall new node
    ## if there is only one group returned, forcing all particles there to be one group
    new_group = []
    for each_son_nodes in new_nodes:
        new_node, pos_list, \
                kNN_list, new_node_list, \
                groupflag_list, father_list = group_particles(
                        args,
                        each_son_nodes,
                        pos_list, 
                        kNN_list,
                        new_node_list,
                        groupflag_list,
                        father_list,
                        )
        new_group.append(new_node)
    if remove_top==0:
        new_indx = add_one_node(new_group, *ret_res)
    else:
        # Not adding new node
        new_indx = None
        for son_indx in new_group:
            assert len(kNN_list[son_indx])==1, "Must be empty lists for sons"
            kNN_list[son_indx] = sorted(
                    new_group, 
                    key=lambda i: np.linalg.norm(
                        pos_list[i] - pos_list[son_indx]))
    
    return [new_indx] + ret_res

def get_all_fathers(start_idx, father_list, curr_list=[]):
    if father_list[start_idx] is None:
        return curr_list
    else:
        curr_father = father_list[start_idx]
        return [curr_father] \
                + get_all_fathers(curr_father, father_list, curr_list)

def get_sparse(dense_mat):
    nonzero_res = np.nonzero(dense_mat)
    nonzero_val = dense_mat[nonzero_res]
    nonzero_res = np.transpose(nonzero_res)
    sparse_rep = np.concatenate(
            [nonzero_res, nonzero_val[:, np.newaxis]], 
            axis=1)

    return sparse_rep

def get_save_dict(args, full_particles):
    # Filter out the invalid particles
    all_ids = full_particles[:, 14]
    valid_mask = all_ids > 0
    full_particles = full_particles[valid_mask, :]

    num_nodes = full_particles.shape[0]
    full_pos = full_particles[:, :3]
    all_ids = full_particles[:, 14]

    if args.ignore_obj_ids==1:
        all_ids = np.ones(all_ids.shape)

    pos_list = [full_pos[i] for i in xrange(num_nodes)]
    kNN_list = [[i] for i in xrange(num_nodes)]
    new_node_list = [0 for i in xrange(num_nodes)]
    groupflag_list = [0 for i in xrange(num_nodes)]
    father_list = [None for i in xrange(num_nodes)]

    num_all_objects = len(np.unique(all_ids))
    all_top_nodes = []
    super_node = None
    if args.mult_obj_special==1:
        assert args.remove_top==0, \
                "Cannot use remove_top when mult_obj_special==1"
    for each_id in np.unique(all_ids):
        curr_prtcls = np.where(all_ids==each_id)[0]
        new_indx, pos_list, \
                kNN_list, new_node_list, \
                groupflag_list, father_list = group_particles(
                        args,
                        curr_prtcls,
                        pos_list, 
                        kNN_list,
                        new_node_list,
                        groupflag_list,
                        father_list,
                        args.remove_top,
                        )
        all_top_nodes.append(new_indx)

    # For multiple object support, add super node
    if args.mult_obj_special==1:
        assert num_all_objects>1, "Must have multiple objects"
        super_node = add_one_node(
            all_top_nodes,
            pos_list, 
            kNN_list,
            new_node_list,
            groupflag_list,
            father_list,
            )

    #for kNN_idx,each_kNN in enumerate(kNN_list):
    #    print(kNN_idx, each_kNN, father_list[kNN_idx], pos_list[kNN_idx])
    #print(len(pos_list))
    #print(father_list)

    #print(get_longest_axis(full_pos))

    # make the matrix for computing the average speeds and positions,
    # as well as the number of particles represented
    num_all_nodes = len(pos_list)
    num_p_rep = np.zeros([num_all_nodes])
    mult_mat = np.zeros([num_all_nodes, num_nodes])
    mult_mat_space = np.zeros([num_all_nodes, num_nodes])
    for idx in xrange(num_all_nodes):
        if new_node_list[idx]==0:
            mult_mat[idx,idx] = 1.0
            mult_mat_space[idx,idx] = 1.0
            num_p_rep[idx] = 1
        else:
            son_nodes = [\
                    x \
                    for x in xrange(num_all_nodes) \
                    if father_list[x]==idx]
            for each_son in son_nodes:
                mult_mat[idx] += mult_mat[each_son]
                mult_mat_space[idx] += mult_mat_space[each_son]
                num_p_rep[idx] += num_p_rep[each_son]
            mult_mat[idx] /= len(son_nodes)
            mult_mat_space[idx] /= len(son_nodes)
            for each_son in son_nodes:
                mult_mat[each_son] -= mult_mat[idx]

    mult_mat_rev = np.zeros([num_nodes, num_all_nodes])
    for idx in xrange(num_nodes):
        mult_mat_rev[idx, idx] = 1
        curr_fathers = get_all_fathers(idx, father_list)
        for each_father in curr_fathers:
            mult_mat_rev[idx, each_father] = 1

    # Add gravity tensor
    grav_flag = np.zeros([num_all_nodes])
    # For every node without father, set grav_flag to be 1
    for idx in xrange(num_all_nodes):
        if father_list[idx] is None:
            grav_flag[idx] = 1

    # Modify kNN_list to make it full, also make the valid flag
    max_kNN_len = np.max([len(each_kNN) for each_kNN in kNN_list])
    kNN_valid_flag = []
    for idx in xrange(num_all_nodes):
        curr_kNN_valid_flag = []
        for temp in xrange(len(kNN_list[idx])):
            curr_kNN_valid_flag.append(1)
        for temp in xrange(len(kNN_list[idx]),max_kNN_len):
            kNN_list[idx].append(kNN_list[idx][0])
            curr_kNN_valid_flag.append(0)
        kNN_valid_flag.append(curr_kNN_valid_flag)

    depth_list = get_hier_mask(father_list)
    if __name__=='__main__':
        print(np.max(depth_list))
    save_dict = {
            'kNN_list':kNN_list,
            'new_node_list':new_node_list,
            'groupflag_list':groupflag_list,
            'father_list':father_list,
            'kNN_valid_flag':kNN_valid_flag,
            'num_p_rep':num_p_rep,
            'mult_mat':mult_mat,
            'mult_mat_space':mult_mat_space,
            'grav_flag':grav_flag,
            'max_kNN_len':max_kNN_len,
            'mult_mat_rev':mult_mat_rev,
            'pos_list':pos_list,
            'depth_list':depth_list,
            }

    if args.add_sparse==1:
        mult_mat_fact = np.zeros([num_all_nodes, num_all_nodes])
        for idx in xrange(num_all_nodes):
            mult_mat_fact[idx, idx] = 1
            direct_father = father_list[idx]
            if direct_father is not None:
                mult_mat_fact[idx, direct_father] = -1
        assert np.allclose(
                np.matmul(mult_mat_fact, mult_mat_space), 
                mult_mat), "Factorization is not successful"

        sparse_mult_mat_rev = get_sparse(mult_mat_rev)
        sparse_mult_mat_space = get_sparse(mult_mat_space)
        #sparse_mult_mat = get_sparse(mult_mat)
        sparse_mult_mat_fact = get_sparse(mult_mat_fact)

        save_dict['sparse_mult_mat_rev'] = sparse_mult_mat_rev
        save_dict['sparse_mult_mat_space'] = sparse_mult_mat_space
        save_dict['sparse_mult_mat_fact'] = sparse_mult_mat_fact

    if args.add_axis==1:
        all_axis = get_axis_for_each_group(pos_list, father_list)
        save_dict['all_axis'] = all_axis

    if args.add_dist==1:
        all_dist = get_dist_mat(pos_list)
        save_dict['all_dist'] = all_dist

    if args.add_local_mat==1:
        mult_mat_local = np.zeros([num_all_nodes, num_all_nodes])
        for idx in xrange(num_all_nodes):
            mult_mat_local[idx, idx] = 1
            if not father_list[idx] is None:
                mult_mat_local[idx, father_list[idx]] = -1
        save_dict['mult_mat_local'] = mult_mat_local

    if args.mult_obj_special==1:
        num_all_p = len(pos_list)
        # Set the initial value
        L2H_attribute = np.zeros([num_all_p, 2], dtype=np.float32)
        WG_attribute = np.zeros([num_all_p, 2], dtype=np.float32)
        H2L_attribute = np.zeros([num_all_p, 2], dtype=np.float32)
        L2H_division = np.ones([num_all_p], dtype=np.float32)
        WG_division = np.ones([num_all_p], dtype=np.float32)

        for o_idx in xrange(num_all_p):
            if o_idx!=super_node:
                L2H_attribute[o_idx, 0] = 1
            else:
                L2H_attribute[o_idx, 1] = 1
                if args.reverse_division==0:
                    L2H_division[o_idx] = num_all_objects
                else:
                    L2H_division[o_idx] = 1.0/num_all_objects
            if o_idx in all_top_nodes:
                WG_attribute[o_idx, 1] = 1
                if args.reverse_division==0:
                    WG_division[o_idx] = num_all_objects - 1
                else:
                    WG_division[o_idx] = 1.0/(num_all_objects - 1)

                H2L_attribute[o_idx, 1] = 1
            else:
                WG_attribute[o_idx, 0] = 1
                H2L_attribute[o_idx, 0] = 1
        save_dict['L2H_attribute'] = L2H_attribute
        save_dict['WG_attribute'] = WG_attribute
        save_dict['H2L_attribute'] = H2L_attribute
        save_dict['L2H_division'] = L2H_division
        save_dict['WG_division'] = WG_division

    return save_dict

def main():
    parser = get_parser()
    args = parser.parse_args()

    # About random seed, might be used for other grouping alg
    if args.rand_seed is not None:
        np.random.seed(args.rand_seed)

    # Get full particles from tfrecrods
    full_particles = load_tfrs(args)

    # Get the grouping results from full_particles
    save_dict = get_save_dict(args, full_particles)

    # Create the folder for saving if not existed
    dir_name = os.path.dirname(args.savepath)
    os.system('mkdir -p %s' % dir_name)
    cPickle.dump(save_dict, open(args.savepath, 'w'))

if __name__=='__main__':
    main()
