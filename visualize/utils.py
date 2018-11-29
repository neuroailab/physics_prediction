#%matplotlib notebook
import pymongo as pm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import juggle_axes
import matplotlib
import matplotlib.pyplot as plt
import gridfs
import cPickle
from PIL import Image
import scipy.signal as signal
import os
import tensorflow as tf
import matplotlib.animation as animation
from IPython.display import HTML
import itertools
from matplotlib import rcParams
import pdb
from matplotlib.backends.backend_pdf import PdfPages

import tfutils

def scale_data(x, y, z, maximum=None, minimum=None):
    if maximum is None or minimum is None:
        return x, y, z
    x = x * (maximum[0] - minimum[0]) + minimum[0]
    y = y * (maximum[1] - minimum[1]) + minimum[1]
    z = z * (maximum[2] - minimum[2]) + minimum[2]
    return x, y, z

def unnormalize_grid(grid, stats):
    unnormalized_grid = grid
    dim = unnormalized_grid.shape[-1]
    unnormalized_grid[:,:,:,0:7] = unnormalized_grid[:,:,:,0:7] * \
            (stats['full_particles']['max'][0,0:7] - \
            stats['full_particles']['min'][0,0:7]) + \
            stats['full_particles']['min'][0,0:7]
    if dim > 7:
        unnormalized_grid[:,:,:,7:10] = unnormalized_grid[:,:,:,7:10] * \
                (stats['actions']['max'][0,0:3] - \
                stats['actions']['min'][0,0:3]) + \
                stats['actions']['min'][0,0:3]
    if dim > 10:
        unnormalized_grid[:,:,:,10:13] = unnormalized_grid[:,:,:,10:13] * \
                (stats['actions']['max'][0,3:6] - \
                stats['actions']['min'][0,3:6]) + \
                stats['actions']['min'][0,3:6]
    if dim > 15:
        unnormalized_grid[:,:,:,15:18] = unnormalized_grid[:,:,:,15:18] * \
                (stats['full_particles']['max'][0,15:18] - \
                stats['full_particles']['min'][0,15:18]) + \
                stats['full_particles']['min'][0,15:18]
    return unnormalized_grid

def create_figures(all_particles, min_all, max_all, name='Particles', index=14, coloring='id', 
        angle=None, elev=None, s=None, num_col=2, pdf_path=None, gap_frame=1):
    num_steps = 100
    delta = np.max(max_all - min_all)
    max_all = min_all + delta
    xi = np.linspace(min_all[0], max_all[0], num_steps)
    yi = np.linspace(min_all[1], max_all[1], num_steps)
    zi = np.linspace(min_all[2], max_all[2], num_steps)
    def get_plot_data(particles, index=index, coloring = coloring):
        x = []; y = []; z = []; c = [];
        if coloring is 'colormap':
            if isinstance(index, list):
                assert len(index) == 2, 'provide start and end of your slice only!'
                index = slice(index[0], index[1])  
            x = np.digitize(particles[:,0], xi)
            y = np.digitize(particles[:,1], yi)
            z = np.digitize(particles[:,2], zi)
            data = [x, y, z]
            val = particles[:,index]
            if len(val.shape) > 1:
                val = val * val
                val = np.sum(val, axis=1)
                val = np.sqrt(val)
                assert len(val.shape) == 1
            try:
                maximum = np.max(val)
            except ValueError:
                maximum = 1
            try:
                minimum = np.min(val)
            except ValueError:
                minimum = 0
            if maximum == minimum:
                maximum = 1
                minimum = 0
            val = (val - minimum) / (maximum - minimum)
            cmap = matplotlib.cm.get_cmap('bwr')
            c = cmap(val)
            c = np.array(c)
            return data[0], data[1], data[2], c, minimum, maximum
    
    # First set up the figure, the axis, and the plot element we want to animate
    n_frame = len(all_particles[0])/gap_frame
    base_height = 2
    fig = plt.figure(figsize=(base_height*n_frame, base_height*num_col))
    fig.subplots_adjust(wspace=0.0, hspace=0.0)
    for curr_suboffset, particles in zip([0, n_frame], all_particles):
        for curr_frame in xrange(n_frame):
            axes = fig.add_subplot(*[num_col,n_frame,curr_frame+1+curr_suboffset], projection='3d')
            axes.get_xaxis().set_visible(False)
            axes.get_yaxis().set_visible(False)
            #axes.set_xlabel("x")
            #axes.set_ylabel("z")
            #axes.set_zlabel("y")
            # first frame
            x, y, z, c, minimum, maximum = get_plot_data(particles[0])
            x = [0, num_steps]
            y = [0, num_steps]
            z = [0, num_steps]
            coord = np.array(list(itertools.product(*[x,y,z])))
            x = coord[:,0]
            y = coord[:,1]
            z = coord[:,2]
            if s is not None:
                im = axes.scatter(x, z, y, s=s)
            else:
                im = axes.scatter(x, z, y)
            im._edgecolor3d = c
            im._facecolor3d = c
            #plt.title('%s' % name)
            ax = plt.gca()
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.zaxis.set_ticklabels([])
            ax.view_init(azim=angle, elev=elev)
            #plt.close()

            # initialization function: plot the background of each frame
            def init():
                x, y, z, c, minimum, maximum = get_plot_data(particles[0])
                im._offsets3d = (x, z, y)
                im._edgecolor3d = c
                im._facecolor3d = c
                return [im]

            # animation function.  This is called sequentially
            def animate(i):
                x, y, z, c, minimum, maximum = get_plot_data(particles[i%len(particles)])
                im._offsets3d = (x, z, y)
                im._edgecolor3d = c
                im._facecolor3d = c
                return [im]

            #init()
            animate(curr_frame*gap_frame)

    if pdf_path is not None:
        if pdf_path.endswith('pdf'):
            pp = PdfPages(pdf_path)
            plt.savefig(pp, format='pdf', bbox_inches='tight')
            pp.close()
        else:
            plt.savefig(pdf_path, bbox_inches='tight')
    plt.show()

def create_video(particles, name='Particles', index=14, coloring='id', angle=None, elev=None, s=None):
    num_steps = 100
    min_all = np.min(particles[:,:,0:3], axis=(0,1))
    max_all = np.max(particles[:,:,0:3], axis=(0,1))
    delta = np.max(max_all - min_all)
    max_all = min_all + delta
    xi = np.linspace(min_all[0], max_all[0], num_steps)
    yi = np.linspace(min_all[1], max_all[1], num_steps)
    zi = np.linspace(min_all[2], max_all[2], num_steps)
    def get_plot_data(particles, index=index, coloring = coloring):
        x = []; y = []; z = []; c = [];
        if coloring is 'colormap':
            if isinstance(index, list):
                assert len(index) == 2, 'provide start and end of your slice only!'
                index = slice(index[0], index[1])  
            x = np.digitize(particles[:,0], xi)
            y = np.digitize(particles[:,1], yi)
            z = np.digitize(particles[:,2], zi)
            data = [x, y, z]
            val = particles[:,index]
            if len(val.shape) > 1:
                val = val * val
                val = np.sum(val, axis=1)
                val = np.sqrt(val)
                assert len(val.shape) == 1
            try:
                maximum = np.max(val)
            except ValueError:
                maximum = 1
            try:
                minimum = np.min(val)
            except ValueError:
                minimum = 0
            if maximum == minimum:
                maximum = 1
                minimum = 0
            val = (val - minimum) / (maximum - minimum)
            cmap = matplotlib.cm.get_cmap('bwr')
            c = cmap(val)
            c = np.array(c)
            return data[0], data[1], data[2], c, minimum, maximum
    
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure(figsize=(4, 4))
    axes = fig.add_subplot(111, projection='3d')
    axes.get_xaxis().set_visible(False)
    axes.get_yaxis().set_visible(False)
    axes.set_xlabel("x")
    axes.set_ylabel("z")
    axes.set_zlabel("y")
    # first frame
    x, y, z, c, minimum, maximum = get_plot_data(particles[0])
    x = [0, num_steps]
    y = [0, num_steps]
    z = [0, num_steps]
    coord = np.array(list(itertools.product(*[x,y,z])))
    x = coord[:,0]
    y = coord[:,1]
    z = coord[:,2]
    if s is not None:
        im = axes.scatter(x, z, y, s=s)
    else:
        im = axes.scatter(x, z, y)
    im._edgecolor3d = c
    im._facecolor3d = c
    plt.title('%s' % name)
    ax = plt.gca()
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    ax.view_init(azim=angle, elev=elev)
    plt.close()

    # initialization function: plot the background of each frame
    def init():
        x, y, z, c, minimum, maximum = get_plot_data(particles[0])
        im._offsets3d = (x, z, y)
        im._edgecolor3d = c
        im._facecolor3d = c
        return [im]

    # animation function.  This is called sequentially
    def animate(i):
        x, y, z, c, minimum, maximum = get_plot_data(particles[i%len(particles)])
        im._offsets3d = (x, z, y)
        im._edgecolor3d = c
        im._facecolor3d = c
        return [im]

    anim = animation.FuncAnimation(fig, animate,
            frames=len(particles), interval=200)
    return anim.to_html5_video()

def correct_videos(videos):
    videos = videos.replace('loop', 'loop style="display:block; margin: 0 auto;"')
    return videos

def vis_true_and_pred(path_true, path_pred, ex=3, 
        static_path=None, center=None, angle=None, elev=None, s=None, unroll_length=None, 
        just_fig=False, pdf_path=None, gap_frame=1):
    with open(path_pred) as f: #results_mparabola16_8.pkl
	pdata = cPickle.load(f)
    with open(path_true) as f: #true_results_mparabola16_8.pkl
	tdata = cPickle.load(f)
    if static_path is not None:
        with open(static_path) as f:
            static_particles = cPickle.load(f)[0]
            # static_particles = static_particles[::4]
            if center is not None:
                static_particles[:,0:3] -= center

    predicted_data = pdata[ex]
    true_data = tdata[ex]
    if unroll_length is not None and \
        unroll_length < max(len(true_data['predicted_particles']), len(predicted_data['predicted_particles'])):
            true_data['predicted_particles'] = true_data['predicted_particles'][:unroll_length]
            predicted_data['predicted_particles'] = predicted_data['predicted_particles'][:unroll_length]
    print(len(true_data['predicted_particles']))
    print(len(predicted_data['predicted_particles']))

    particles = true_data['predicted_particles']
    if static_path is not None:
        min_all = np.min(particles[:,:,0:3], axis=(0,1))
        max_all = np.max(particles[:,:,0:3], axis=(0,1))
        dist = np.linalg.norm(min_all-max_all)
        min_all -= 0.25 * dist
        max_all += 0.25 * dist
        idx = np.logical_and.reduce([static_particles[:,0]>= min_all[0], static_particles[:,0]<=max_all[0], static_particles[:,2]>= min_all[2], static_particles[:,2]<=max_all[2]])
        temp_particles = static_particles[idx]
        temp_particles = np.tile(np.expand_dims(np.concatenate([temp_particles, np.zeros((temp_particles.shape[0], particles.shape[2]-temp_particles.shape[1]))], axis=1), 0), [particles.shape[0], 1, 1])
        temp_particles[:,:,14] = 2
        particles = np.concatenate([temp_particles, particles], axis=1)
    if not just_fig:
        video = create_video(particles, coloring='colormap', index=[14,15], angle=angle, elev=elev, s=s)
        video_0 = correct_videos(video)
    else:
        gt_particles = particles
    particles = predicted_data['predicted_particles']
    if static_path is not None:
        min_all = np.min(particles[:,:,0:3], axis=(0,1))
        max_all = np.max(particles[:,:,0:3], axis=(0,1))
        dist = np.linalg.norm(min_all-max_all)
        min_all -= 0.25 * dist
        max_all += 0.25 * dist
        idx = np.logical_and.reduce([static_particles[:,0]>= min_all[0], static_particles[:,0]<=max_all[0], static_particles[:,2]>= min_all[2], static_particles[:,2]<=max_all[2]])
        temp_particles = static_particles[idx]
        temp_particles = np.tile(np.expand_dims(np.concatenate([temp_particles, np.zeros((temp_particles.shape[0], particles.shape[2]-temp_particles.shape[1]))], axis=1), 0), [particles.shape[0], 1, 1])
        temp_particles[:,:,14] = 2
        particles = np.concatenate([temp_particles, particles], axis=1)
    if not just_fig:
        video = create_video(particles, coloring='colormap', index=[14,15], angle=angle, elev=elev, s=s)
        video_1 = correct_videos(video)

        return video_0, video_1
    else:
        min_all = np.min(gt_particles[:,:,0:3], axis=(0,1))
        max_all = np.max(gt_particles[:,:,0:3], axis=(0,1))
        min_all = np.minimum(np.min(particles[:,:,0:3], axis=(0,1)), min_all)
        max_all = np.maximum(np.max(gt_particles[:,:,0:3], axis=(0,1)), max_all)
        create_figures([gt_particles, particles], coloring='colormap', index=[14,15], angle=angle, elev=elev, s=s, min_all=min_all, max_all=max_all, pdf_path=pdf_path, gap_frame=gap_frame)
        #create_figures(particles, coloring='colormap', index=[14,15], angle=angle, elev=elev, s=s, sub_offset=len(particles), min_all=min_all, max_all=max_all, fig=fig)
