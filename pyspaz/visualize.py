'''
visualize.py -- visualization functions for localizations
and trajectories

'''
# Numerical stuff
import numpy as np 

# Reading / writing trajectories
from scipy import io as sio 

# For hard-writing a 24-bit RGB in overlay_trajs
import tifffile

# Dataframes
import pandas as pd 

# I/O
import os
import sys
from . import spazio 
from . import localize

# Plotting
import matplotlib.pyplot as plt 
from matplotlib import cm
import seaborn as sns
sns.set(style = 'ticks')

# Progress bar
from tqdm import tqdm 

# Interactive functions for Jupyter notebooks
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual

def wrapup(out_png, dpi = 400, open_result = True):
    ''' Save a plot to PNG '''
    plt.tight_layout()
    plt.savefig(out_png, dpi = dpi)
    plt.close()
    if open_result:
        os.system('open %s' % out_png)

def loc_density(
    locs,
    metadata = {},
    ax = None,
    upsampling_factor = 20,
    kernel_width = 0.5,
    verbose = False,
    y_col = 'y_pixels',
    x_col = 'x_pixels',
    convert_to_um = True,
    vmax_mod = 1.0,
    cmap = 'gray',
    out_png = 'default_loc_density_out.png',
    save_plot = True,
): 
    # Get the list of localized positions
    m_keys = list(metadata.keys())
    positions = np.asarray(locs[[y_col, x_col]])
    if convert_to_um and ('pixel_size_um' in m_keys):
        positions = positions * metadata['pixel_size_um']

    # Make the size of the out frame
    if ('N' in m_keys) and ('M' in m_keys):
        if convert_to_um:
            n_up = int(metadata['N'] * metadata['pixel_size_um']) * upsampling_factor
            m_up = int(metadata['M'] * metadata['pixel_size_um']) * upsampling_factor
        else:
            n_up = int(metadata['N']) * upsampling_factor
            m_up = int(metadata['M']) * upsampling_factor
    else:
        if convert_to_um:
            n_up = int(positions[:,0].max() * metadata['pixel_size_um']) * upsampling_factor
            m_up = int(positions[:,1].max() * metadata['pixel_size_um']) * upsampling_factor
        else:
            n_up = int(positions[:,0].max()) * upsampling_factor
            m_up = int(positions[:,1].max()) * upsampling_factor 

    density = np.zeros((n_up, m_up), dtype = 'float64')

    # Determine the size of the Gaussian kernel to use for
    # KDE
    sigma = kernel_width * upsampling_factor
    w = int(6 * sigma)
    if w % 2 == 0: w+=1
    half_w = w // 2
    r2 = sigma ** 2
    kernel_y, kernel_x = np.mgrid[:w, :w]
    kernel = np.exp(-((kernel_x-half_w)**2 + (kernel_y-half_w)**2) / (2*r2))

    n_locs = len(locs)
    for loc_idx in range(n_locs):
        y = int(round(positions[loc_idx, 0] * upsampling_factor, 0))
        x = int(round(positions[loc_idx, 1] * upsampling_factor, 0))
        try:
            # Localization is entirely inside the borders
            density[
                y-half_w : y+half_w+1,
                x-half_w : x+half_w+1,
            ] += kernel
        except ValueError:
            # Localization is close to the edge
            k_y, k_x = np.mgrid[y-half_w:y+half_w+1, x-half_w:x+half_w+1]
            in_y, in_x = ((k_y>=0) & (k_x>=0) & (k_y<n_up) & (k_x<m_up)).nonzero()
            density[k_y.flatten()[in_y], k_x.flatten()[in_x]] = \
                density[k_y.flatten()[in_y], k_x.flatten()[in_x]] + kernel[in_y, in_x]

        if verbose:
            sys.stdout.write('Finished compiling the densities of %d/%d localizations...\r' % (loc_idx+1, n_locs))
            sys.stdout.flush()
    if (save_plot == True) and (ax == None):
        fig, ax = plt.subplots(figsize = (4, 4))
        ax.imshow(
            density[::-1,:],
            cmap=cmap,
            vmax=density.mean() + density.std() * vmax_mod,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        wrapup(out_png)
    else:
        ax.imshow(
            density[::-1,:],
            cmap=cmap,
            vmax=density.mean() + density.std() * vmax_mod,
        )
    return density 
    
def loc_density_from_trajs(
    trajs,
    ax = None,
    upsampling_factor = 20,
    verbose = False,
):
    locs = spazio.extract_positions_from_trajs(trajs)
    loc_density(locs, ax=ax, upsampling_factor=upsampling_factor,
        verbose=verbose)

def show_trajectories(
    trajs,
    ax = None,
    cmap = 'viridis',
    cap = 3000,
    verbose = True,
    n_colors = 100,
    color_by = None,
    upsampling_factor = 1,
):
    n_trajs = trajs.shape[0]
    if n_trajs > cap:
        trajs = trajs[:cap, :]
        n_trajs = trajs.shape[0]
    else:
        cap = n_trajs 

    if color_by == None:
        color_index = (np.arange(n_trajs) * 33).astype('uint16') % n_colors
    else:
        color_values = np.array([i[0] for i in trajs[:, color_by]])
        color_bins = np.arange(n_colors) * color_values.max() / (n_colors-1)
        color_index = np.digitize(color_values, bins = color_bins)

    colors = sns.color_palette(cmap, n_colors)

    if ax == None:
        fig, ax = plt.subplots(figsize = (4, 4))
        finish_plot = True
    else:
        finish_plot = False 

    for traj_idx in range(cap):
        traj = trajs[traj_idx]
        ax.plot(
            traj[0][:,1] * upsampling_factor - 0.5,
            traj[0][:,0] * upsampling_factor - 0.5,
            marker = '.',
            markersize = 2,
            linestyle = '-',
            linewidth = 0.5,
            color = colors[color_index[traj_idx]],
        )
        if verbose:
            sys.stdout.write('Finished plotting %d/%d trajectories...\r' % (traj_idx + 1, cap))
            sys.stdout.flush()
    ax.set_aspect('equal')


#
# Functions that operate directly on files
#
def plot_tracked_mat(
    tracked_mat_file,
    out_png,
    cmap = 'viridis',
    cap = 3000,
    verbose = True,
    n_colors = 100,
    color_index = None,
):
    spazio.check_file_exists(tracked_mat_file)

    trajs, metadata, traj_cols = spazio.load_trajs(tracked_mat_file)
    kwargs = {'cmap' : cmap, 'cap' : cap, 'verbose' : verbose,
        'n_colors' : n_colors, 'color_index' : color_index}
    ax = plot_trajectories(trajs, ax=None, **kwargs)
    wrapup(out_png)

def loc_density_from_file(
    loc_file,
    out_png,
    upsampling_factor = 20,
    kernel_width = 0.5,
    y_col = 'y_pixels',
    x_col = 'x_pixels',
    convert_to_um = True,
    vmax_mod = 1.0,
    verbose = False,
): 
    spazio.check_file_exists(loc_file)
    if 'Tracked.mat' in loc_file:
        trajs, metadata, traj_cols = spazio.load_trajs(loc_file)

    locs, metadata = spazio.load_locs(loc_file)
    density = loc_density(
        locs,
        metadata = metadata,
        ax = None,
        upsampling_factor = upsampling_factor,
        kernel_width = kernel_width,
        y_col = y_col, 
        x_col = x_col,
        convert_to_um = convert_to_um,
        vmax_mod = vmax_mod,
        verbose = verbose,
        out_png = out_png,
    )

def overlay_trajs(
    nd2_file,
    tracked_mat_file,
    start_frame,
    stop_frame,
    out_tif = None,
    vmax_mod = 1.0,
    upsampling_factor = 1,
    crosshair_len = 2,
    pixel_size_um = 0.16,
):
    n_frames_plot = stop_frame - start_frame + 1

    # Load trajectories and metadata
    trajs, metadata, traj_cols = spazio.load_trajs(tracked_mat_file)

    # Load image data
    reader = spazio.ImageFileReader(nd2_file)
    N, M, n_frames = reader.get_shape()

    # The output is upsampled to show localizations at higher-than-
    # pixel resolution
    N_up = N * upsampling_factor
    M_up = M * upsampling_factor

    # Get image min and max
    image_min, image_max = reader.min_max()
    vmin = image_min 
    vmax = image_max * vmax_mod 

    # Break apart the trajectories into individual localizations,
    # keeping track of the trajectory they came from
    locs = spazio.trajs_to_locs(trajs, traj_cols)
    n_locs = len(locs)
    required_columns = ['frame_idx', 'traj_idx', 'y_um', 'x_um']
    if any([c not in locs.columns for c in required_columns]):
        raise RuntimeError('overlay_trajs: dataframe must contain frame_idx, traj_idx, y_um, x_um')

    # Convert to ndarray
    locs = np.asarray(locs[required_columns])

    # Convert from um to pixels
    locs[:, 2:] = locs[:, 2:] * upsampling_factor / metadata['pixel_size_um']
    locs = locs.astype('int64') 


    # Add a unique random index for each trajectory
    new_locs = np.zeros((locs.shape[0], 5), dtype = 'int64')
    new_locs[:,:4] = locs 
    new_locs[:,4] = (locs[:,1] * 173) % 256
    locs = new_locs 

    # Do the plotting
    colors = generate_rainbow_palette()

    result = np.zeros((n_frames_plot, N_up, M_up * 2 + upsampling_factor, 4), dtype = 'uint8')
    frame_exp = np.zeros((N_up, M_up), dtype = 'uint8')
    for frame_idx in tqdm(range(n_frames_plot)):
        frame = reader.get_frame(frame_idx + start_frame).astype('float64')
        frame_rescaled = ((frame / vmax) * 255)
        frame_rescaled[frame_rescaled > 255] = 255 
        frame_8bit = frame_rescaled.astype('uint8')

        for i in range(upsampling_factor):
            for j in range(upsampling_factor):
                frame_exp[i::upsampling_factor, j::upsampling_factor] = frame_8bit

        result[frame_idx, :, :M_up, 3] = frame_exp.copy()
        result[frame_idx, :, M_up + upsampling_factor:, 3] = frame_exp.copy()

        result[frame_idx, :, M_up:M_up+upsampling_factor, :] = 255

        for j in range(3):
            result[frame_idx, :, :M_up, j] = frame_exp.copy()
            result[frame_idx, :, M_up + upsampling_factor:, j] = frame_exp.copy()

        locs_in_frame = locs[(locs[:,0] == frame_idx + start_frame).astype('bool'), :]

        for loc_idx in range(locs_in_frame.shape[0]):
            try:
                result[frame_idx, locs_in_frame[loc_idx, 2], M_up + locs_in_frame[loc_idx, 3] + upsampling_factor, :] = \
                    colors[locs_in_frame[loc_idx, 4], :]
            except (KeyError, ValueError, IndexError) as e2: #edge loc
                pass
            for j in range(1, crosshair_len + 1):
                try:
                    result[frame_idx, locs_in_frame[loc_idx, 2], M_up + locs_in_frame[loc_idx, 3] + j + upsampling_factor, :] = \
                        colors[locs_in_frame[loc_idx, 4], :]
                    result[frame_idx, locs_in_frame[loc_idx, 2], M_up + locs_in_frame[loc_idx, 3] - j + upsampling_factor, :] = \
                        colors[locs_in_frame[loc_idx, 4], :]
                    result[frame_idx, locs_in_frame[loc_idx, 2] + j, M_up + locs_in_frame[loc_idx, 3] + upsampling_factor, :] = \
                        colors[locs_in_frame[loc_idx, 4], :]
                    result[frame_idx, locs_in_frame[loc_idx, 2] - j, M_up + locs_in_frame[loc_idx, 3] + upsampling_factor, :] = \
                        colors[locs_in_frame[loc_idx, 4], :]
                except (KeyError, ValueError, IndexError) as e3:  #edge loc 
                    continue 

    if out_tif == None:
        out_tif = 'default_overlay_trajs.tif'

    tifffile.imsave(out_tif, result)
    reader.close()

#
# Interactive functions -- for Jupyter notebooks
#

def overlay_trajs_interactive(
    nd2_file,
    tracked_mat_file,
    start_frame,
    stop_frame,
    vmax_mod = 1.0,
    upsampling_factor = 1,
    crosshair_len = 'dynamic',
    continuous_update = True
):
    out_tif = '%soverlay.tif' % tracked_mat_file.replace('Tracked.mat', '')
    if crosshair_len == 'dynamic':
        crosshair_len = int(3 * upsampling_factor)
    overlay_trajs(
        nd2_file,
        tracked_mat_file,
        start_frame,
        stop_frame,
        out_tif = out_tif,
        vmax_mod = vmax_mod,
        upsampling_factor = upsampling_factor,
        crosshair_len = crosshair_len,
    )

    reader = tifffile.TiffFile(out_tif)
    n_frames = len(reader.pages)
    
    def update(frame_idx):
        fig, ax = plt.subplots(figsize = (14, 7))
        page = reader.pages[frame_idx].asarray()
        page[:,:,-1] = 255
        ax.imshow(
            page,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine_dir in ['top', 'bottom', 'left', 'bottom']:
            ax.spines[spine_dir].set_visible(False)
        plt.show(); plt.close()

    interact(update, frame_idx = widgets.IntSlider(
        min=0, max=n_frames, continuous_update=continuous_update))

def overlay_locs_interactive(
    locs,
    nd2_file,
    vmax_mod = 0.5,
    dpi = 20,
    continuous_update = False,
):
    # Load the ND2 file
    reader = spazio.ImageFileReader(nd2_file)
    N, M, n_frames = reader.get_shape()

    # Figure out the intensity scaling
    stack_min, stack_max = reader.min_max()
    vmin = stack_min
    vmax = stack_max * vmax_mod

    # Define the update function
    def update(frame_idx):
        fig, ax = plt.subplots(figsize = (8, 8))
        ax.imshow(
            reader.get_frame(frame_idx),
            cmap = 'gray',
            vmin = vmin,
            vmax = vmax,
        )
        ax.plot(
            locs.loc[locs['frame_idx'] == frame_idx]['x_pixels'] - 0.5,
            locs.loc[locs['frame_idx'] == frame_idx]['y_pixels'] - 0.5,
            marker = '.',
            markersize = 15,
            color = 'r',
            linestyle = '',
        )
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show(); plt.close()

    interact(update, frame_idx = widgets.IntSlider(
        min=0, max=n_frames, continuous_update=continuous_update))

def optimize_detection_interactive(
    image_file,
    offset_by_half = False,
):
    reader = spazio.ImageFileReader(image_file)
    N, M, n_frames = reader.get_shape()

    def update(frame_idx, sigma, detect_threshold, window_size):
        image = reader.get_frame(frame_idx)
        LL, detections, peaks, detect_positions = localize._detect_gaussian_kernel(
            image,
            sigma = sigma,
            detect_threshold = detect_threshold,
            window_size = window_size,
            offset_by_half = offset_by_half,
        )

        plot_image = image.copy()
        im_max = plot_image.max()
        for pos_idx in range(detect_positions.shape[0]):
            y, x = detect_positions[pos_idx, :].astype('uint16') + 1
            for i in range(-2, 3):
                try:
                    plot_image[y+i,x] = im_max 
                except IndexError:
                    pass
                try:
                    plot_image[y,x+i] = im_max 
                except IndexError:
                    pass 

        fig, ax = plt.subplots(2, 2, figsize = (16, 16))
        ax[0,0].imshow(
            image, cmap = 'gray'
        )
        ax[0,1].imshow(
            LL, cmap = 'gray',
        )
        ax[1,0].imshow(
            detections, cmap = 'gray',
        )
        ax[1,1].imshow(
            plot_image, cmap = 'gray',
        )
        for i in range(2):
            for j in range(2):
                ax[i,j].set_xticks([])
                ax[i,j].set_yticks([])
        ax[0,0].set_title('Original', fontdict = {'fontsize' : 30})
        ax[0,1].set_title('Log-likelihood map', fontdict = {'fontsize' : 30})
        ax[1,0].set_title('LLR > threshold', fontdict = {'fontsize' : 30})
        ax[1,1].set_title('Detections', fontdict = {'fontsize' : 30})

        plt.show(); plt.close()

    interact(
        update,
        frame_idx = widgets.IntSlider(min = 0, max = n_frames, continuous_update = False),
        sigma = widgets.FloatSlider(min = 0.4, max = 3.0, continuous_update = False),
        detect_threshold = widgets.FloatSlider(min = 10.0, max = 30.0, continuous_update = False),
        window_size = widgets.IntSlider(min = 3, max = 21, step = 2, continuous_update = False),
    )

def optimize_detection_dog_interactive(
    image_file,
):
    reader = spazio.ImageFileReader(image_file)
    N, M, n_frames = reader.get_shape()

    def update(
        frame_idx,
        bg_kernel_width = 10,
        bg_sub_mag = 1.0,
        spot_kernel_width = 1.0,
        threshold = 10000.0,
    ):
        image = reader.get_frame(frame_idx)
        image_bg, image_bg_sub, image_filt, detections, peaks, detect_positions = \
            localize._detect_dog_filter(
                image,
                bg_kernel_width = bg_kernel_width,
                bg_sub_mag = bg_sub_mag,
                spot_kernel_width = spot_kernel_width,
                threshold = threshold,
            )

        fig, ax = plt.subplots(2, 2, figsize = (16, 16))
        ax[0,0].imshow(image, cmap = 'gray')
        ax[0,1].imshow(image_bg, cmap = 'gray')
        ax[1,0].imshow(image_filt, cmap = 'gray')
        ax[1,1].imshow(detections, cmap = 'gray')

        plt.show(); plt.close()

    interact(
        update,
        frame_idx = widgets.IntSlider(value = 0, min = 0, max = n_frames-1, continuous_update = False),
        bg_kernel_width = widgets.IntSlider(value = 10, min = 1, max = 20, continuous_update = False),
        bg_sub_mag = widgets.FloatSlider(value = 1.0, min = 0.0, max = 2.0, continuous_update = False),
        spot_kernel_width = widgets.FloatSlider(value = 1.0, min = 0.2, max = 2.0, continuous_update = False),
        threshold = widgets.FloatSlider(value = 3000, min = 100.0, max = 10000.0, continuous_update = False),
    )

def optimize_detection_log_interactive(
    image_file,
):
    reader = spazio.ImageFileReader(image_file)
    N, M, n_frames = reader.get_shape()

    def update(
        frame_idx,
        bg_kernel_width = 10,
        bg_sub_mag = 1.0,
        spot_kernel_width = 2.0,
        threshold = 10000.0,
    ):
        image = reader.get_frame(frame_idx)
        image_filt, detections, peaks, detect_positions = \
            localize._detect_log_filter(
                image,
                bg_kernel_width = bg_kernel_width,
                bg_sub_mag = bg_sub_mag,
                spot_kernel_width = spot_kernel_width,
                threshold = threshold,
            )

        plot_image = image.copy()
        image_max = plot_image.max()
        for detect_idx in range(detect_positions.shape[0]):
            y, x = detect_positions[detect_idx, :].astype('uint16')
            for i in range(-2, 3):
                try:
                    plot_image[y+i, x] = image_max
                except IndexError:
                    pass
                try:
                    plot_image[y, x+i] = image_max 
                except IndexError:
                    pass 

        fig, ax = plt.subplots(2, 2, figsize = (16, 16))
        ax[0,0].imshow(image, cmap = 'gray')
        ax[0,1].imshow(image_filt, cmap = 'gray')
        ax[1,0].imshow(detections, cmap = 'gray')
        ax[1,1].imshow(plot_image, cmap = 'gray')

        plt.show(); plt.close()

    interact(
        update,
        frame_idx = widgets.IntSlider(value = 0, min = 0, max = n_frames-1, continuous_update = False),
        bg_kernel_width = widgets.IntSlider(value = 10, min = 1, max = 20, continuous_update = False),
        bg_sub_mag = widgets.FloatSlider(value = 1.0, min = 0.0, max = 2.0, continuous_update = False),
        spot_kernel_width = widgets.FloatSlider(value = 2.0, min = 0.5, max = 5.0, continuous_update = False),
        threshold = widgets.FloatSlider(value = 100, min = 0.1, max = 1000.0, continuous_update = False),
    )




# 
# Plotting utilities
#


def generate_rainbow_palette(n_colors = 256):
    '''
    Generate a rainbow color palette in RGBA format.
    '''
    result = np.zeros((n_colors, 4), dtype = 'uint8')
    for color_idx in range(n_colors):
        result[color_idx, :] = (np.asarray(cm.gist_rainbow(color_idx)) * \
            255).astype('uint8')
    return result 












