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
import pyspaz 
from pyspaz import spazio 
from pyspaz import localize
from pyspaz import interpolate_mask 

# Plotting
import matplotlib
import matplotlib.pyplot as plt 
from matplotlib import cm
import seaborn as sns
sns.set(style = 'ticks')

# Progress bar
from tqdm import tqdm 

# Interactive functions for Jupyter notebooks
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual

def imshow(*imgs, vmax=1.0):
    n = len(imgs)
    if n == 1:
        fig, ax = plt.subplots(figsize = (3, 3))
        ax.imshow(imgs[0], cmap='gray', vmax = imgs[0].max()*vmax)
        plt.show(); plt.close()
    else:
        fig, ax = plt.subplots(1, n, figsize = (3 * n, 3))
        for i in range(n):
            ax[i].imshow(imgs[i], cmap='gray', vmax=imgs[i].max()*vmax)
        plt.show(); plt.close()

def wrapup(out_png, dpi = 400, open_result = True):
    ''' Save a plot to PNG '''
    plt.tight_layout()
    plt.savefig(out_png, dpi = dpi)
    plt.close()
    if open_result:
        os.system('open %s' % out_png)

def loc_density(
    locs,
    metadata,
    ax = None,
    upsampling_factor = 20,
    kernel_width = 0.5,
    verbose = False,
    y_col = 'y_pixels',
    x_col = 'x_pixels',
    convert_to_um = False,
    vmax_mod = 1.0,
    cmap = 'gray',
    out_png = 'default_loc_density_out.png',
    save_plot = True,
): 
    # Get the list of localized positions
    m_keys = list(metadata.keys())
    positions = np.asarray(locs[[y_col, x_col]])
    if convert_to_um and ('pixel_size_um' in m_keys):
        metadata['pixel_size_um'] = float(metadata['pixel_size_um'])
        positions = positions * metadata['pixel_size_um']

    # Make the size of the out frame
    if ('N' in m_keys) and ('M' in m_keys):
        metadata['N'] = int(metadata['N'])
        metadata['M'] = int(metadata['M'])
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
    if (save_plot == True):
        if (ax == None):
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

def show_locs(
    locs,
    ax = None,
    color_attrib = 'I0',
    cmap = 'inferno',
    max_value = 300,
    min_value = 0,
    ylim = None,
    xlim = None,
):
    if ax == None:
        fig, ax = plt.subplots(figsize = (6, 6))
        finish_plot = True 
    else:
        finish_plot = False 

    color_index = locs[color_attrib].copy()
    color_index = (color_index - min_value)
    color_index[color_index > max_value] = max_value

    ax.scatter(
        locs['x_pixels'],
        locs['y_pixels'],
        c = color_index,
        cmap = cmap,
        s = 2,
    )
    if type(ylim) == type((0,0)):
        ax.set_ylim(ylim)
    if type(xlim) == type((0,0)):
        ax.set_xlim(xlim)

    if finish_plot:
        plt.show(); plt.close()

def attrib_dist(
    locs,
    ax = None,
    attrib = 'I0',
    color = '#C2C2C2',
    bin_size = 20,
    max_value = 500,
    label = None,
):
    bin_edges = np.arange(0, max_value+bin_size, bin_size)
    histo, edges = np.histogram(locs[attrib], bins=bin_edges)
    bin_centers = bin_edges[:-1] + bin_size/2

    if ax == None:
        fig, ax = plt.subplots(figsize = (3, 2))
        finish_plot = True 
    else:
        finish_plot = False 

    ax.bar(
        bin_centers,
        histo,
        width = bin_size * 0.8,
        edgecolor = 'k',
        linewidth = 2,
        color = color,
    )
    if label == None:
        ax.set_xlabel(attrib)
    else:
        ax.set_xlabel(label)
    ax.set_ylabel('Localizations')

    if finish_plot:
        plt.tight_layout(); plt.show(); plt.close()

    
def loc_density_from_trajs_mat_format(
    trajs,
    metadata,
    ax = None,
    upsampling_factor = 20,
    verbose = False,
):
    locs = pd.DataFrame(spazio.extract_positions_from_trajs(trajs)/metadata['pixel_size_um'], columns = ['y_um', 'x_um'])
    density = loc_density(
        locs,
        metadata = metadata,
        ax=ax,
        upsampling_factor=upsampling_factor,
        verbose=verbose,
        y_col = 'y_um',
        x_col = 'x_um',
        convert_to_um = False,
    )
    return density 

def show_trajectories_mat_format(
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

    if finish_plot:
        plt.show(); plt.close()

def show_trajectories(
    trajs,
    ax = None,
    cmap = 'viridis',
    cap = 3000,
    color_by = None,
    color_by_max = None,
    n_colors = 256,
    min_traj_len = 1,
):
    if min_traj_len > 1:
        if not 'traj_len' in trajs.columns:
            utils.assign_traj_len(trajs)
        trajs = trajs.loc[trajs['traj_len'] >= min_traj_len]
    if ax == None:
        fig, ax = plt.subplots(figsize = (4, 4))
        finish_plot = True
    else:
        finish_plot = False

    n_trajs = trajs['traj_idx'].max()

    colors = sns.color_palette('viridis', n_colors)
    if color_by != None:
        if color_by_max == None:
            max_value = trajs[color_by].max()
        else:
            max_value = color_by_max 

    c_idx = 0
    for traj_idx, traj in tqdm(trajs.groupby('traj_idx')):
        if c_idx >= cap: break 
        if color_by == None:
            color_idx = (traj_idx * 173) % n_colors
        else:
            color_idx = int((traj[color_by].iloc[0] * (n_colors-1)) / max_value)
            if color_idx > n_colors:
                color_idx = n_colors - 1
        ax.plot(
            traj['x_pixels'],
            traj['y_pixels'],
            color = colors[color_idx],
            marker = '.',
        )
        c_idx += 1
    ax.set_aspect('equal')

    if finish_plot:
        plt.show(); plt.close()

def show_masked_trajectories(
    trajs,
    mask_column,
    ax = None,
    cmaps = ['inferno', 'viridis'],
    cap = 3000,
    n_colors = 256,
    min_traj_len = 1,
    criterion = 'any',
):
    '''
    Plot individual trajectories, using separate color schemes
    for trajectories inside and outside of a mask.

    args
        trajs                   :   pandas.DataFrame
        mask_column             :   str, boolean column in *trajs*
                                        that indicates mask 
                                        membership
        ax                      :   matplotlib.Axis object
        cmaps                   :   list of str
        cap                     :   int, max # trajectories to plot
        n_colors                :   int
        min_traj_len            :   int
        criterion               :   'any' or 'all', whether to 
                                        include trajectories that
                                        contain any or all localizations
                                        inside the mask

    '''
    # Plot only long trajectories, if desired
    if min_traj_len > 1:
        if not 'traj_len' in trajs.columns:
            utils.assign_traj_len(trajs)

    # If the user doesn't specify a matplotlib.Axis
    # object, make one
    if ax == None:
        fig, ax = plt.subplots(figsize = (4, 4))
        finish_plot = True 
    else:
        finish_plot = False

    # Generate the color palettes
    n_trajs = trajs['traj_idx'].max()
    color_palettes = [
        sns.color_palette(cmaps[0], n_colors),
        sns.color_palette(cmaps[1], n_colors),
    ]

    c_idx = 0
    for traj_idx, traj in tqdm(trajs.groupby('traj_idx')):
        if c_idx >= cap: break
        if (criterion == 'any' and traj[mask_column].any()) or \
            (criterion == 'all' and traj[mask_column].all()):
            color = color_palettes[0][(traj_idx * 173) % n_colors]
        else:
            color = color_palettes[1][(traj_idx * 173) % n_colors]

        ax.plot(
            traj['x_pixels'],
            traj['y_pixels'],
            color = color,
            marker = '.',
        )
        c_idx += 1
    ax.set_aspect('equal')
    if finish_plot:
        plt.show(); plt.close()



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

def overlay_trajs_df(
    nd2_file,
    trajs,
    start_frame,
    stop_frame,
    out_tif = None,
    vmax_mod = 1.0,
    upsampling_factor = 1,
    crosshair_len = 2,
    white_out_singlets = True,
): 
    n_frames_plot = stop_frame - start_frame + 1
    reader = spazio.ImageFileReader(nd2_file)
    N, M, n_frames = reader.get_shape()
    N_up = N * upsampling_factor 
    M_up = M * upsampling_factor
    image_min, image_max = reader.min_max()
    vmin = image_min
    vmax = image_max * vmax_mod 

    trajs = trajs.assign(color_idx = (trajs['traj_idx'] * 173) % 256)
    n_locs = len(trajs)
    required_columns = ['frame_idx', 'traj_idx', 'y_pixels', 'x_pixels']
    if any([c not in trajs.columns for c in required_columns]):
        raise RuntimeError('overlay_trajs: dataframe must contain frame_idx, traj_idx, y_pixels, x_pixels')

   # Convert to ndarray -> faster indexing
    locs = np.asarray(trajs[required_columns])

    # Convert to upsampled pixels
    locs[:, 2:] = locs[:, 2:] * upsampling_factor
    locs = locs.astype('int64') 

    # Add a unique random index for each trajectory
    new_locs = np.zeros((locs.shape[0], 5), dtype = 'int64')
    new_locs[:,:4] = locs 
    new_locs[:,4] = (locs[:,1] * 173) % 256
    locs = new_locs 

    # If the length of a trajectory is 1, then make its color white
    if white_out_singlets:
        for traj_idx in range(locs[:,1].max()):
            if (locs[:,1] == traj_idx).sum() == 1:
                locs[(locs[:,1] == traj_idx), 4] = -1

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

            # Get the color corresponding to this trajectory
            color_idx = locs_in_frame[loc_idx, 4]
            if color_idx == -1:
                color = np.array([255, 255, 255, 255]).astype('uint8')
            else:
                color = colors[color_idx, :]

            try:
                result[frame_idx, locs_in_frame[loc_idx, 2], M_up + locs_in_frame[loc_idx, 3] + upsampling_factor, :] = color 
            except (KeyError, ValueError, IndexError) as e2: #edge loc
                pass
            for j in range(1, crosshair_len + 1):
                try:
                    result[frame_idx, locs_in_frame[loc_idx, 2], M_up + locs_in_frame[loc_idx, 3] + j + upsampling_factor, :] = color
                    result[frame_idx, locs_in_frame[loc_idx, 2], M_up + locs_in_frame[loc_idx, 3] - j + upsampling_factor, :] = color
                    result[frame_idx, locs_in_frame[loc_idx, 2] + j, M_up + locs_in_frame[loc_idx, 3] + upsampling_factor, :] = color 
                    result[frame_idx, locs_in_frame[loc_idx, 2] - j, M_up + locs_in_frame[loc_idx, 3] + upsampling_factor, :] = color
                except (KeyError, ValueError, IndexError) as e3:  #edge loc 
                    continue 

    if out_tif == None:
        out_tif = 'default_overlay_trajs.tif'

    tifffile.imsave(out_tif, result)
    reader.close()


def overlay_trajs_tracked_mat(
    nd2_file,
    tracked_mat_file,
    start_frame,
    stop_frame,
    out_tif = None,
    vmax_mod = 1.0,
    upsampling_factor = 1,
    crosshair_len = 2,
    pixel_size_um = 0.16,
    white_out_singlets = True,
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

    # If the length of a trajectory is 1, then make its color white
    if white_out_singlets:
        for traj_idx in range(locs[:,1].max()):
            if (locs[:,1] == traj_idx).sum() == 1:
                locs[(locs[:,1] == traj_idx), 4] = -1

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

            # Get the color corresponding to this trajectory
            color_idx = locs_in_frame[loc_idx, 4]
            if color_idx == -1:
                color = np.array([255, 255, 255, 255]).astype('uint8')
            else:
                color = colors[color_idx, :]

            try:
                result[frame_idx, locs_in_frame[loc_idx, 2], M_up + locs_in_frame[loc_idx, 3] + upsampling_factor, :] = color 
            except (KeyError, ValueError, IndexError) as e2: #edge loc
                pass
            for j in range(1, crosshair_len + 1):
                try:
                    result[frame_idx, locs_in_frame[loc_idx, 2], M_up + locs_in_frame[loc_idx, 3] + j + upsampling_factor, :] = color
                    result[frame_idx, locs_in_frame[loc_idx, 2], M_up + locs_in_frame[loc_idx, 3] - j + upsampling_factor, :] = color
                    result[frame_idx, locs_in_frame[loc_idx, 2] + j, M_up + locs_in_frame[loc_idx, 3] + upsampling_factor, :] = color 
                    result[frame_idx, locs_in_frame[loc_idx, 2] - j, M_up + locs_in_frame[loc_idx, 3] + upsampling_factor, :] = color
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
    trajs,
    start_frame,
    stop_frame,
    vmax_mod = 1.0,
    upsampling_factor = 1,
    crosshair_len = 'dynamic',
    continuous_update = True,
    white_out_singlets = True,
):
    if crosshair_len == 'dynamic':
        crosshair_len = int(3 * upsampling_factor)

    if type(trajs) == type('') and 'Tracked.mat' in trajs:
        out_tif = '%soverlay.tif' % tracked_mat_file.replace('Tracked.mat', '')
        overlay_trajs_tracked_mat(
            nd2_file,
            trajs,
            start_frame,
            stop_frame,
            out_tif = out_tif,
            vmax_mod = vmax_mod,
            upsampling_factor = upsampling_factor,
            crosshair_len = crosshair_len,
            white_out_singlets = white_out_singlets,
        )
    elif type(trajs) == type('') and ('.trajs' in trajs or '.txt' in trajs):
        out_tif = '%s_overlay.tif' % os.path.splitext(trajs)[0]
        trajs, metadata = spazio.load_locs(trajs)
        print(trajs.columns)
        overlay_trajs_df(
            nd2_file,
            trajs,
            start_frame,
            stop_frame,
            out_tif = out_tif,
            vmax_mod = vmax_mod,
            upsampling_factor = upsampling_factor,
            crosshair_len = crosshair_len,
            white_out_singlets = white_out_singlets,
        )
    elif type(trajs) == type(pd.DataFrame([])):
        out_tif = 'default_overlay.tif'
        overlay_trajs_df(
            nd2_file,
            trajs,
            start_frame,
            stop_frame,
            out_tif = out_tif,
            vmax_mod = vmax_mod,
            upsampling_factor = upsampling_factor,
            crosshair_len = crosshair_len,
            white_out_singlets = white_out_singlets,
        )
    else:
        raise RuntimeError('overlay_trajs_interactive: trajs argument not understood')

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

def overlay_single_traj_interactive(
    nd2_file,
    trajs,
    traj_idx,
    vmax_mod = 1.0,
    upsampling_factor = 1,
    crosshair_len = 'dynamic',
    continuous_update = True,
):
    '''
    args
        nd2_file        :   str
        trajs           :   pandas.DataFrame
        traj_idx        :   int
        vmax_mod        :   float
        upsampling_factor:  float
        crosshair_len   :   str
        continuous_update   :   bool
    '''
    if crosshair_len == 'dynamic':
        crosshair_len = int(3 * upsampling_factor)

    traj = trajs.loc[trajs['traj_idx'] == traj_idx]
    start_frame = max([0, traj['frame_idx'].min() - 1])
    stop_frame = min([trajs['frame_idx'].max(), traj['frame_idx'].max() + 1])

    trajs_copy = trajs.copy()
    trajs_copy.loc[trajs_copy['traj_idx'] != traj_idx, 'traj_idx'] = traj_idx + 100

    out_tif = 'default_overlay.tif'
    overlay_trajs_df(
        nd2_file,
        trajs_copy,
        start_frame,
        stop_frame,
        out_tif = out_tif,
        vmax_mod = vmax_mod,
        upsampling_factor = upsampling_factor,
        crosshair_len = crosshair_len,
        white_out_singlets = True,
    )
    reader = tifffile.TiffFile(out_tif)
    n_frames = len(reader.pages) - 1

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
        fig, ax = plt.subplots(1, 2, figsize = (12, 6))
        for j in range(2):
            ax[j].imshow(
                reader.get_frame(frame_idx),
                cmap = 'gray',
                vmin = vmin,
                vmax = vmax,
            )
        ax[1].plot(
            locs.loc[locs['frame_idx'] == frame_idx]['x_pixels'] - 0.5,
            locs.loc[locs['frame_idx'] == frame_idx]['y_pixels'] - 0.5,
            marker = '+',
            markersize = 15,
            color = '#F70000',
            linestyle = '',
        )
        for j in range(2):
            ax[j].set_xticks([])
            ax[j].set_yticks([])
        plt.show(); plt.close()

    interact(update, frame_idx = widgets.IntSlider(
        min=0, max=n_frames, continuous_update=continuous_update))

def optimize_detection_interactive(
    image_file,
    offset_by_half = False,
    vmax_mod = 1.0,
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
            image, cmap = 'gray', vmax = image.max() * vmax_mod 
        )
        ax[0,1].imshow(
            LL, cmap = 'gray',
        )
        ax[1,0].imshow(
            detections, cmap = 'gray',
        )
        ax[1,1].imshow(
            plot_image, cmap = 'gray', vmax = image.max() * vmax_mod, 
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
        frame_idx = widgets.IntSlider(value = 0, min = 0, max = n_frames, continuous_update = False),
        sigma = widgets.FloatSlider(value = 1.0, min = 0.4, max = 3.0, continuous_update = False),
        detect_threshold = widgets.FloatSlider(value = 20.0, min = 10.0, max = 30.0, continuous_update = False),
        window_size = widgets.IntSlider(value = 9, min = 3, max = 21, step = 2, continuous_update = False),
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
        for i in range(2):
            for j in range(2):
                ax[i,j].set_xticks([])
                ax[i,j].set_yticks([])
        ax[0,0].set_title('Original', fontdict = {'fontsize' : 30})
        ax[0,1].set_title('conv w/ BG kernel', fontdict = {'fontsize' : 30})
        ax[1,0].set_title('DoG-filtered', fontdict = {'fontsize' : 30})
        ax[1,1].set_title('Detections', fontdict = {'fontsize' : 30})

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
        for i in range(2):
            for j in range(2):
                ax[i,j].set_xticks([])
                ax[i,j].set_yticks([])
        ax[0,0].set_title('Original', fontdict = {'fontsize' : 30})
        ax[0,1].set_title('LoG-filtered', fontdict = {'fontsize' : 30})
        ax[1,0].set_title('> threshold', fontdict = {'fontsize' : 30})
        ax[1,1].set_title('Detections', fontdict = {'fontsize' : 30})

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
# Visualize mask interpolator object
#
def plot_mask_interpolator(
    mask_interpolator,
    ax = None,
    n_frames = 100,
    cmap = 'viridis',
):
    '''
    Visualize a mask interpolator object by overlaying the 
    interpolated mask edges at various frames.

    args
        mask_interpolator       :   either a pyspaz.interpolate_mask.LinearMaskInterpolator
                                    or pyspaz.interpolate_mask.SplineMaskInterpolator

        ax                      :   matplotlib.axes

        n_frames                :   int, the number of frame intervals
                                    to use

        cmap                    :   str, argument for seaborn.color_palette

    returns
        None

    '''
    # Make sure a correct mask interpolator object is passed
    if 'interpolate' not in dir(mask_interpolator):
        raise RuntimeError('pyspaz.visualize.plot_mask_interpolator: interpolator must have an interpolate method')

    if ax == None:
        fig, ax = plt.subplots(figsize = (6, 6))
        finish_plot = True
    else:
        finish_plot = False

    colors = sns.color_palette(cmap, n_frames)

    min_frame = mask_interpolator.min_frame
    max_frame = mask_interpolator.max_frame 
    frames = np.linspace(min_frame, max_frame, n_frames)
    for frame_idx, frame in enumerate(frames):
        edge = mask_interpolator.interpolate(frame)
        ax.plot(
            edge[:,0],
            edge[:,1],
            color = colors[frame_idx],
            linestyle = '-',
            marker = None,
        )

        # Reconnect the first with the last point
        ax.plot(
            [edge[-1,0], edge[0,0]],
            [edge[-1,1], edge[0,1]],
            color = colors[frame_idx],
            linestyle = '-',
            marker = None,
        )

    ax.set_aspect('equal')

    if finish_plot:
        plt.show(); plt.close()

def plot_mask_interpolators(
    mask_interpolators,
    ax = None,
    n_frames = 100,
    cmap = 'viridis',
):
    '''
    Visualize a set of mask interpolator objects by overlaying the 
    interpolated mask edges at various frames.

    args
        mask_interpolators      :   list of either pyspaz.interpolate_mask.LinearMaskInterpolator
                                    or pyspaz.interpolate_mask.SplineMaskInterpolator
                                    objects

        ax                      :   matplotlib.axes

        n_frames                :   int, the number of frame intervals
                                    to use

        cmap                    :   str, argument for seaborn.color_palette

    returns
        None
        
    '''
    if ax == None:
        fig, ax = plt.subplots(figsize = (6, 6))
        finish_plot = True
    else:
        finish_plot = False 

    for mask_interpolator in mask_interpolators:
        plot_mask_interpolator(
            mask_interpolator,
            ax = ax,
            n_frames = n_frames,
            cmap = cmap,
        )

    if finish_plot:
        plt.show(); plt.close()

#
# Visualize model fits to radial displacement histograms
#
def plot_radial_disps_with_model(
    trajs,
    model_pdf_support,
    model_pdf,
    model_cdf_support,
    model_cdf,
    dt = 0.00548, # sec
    pdf_max_r = 2.0,
    n_bins_pdf = 51,
    n_bins_cdf = 5001,
    time_delays = 4,
    n_gaps = 0,
    color_palette = 'magma',
):
    '''
    Plot radial displacement CDFs with a model.

    UNTESTED.


    '''    
    # Make the empirical radial displacement histograms
    radial_disp_histograms_pdf, bin_edges_pdf = traj_analysis.compile_displacements(
        trajs,
        n_gaps = n_gaps,
        time_delays = time_delays,
        n_bins = n_bins_pdf,
        max_disp = 5.0,
        pixel_size_um = 0.16,
        traj_col = 'traj_idx',
        frame_col = 'frame_idx',
        y_col = 'y_pixels',
        x_col = 'x_pixels',
    )
    radial_disp_histograms_cdf, bin_edges_cdf = traj_analysis.compile_displacements(
        trajs,
        n_gaps = n_gaps,
        time_delays = time_delays,
        n_bins = n_bins_cdf,
        max_disp = 5.0,
        pixel_size_um = 0.16,
        traj_col = 'traj_idx',
        frame_col = 'frame_idx',
        y_col = 'y_pixels',
        x_col = 'x_pixels',
    )

    # Get the size of the empirical radial displacement bins to plot
    exp_bin_size = bin_edges_pdf[1] - bin_edges_pdf[0]
    exp_bar_width = exp_bin_size * 0.8
    exp_bin_centers = bin_edges_pdf[:-1] + exp_bin_size/2

    # Get the size of the model bins, and figure out the inflation factor
    # for model vs. empirical
    model_bin_size_pdf = model_pdf_support[1] - model_pdf_support[0]
    pdf_inflation_factor = model_bin_size_pdf / exp_bin_size 
    model_pdf_inflated = model_pdf * pdf_inflation_factor 

    # Plot each time delay as a separate PDF
    fig, ax = plt.subplots(time_delays, 1, figsize = (2.8, 0.9 * time_delays))
    palette = sns.color_palette(color_palette, time_delays)

    for dt_idx in range(time_delays):
        exp_pdf = radial_disp_histograms[:, dt_idx] / radial_disp_histograms[:, dt_idx].sum()
        ax[dt_idx].bar(
            exp_bin_centers,
            exp_pdf,
            color = palette[dt_idx],
            edgecolor = 'black',
            linewidth = 1,
            width = exp_bar_width,
            label = None,
        )
        ax[dt_idx].plot(
            model_pdf_support,
            model_pdf_inflated,
            linestyle = '-',
            linewidth = 1.5,
            color = 'k',
            label = None,
        )
        ax[dt_idx].plot(
            [], [], linestyle = '',
            marker = None, color = 'w',
            label = '$\Delta t = $%.4f sec' % ((dt_idx + 1) * dt)
        )
        ax[dt_idx].legend(frameon=False, prop={'size':6}, loc='upper right')
        ax[dt_idx].set_yticks([])
        if dt_idx != time_delays-1:
            for spine_dir in ['top', 'bottom', 'left', 'right']:
                ax[dt_idx].spines[spine_dir].set_visible(False)
            ax[dt_idx].set_xticks([])

        ax[dt_idx].set_xlim((0, pdf_max_r))
    ax[-1].set_xlabel('Radial displacement ($\mu$m)', fontsize = 10)

    if out_png != None:
        wrapup(out_png, dpi=600)
    else:
        plt.tight_layout(); plt.show(); plt.close()


def plot_pdf_with_model_from_histogram(
    radial_disp_histograms,
    histogram_bin_edges,
    model_pdf,
    model_bin_centers,
    dt,
    max_r = 2.0,
    exp_bin_size = 0.02,
    out_png = None,
    color_palette = 'magma',
    figsize_mod = 1.0,
    ax = None,
):
    '''
    Plot the radial displacement histograms of tracking data 
    alongside the model PDF.

    args
        radial_disp_histograms      :   2D ndarray of shape (n_bins, n_dt)

        histogram_bin_edges         :   1D ndarray of shape (n_bins_1,)

        model_pdf                   :   2D ndarray of shape (m, n_dt)

        model_bin_centers           :   1D ndarray of shape (m,)

        dt                          :   float, seconds between frames

        max_r                       :   float, maximum disp to show in um

        exp_bin_size                :   float, size of plot histogram bins
                                            in um. Must be larger than the
                                            bins in *histogram_bin_edges*

        out_png                     :   str, file to save plot to

        color_palette               :   str

    returns
        None

    '''
    # Check user inputs
    assert len(radial_disp_histograms.shape) == 2
    assert len(histogram_bin_edges.shape) == 1
    assert len(model_pdf.shape) == 2
    assert len(model_bin_centers.shape) == 1
    assert radial_disp_histograms.shape[1] == model_pdf.shape[1]
    assert model_pdf.shape[0] == model_bin_centers.shape[0]

    # Get bar graph parameters
    n_bins, n_dt = radial_disp_histograms.shape 
    exp_bin_size_orig = histogram_bin_edges[1] - histogram_bin_edges[0]
    model_bin_size = model_bin_centers[1] - model_bin_centers[0]

    # The number of original bins per plot bin
    aggregation_factor = int(exp_bin_size / exp_bin_size_orig)
    n_bins_plot = radial_disp_histograms.shape[0] // aggregation_factor

    new_displacements = np.zeros((n_bins_plot, n_dt), dtype = 'int64')
    for dt_idx in range(n_dt):
        for agg_idx in range(aggregation_factor):
            new_displacements[:, dt_idx] = new_displacements[:, dt_idx] + radial_disp_histograms[agg_idx::aggregation_factor, dt_idx]
    new_bin_edges = histogram_bin_edges[::aggregation_factor]

    exp_bar_width = exp_bin_size * 0.8
    exp_bin_centers = new_bin_edges[:-1] + exp_bin_size/2
    model_inflation_factor = exp_bin_size 

    if (ax is None):
        fig, ax = plt.subplots(n_dt, 1, figsize = (2.8 * figsize_mod, 0.9 * n_dt * figsize_mod))
        finish_plot = True
    else:
        finish_plot = False

    palette = sns.color_palette(color_palette, n_dt)
    for dt_idx in range(n_dt):
        exp_pdf = new_displacements[:, dt_idx] / new_displacements[:, dt_idx].sum()
        ax[dt_idx].bar(
            exp_bin_centers[:n_bins_plot],
            exp_pdf[:n_bins_plot],
            color = palette[dt_idx],
            edgecolor = 'black',
            linewidth = 1,
            width = exp_bar_width,
            label = None,
        )
        ax[dt_idx].plot(
            model_bin_centers,
            model_pdf[:, dt_idx] * model_inflation_factor,
            linestyle = '-',
            linewidth = 1.5,
            color = 'k',
            label = None,
        )
        ax[dt_idx].plot([], [], linestyle = '',
            marker = None, color = 'w', label = '$\Delta t = $%.4f sec' % ((dt_idx + 1) * dt))

        ax[dt_idx].legend(frameon=False, prop={'size':6}, loc='upper right')
        ax[dt_idx].set_yticks([])
        if dt_idx != n_dt-1:
            for spine_dir in ['top', 'left', 'right']:
                ax[dt_idx].spines[spine_dir].set_visible(False)
            ax[dt_idx].set_xticks([])
        else:
            for spine_dir in ['top', 'left', 'right']:
                ax[dt_idx].spines[spine_dir].set_visible(False)
        ax[dt_idx].set_xlim((0, max_r))
    ax[-1].set_xlabel('Radial displacement ($\mu$m)', fontsize = 10)

    if finish_plot:
        if out_png != None:
            wrapup(out_png, dpi = 600)
        else:
            plt.tight_layout(); plt.show(); plt.close()
    else:
        return ax 

def plot_cdf_with_model_from_histogram(
    radial_disp_histograms,
    histogram_bin_edges,
    model_cdf,
    model_bin_rights,
    dt,
    out_png = None,
    color_palette = 'magma',
    figsize_mod = 1.0,
):
    '''
    Plot the empirical distribution functions for radial displacements
    in tracking data alongside the model CDF.

    args
        radial_disp_histograms      :   2D ndarray of shape (n_bins, n_dt)

        histogram_bin_edges         :   1D ndarray of shape (n_bins_1,)

        model_pdf                   :   2D ndarray of shape (m, n_dt)

        model_bin_centers           :   1D ndarray of shape (m,)

        dt                          :   float, seconds between frames

        out_png                     :   str, file to save plot to

        color_palette               :   str

    returns
        None

    '''
    n_bins, n_dt = radial_disp_histograms.shape 
    bins_right = histogram_bin_edges[1:]

    assert n_bins == bins_right.shape[0]

    experiment_cdfs = np.zeros(radial_disp_histograms.shape, dtype = 'float64')
    for dt_idx in range(n_dt):
        experiment_cdfs[:, dt_idx] = np.cumsum(radial_disp_histograms[:, dt_idx])
        experiment_cdfs[:, dt_idx] = experiment_cdfs[:, dt_idx] / experiment_cdfs[-1, dt_idx]

    fig, ax = plt.subplots(2, 1, figsize = (3 * figsize_mod, 3 * figsize_mod),
        gridspec_kw = {'height_ratios' : [3, 1]},
        sharex = True,
    )
    palette = sns.color_palette(color_palette, n_dt)

    for dt_idx in range(n_dt):
        ax[0].plot(
            bins_right,
            experiment_cdfs[:, dt_idx],
            color = palette[dt_idx],
            linestyle = '-',
            label = '%.4f sec ' % ((dt_idx+1) * dt),
        )
        ax[0].plot(
            model_bin_rights,
            model_cdf[:, dt_idx],
            color = 'k',
            linestyle = '--',
            label = None,
        )
    ax[0].plot([], [], color = 'k', linestyle = '--', label = 'Model')
    ax[0].set_ylabel('CDF', fontsize = 12)
    ax[0].legend(frameon=False, prop={'size' : 6})

    # Currently the experimental CDFs must be the same shape
    # as the model CDF in order to subtract the two 
    residuals = experiment_cdfs - model_cdf 

    for dt_idx in range(n_dt):
        ax[1].plot(
            bins_right,
            residuals[:, dt_idx],
            color = palette[dt_idx],
            linestyle = '-',
            label = '%.4f sec' % ((dt_idx+1) * dt),
            linewidth = 1,
        )
    ax[1].set_xlabel('Radial displacement ($\mu$m)', fontsize = 12)
    ax[1].set_ylabel('Residuals', fontsize = 12)

    ax1_ylim = np.abs(residuals).max() * 1.5
    ax[1].set_ylim((-ax1_ylim, +ax1_ylim))
    fig.align_ylabels(ax)
    if out_png != None:
        wrapup(out_png, dpi = 600)
    else:
        plt.tight_layout(); plt.show(); plt.close()






# 
# Various low-level utilities
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

def hide_axis(matplotlib_axis):
    matplotlib_axis.grid(False)
    for spine_dir in ['top', 'bottom', 'left', 'right']:
        matplotlib_axis.spines[spine_dir].set_visible(False)
    matplotlib_axis.set_xticks([])
    matplotlib_axis.set_yticks([])










