'''
qc.py -- quality control utilities
'''
# Numerical tools / type conversions
import numpy as np 

# Dataframes
import pandas as pd 

# Plotting
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set(style = 'ticks')
from mpl_toolkits.mplot3d import Axes3D

# File reading
import tifffile

# pyspaz utilities
from pyspaz import spazio
from pyspaz import visualize
from pyspaz import mask 

# Progress bar
from tqdm import tqdm 

# Get operating system for some plotting options
import sys
import os

def localization_qc(
    locs,
    metadata,
    nd2_file = None,
    psf_window_size = 11,
    psf_distance_from_center = 0.25,
    loc_density_upsampling_factor = 10,
    y_col = 'y_pixels',
    x_col = 'x_pixels',
    text_summary = True,
    out_png = None,
):
    '''
    Make a panel of plots related to the quality of the 
    localizations. 

    args
        locs            :       pandas.DataFrame
        metadata        :       dict
        nd2_file        :       str, the ND2 file for the PSF 
                                calculation. If not given, attempts
                                to find it in the metadata
        psf_window_size                 :       int
        psf_distance_from_center        :   float
        loc_density_upsampling_factor   :   int
        y_col           :       str, column for y position in *locs*
        x_col           :       str, column for x position in *locs*
        text_summary    :       bool, make a short text summary 
                                    with parameters of interest
        out_png         :       str, file to save plot to

    '''
    if nd2_file == None:
        try: 
            nd2_file = metadata['image_file_path']
        except KeyError:
            raise RuntimeError("qc.localization_qc: Cannot find the attribute image_file_path in metadata")

    if 'err_y_pixels' in locs.columns:
        fig, ax = plt.subplots(3, 3, figsize = (9, 9))
    else:
        fig, ax = plt.subplots(2, 3, figsize = (9, 6))

    # Show the mean PSF 
    try:
        print('Compiling mean PSF...')
        psf = calculate_psf(
            nd2_file,
            locs,
            window_size = psf_window_size,
            distance_from_center = psf_distance_from_center,
            y_col = y_col,
            x_col = x_col,
        )
        ax[0,0].imshow(psf, cmap='inferno')
        ax[0,0].set_xticks([])
        ax[0,0].set_yticks([])
        ax[0,0].set_title('Mean PSF')
        ax[0,0].set_aspect('equal')
    except FileNotFoundError:
        print('Could not find file %s for the PSF calculation' % nd2_file)
        ax[0,0].grid(False)
        for spine_dir in ['top', 'bottom', 'left', 'right']:
            ax[0,0].spines[spine_dir].set_visible(False)
        ax[0,0].set_xticks([])
        ax[0,0].set_yticks([])

    # Show the distribution of photon counts per localization
    visualize.attrib_dist(
        locs,
        ax = ax[0,1],
        attrib = 'I0',
        label = 'Photon count',
        color = '#C2C2C2',
        max_value = 500,
        bin_size = 20,
    )

    # Show the distribution of photon counts as a function of
    # space
    visualize.show_locs(
        locs,
        ax = ax[0,2],
        color_attrib = 'I0',
        cmap = 'viridis',
        max_value = 500,
        min_value = 0,
        ylim = ((0, metadata['N'])),
        xlim = ((0, metadata['M'])),
    )
    ax[0,2].set_title('Photon counts')
    ax[0,2].set_aspect('equal')

    # Show the localization density
    density = visualize.loc_density(
        locs,
        metadata,
        ax = ax[1,0],
        upsampling_factor = loc_density_upsampling_factor,
        kernel_width = 0.4,
        verbose = False,
        y_col = y_col,
        x_col = x_col,
        convert_to_um = False,
        save_plot = True,
    )
    ax[1,0].set_xticks([])
    ax[1,0].set_yticks([])
    ax[1,0].set_title("Localization density")
    ax[1,0].set_aspect('equal')

    # Show the pixel localization density
    pixel_density, y_field, x_field = pixel_localization_density(
        locs,
        bin_size = 0.05,
        y_col = y_col,
        x_col = x_col,
        plot = False,
    )
    ax[1,1].imshow(pixel_density[::-1,:], cmap = 'gray', vmin = 0, vmax = pixel_density.max())
    ax[1,1].set_xticks([-0.5, 4.5, 9.5, 14.5, 19.5])
    ax[1,1].set_xticklabels([0.00, 0.25, 0.50, 0.75, 1.00])
    ax[1,1].set_yticks([-0.5, 4.5, 9.5, 14.5, 19.5])
    ax[1,1].set_yticklabels([0.00, 0.25, 0.50, 0.75, 1.00])
    ax[1,1].set_xlabel('x mod 1 (pixels)')
    ax[1,1].set_ylabel('y mod 1 (pixels)')
    ax[1,1].set_title('Pixel loc density')
    ax[1,1].set_aspect('equal')

    # Show the number of localizations per frame
    locs_per_frame, interval_times = loc_density_time(
        locs, metadata,
        ax = ax[1,2],
    )

    if 'err_y_pixels' in locs.columns:

        # Show localization error as a function of space
        new_locs = locs.assign(average_error = locs[['err_y_pixels', 'err_x_pixels']].mean(axis = 1) * 160)
        new_locs = new_locs[new_locs['average_error'] <= 500]
        new_locs = new_locs[new_locs['average_error'] > 0.0]
        visualize.show_locs(
            new_locs,
            ax = ax[2,0],
            color_attrib = 'average_error',
            cmap = 'inferno',
            max_value = 150,
            min_value = 0,
            ylim = ((0, metadata['N'])),
            xlim = ((0, metadata['M'])),
        )
        ax[2,0].set_title('Localization error')

        # Show distribution of pixel localization errors
        visualize.attrib_dist(
            new_locs,
            ax = ax[2,1],
            attrib = 'average_error',
            color = '#C2C2C2',
            max_value = 200,
            bin_size = 10,
        )
        ax[2,1].set_xlabel('Est. localization error (nm)')
        ax[2,1].set_ylabel('Localizations')

        # Show distribution of I0 errors
        visualize.attrib_dist(
            locs,
            ax = ax[2,2],
            attrib = 'err_I0',
            color = '#C2C2C2',
            max_value = 100,
            bin_size = 5,
        )
        ax[2,2].set_xlabel('Est. I0 error (photons)')
        ax[2,2].set_ylabel('Localizations')

    # Give a text summary, if desired
    if text_summary:

        # Some errors are crazy due to matrix inversion; take only reasonable errors
        err_i0 = locs[locs["err_I0"] < 500]['err_I0']
        err_bg = locs[locs['err_bg'] < 10]['err_bg']

        print('\tSummary:')
        print('%d localizations' % len(locs))
        print('Photon count:\t%.3f +/- %.3f' % (locs['I0'].mean(), locs['I0'].std()))
        print('Localization estimated error:\t%.3f +/- %.3f nm' % (new_locs['average_error'].mean(), new_locs['average_error'].std()))
        print('Photon count estimated error:\t%.3f +/- %.3f photons' % (err_i0.mean(), err_i0.std()))
        print('BG photon counting estimated error:\t%.3f +/- %.3f photons' % (err_bg.mean(), err_bg.std()))
        print('')

    # Save to a PNG, if desired
    plt.tight_layout()
    if type(out_png) == type(''):
        plt.savefig(out_png, dpi = 600)
        plt.close()
        if sys.platform == 'darwin':
            os.system('open %s' % out_png)
    else:
        plt.show(); plt.close()

def tracking_qc(
    trajs,
    metadata,
    out_png = None,
    y_col = 'y_pixels',
    x_col = 'x_pixels',
):
    '''
    Make a summary plot with various quantities related to the
    quality of trajectories.

    args
        trajs           :   pandas.DataFrame
        metadata        :   dict
        out_png         :   str, out file if desired
        y_col, x_col    :   str, columns in *trajs* with the localization
                                coordinates


    '''
    fig, ax = plt.subplots(3, 3, figsize = (9, 9))

    # Plot trajectory lengths
    traj_lens = trajs.groupby('traj_idx').size()
    bin_seq = np.arange(11)
    histo, bins = np.histogram(traj_lens, bins = bin_seq)
    bin_centers = bin_seq[:-1] + 0.5
    ax[0,0].bar(
        bin_seq[:-1],
        histo,
        width = 0.8,
        edgecolor = 'k',
        linewidth = 2,
        color = '#C2C2C2',
    )
    ax[0,0].set_xlabel('Trajectory length')
    ax[0,0].set_ylabel('Count')

    # Plot the distribution of trajectory lengths
    first_pos = trajs.groupby('traj_idx').first()
    first_pos['traj_len'] = trajs.groupby('traj_idx').size()
    visualize.show_locs(
        first_pos.sort_values(by = 'traj_len'),
        ax = ax[0,1],
        color_attrib = 'traj_len',
        max_value = 7,
        min_value = 0,
        cmap = 'Blues',
        ylim = ((0, metadata['N'])),
        xlim = ((0, metadata['M'])),
    )
    ax[0,1].set_title('Trajectory length')
    ax[0,1].set_aspect('equal')

    # Plot only nonzero trajectory lengths
    first_pos_long = first_pos[first_pos['traj_len'] > 1]
    visualize.show_locs(
        first_pos_long.sort_values(by = 'traj_len'),
        ax = ax[0,2],
        color_attrib = 'traj_len',
        max_value = 7,
        min_value = 0,
        cmap = 'Blues',
        ylim = ((0, metadata['N'])),
        xlim = ((0, metadata['M'])),
    )
    ax[0,2].set_title('Trajectory length (> 1 disp)')
    ax[0,2].set_aspect('equal')

    # Plot the subproblem sizes
    visualize.attrib_dist(
        trajs,
        attrib = 'subproblem_n_traj',
        ax = ax[1,0],
        max_value = 10,
        bin_size = 1,
        label = '# trajs per subproblem'
    )
    visualize.attrib_dist(
        trajs,
        attrib = 'subproblem_n_loc',
        ax = ax[1,1],
        max_value = 10,
        bin_size = 1,
        label = '# locs per subproblem'
    )

    # Show the subproblem sizes as a function of spatial
    # position
    new_trajs = trajs.assign(mean_subproblem_size = trajs[['subproblem_n_loc', 'subproblem_n_traj']].mean(axis = 1))
    new_trajs = new_trajs.sort_values(by = 'mean_subproblem_size')
    visualize.show_locs(
        new_trajs,
        ax = ax[1,2],
        color_attrib = 'mean_subproblem_size',
        cmap = 'BuGn',
        max_value = 5,
        min_value = 0,
        ylim = ((0, metadata['N'])),
        xlim = ((0, metadata['M'])),
    )
    ax[1,2].set_title('Mean subproblem size')
    ax[1,2].set_aspect('equal')

    # Show localization density, for comparison
    visualize.loc_density(
        trajs,
        metadata,
        ax = ax[2,0],
        upsampling_factor = 10,
        kernel_width = 0.1,
        y_col = y_col,
        x_col = x_col,
        save_plot = True,
    )
    ax[2,0].set_xticks([])
    ax[2,0].set_yticks([])
    ax[2,0].set_title('Localization density')

    # Show the MSD of each trajectory as a function of position
    traj_df = trajs.copy()
    traj_df['delta_y'] = trajs.groupby('traj_idx')[y_col].diff()
    traj_df['delta_x'] = trajs.groupby('traj_idx')[x_col].diff()
    traj_df = traj_df[~pd.isnull(traj_df['delta_y'])]
    traj_df['r2'] = (traj_df[['delta_y', 'delta_x']]**2).sum(axis=1)
    first_pos = traj_df.groupby('traj_idx').first().reset_index()
    first_pos['msd'] = traj_df.groupby('traj_idx')['r2'].mean()

    visualize.show_locs(
        first_pos.sort_values(by = 'r2', ascending = False),
        ax = ax[2,1],
        color_attrib = 'r2',
        cmap = 'RdPu_r',
        max_value = 0.3,
        min_value = 0.0,
        ylim = ((0, metadata['N'])),
        xlim = ((0, metadata['M'])),
    )
    ax[2,1].set_title('Squared displacement')
    ax[2,1].set_aspect('equal')

    # Set the last one to zero until something else comes up
    visualize.hide_axis(ax[2,2])

    plt.tight_layout()
    if type(out_png) == type(''):
        plt.savefig(out_png, dpi = 600)
        plt.close()
        if sys.platform == 'darwin':
            os.system('open %s' % out_png)
    else:
        plt.show(); plt.close()

def mask_qc(
    mask_objects,
    trajs,
    metadata,
    out_png = None,
    verbose = True,
    y_col = 'y_pixels',
    x_col = 'x_pixels',
    cap = 2000,
):
    '''
    args
        mask_object             :   interpolate_mask.LinearMaskInterpolator
                                        or interpolate_mask.SplineMaskInterpolator
        trajs                   :   pandas.DataFrame, localization data
        out_png                 :   str, out file if desired

    '''
    if type(mask_objects) != type([]):
        mask_objects = [mask_objects]

    fig, ax = plt.subplots(2, 3, figsize = (9, 6))

    N = metadata['N']
    M = metadata['M']
    y_field, x_field = np.mgrid[:N, :M]

    yx = np.asarray([y_field.flatten(), x_field.flatten()]).T 

    # Plot each of the defined mask outlines
    for mask_object in mask_objects:
        colors_0 = sns.color_palette('viridis', len(mask_object.mask_frames))
        for interval_idx, interval in enumerate(mask_object.mask_frames):
            points = mask_object.interpolate(interval)
            ax[0,0].plot(
                points[:,1],
                points[:,0],
                color = colors_0[interval_idx],
                linewidth = 2,
            )
    ax[0,0].set_ylim((0, N))
    ax[0,0].set_xlim((0, M))
    ax[0,0].set_title('Mask definition')

    # Show some interpolation
    colors_1 = sns.color_palette('viridis', 200)
    for mask_object in mask_objects:
        frames = np.linspace(mask_object.min_frame, mask_object.max_frame, 200)
        for frame_idx, frame in enumerate(frames):
            points = mask_object.interpolate(frame)
            ax[0,1].plot(
                points[:,1],
                points[:,0],
                color = colors_1[frame_idx],
                linewidth = 2,
            )
    ax[0,1].set_ylim((0, N))
    ax[0,1].set_xlim((0, M))
    ax[0,1].set_title('Edge interpolation')

    # Show the localization density, for reference
    visualize.loc_density(
        trajs,
        metadata,
        ax = ax[0,2],
        upsampling_factor = 20,
        kernel_width = 0.1,
        save_plot = True
    )
    ax[0,2].set_title('Localization density')

    # Show the original trajectories
    if verbose: print('assigning trajectories to each mask...')
    in_mask = mask.mask_membership(mask_objects, trajs)
    in_mask_all = in_mask.all(axis = 1)
    in_mask_any = in_mask.any(axis = 1)

    visualize.show_locs(
        trajs,
        ax = ax[1,0],
        cmap = 'viridis',
        color_attrib = 'I0',
        max_value = 300,
        min_value = 0,
    )
    visualize.show_locs(
        trajs[~in_mask_any],
        ax = ax[1,2],
        cmap = 'viridis',
        color_attrib = 'I0',
        max_value = 300,
        min_value = 0,
    )
    visualize.show_locs(
        trajs[in_mask_any],
        ax = ax[1,2],
        cmap = 'inferno',
        color_attrib = 'I0',
        max_value = 300,
        min_value = 0,
    )
    visualize.show_locs(
        trajs[~in_mask_any],
        ax = ax[1,1],
        cmap = 'viridis',
        color_attrib = "I0",
    )
    ax[1,0].set_title('All trajectories')
    ax[1,1].set_title('Not in mask')
    ax[1,2].set_title('Mask assignments')

    plt.tight_layout()
    if type(out_png) == type(''):
        plt.savefig(out_png, dpi = 600)
        plt.close()
        if sys.platform == 'darwin':
            os.system('open %s' % out_png)
    else:
        plt.show(); plt.close()



def calculate_psf(
    image_file,
    locs,
    window_size = 11,
    frame_col = 'frame_idx',
    y_col = 'y_pixels',
    x_col = 'x_pixels',
    distance_from_center = 0.5,
    plot = False,
):
    '''
    Calculate the mean PSF for a set of localizations.

    args
        image_file              :   str, either TIF or ND2

        locs                    :   pandas.DataFrame, localizations

        window_size             :   int, the size of the PSF image window

        frame_col, y_col, x_col :   str, column names in *locs*

        distance_from_center    :   float, the maximum tolerated distance
                                        from the center of the pixel along
                                        either the y or x directions. If 0.5,
                                        then all localizations are used (even
                                        if they fall on a corner). If 0.25, then
                                        only pixels in the range [0.25, 0.75]
                                        on each pixel would be used, etc.

    returns
        2D ndarray of shape (window_size, window_size), the averaged
            PSF image 

    '''
    in_radius = (((locs[y_col]%1.0)-0.5)**2 + ((locs[x_col]%1.0)-0.5)**2) <= (distance_from_center**2)
    select_locs = locs[in_radius]

    reader = spazio.ImageFileReader(image_file)
    psf_image = np.zeros((window_size, window_size), dtype = 'float64')
    n_psfs = 0
    half_w = window_size // 2

    for frame_idx, frame_locs in tqdm(select_locs.groupby(frame_col)):
        image = reader.get_frame(frame_idx)
        for loc_idx in frame_locs.index:
            y, x = np.asarray(frame_locs.loc[loc_idx, [y_col, x_col]]).astype('uint16')
            try:
                subim = image[
                    y-half_w : y+half_w+1,
                    x-half_w : x+half_w+1,
                ]
                psf_image = psf_image + subim
                n_psfs += 1
            except ValueError: #edge loc 
                pass 

    psf_image = psf_image / n_psfs 
    reader.close()
    if plot: plt.imshow(psf_image); plt.show(); plt.close()
    return psf_image 

def pixel_localization_density(
    locs,
    bin_size = 0.05,
    y_col = 'y_pixels',
    x_col = 'x_pixels',
    plot = True,
):
    '''
    Calculate the pixel localization density as a function 
    of position along each camera pixel, to check for pixel
    edge bias.

    args
        locs        :   pandas.DataFrame
        bin_size    :   float, the fraction of a pixel at   
                        which to make bins
        y_col       :   str, colunn in *locs* with y-coordinate
        x_col       :   str, column in *locs* with x-coordinate

    returns
        2D ndarray of shape (n_bins, n_bins), the 2D histogram
            of localization counts in each spatial bin

    '''
    bin_seq = np.arange(0, 1.0+bin_size, bin_size)
    n_bins = len(bin_seq) - 1

    histo, xedges, yedges = np.histogram2d(
        locs[y_col] % 1.0,
        locs[x_col] % 1.0,
        bins = bin_seq,
    )
    y_field, x_field = np.mgrid[:n_bins, :n_bins] / n_bins + bin_size/2

    if plot:
        y_field, x_field = np.mgrid[:n_bins, :n_bins] / n_bins + bin_size/2
        fig = plt.figure(figsize = (6, 4))
        ax = fig.add_subplot(111, projection = '3d')
        ax.plot_surface(
            y_field,
            x_field,
            histo,
            cmap = 'inferno',
            linewidth = 0.1,
        )
        ax.set_zlabel('Localization count')
        ax.set_xlabel('x mod 1 (pixels)')
        ax.set_ylabel('y mod 1 (pixels)')
        plt.show(); plt.close()

    return histo, y_field, x_field 

def loc_density_time(
    locs,
    metadata,
    ax = None,
    frame_interval = 5,
    frame_col = 'frame_idx',
):
    new_locs = locs.assign(frame_group = locs[frame_col] // frame_interval)
    locs_per_frame = new_locs.groupby('frame_group').size() / frame_interval 

    if ax == None:
        fig, ax = plt.subplots(figsize = (3, 3))
        finish_plot = True
    else:
        finish_plot = False 

    interval_times = np.arange(len(locs_per_frame)) * frame_interval
    ax.plot(
        interval_times,
        locs_per_frame,
        color = 'k',
        linestyle = '-',
        linewidth = 2,
    )
    ax.set_xlabel('Frame index')
    ax.set_ylabel('Localizations per frame')

    if finish_plot:
        plt.tight_layout()
        plt.show(); plt.close()

    return locs_per_frame, interval_times


def fast_sample_gaussian(
    image_size,
    n_particles,
    sigma = 0.9,
    I0 = 300,
    bg = 5,
    subwindow_size = 15
):
    from scipy.special import erf
    P = np.zeros((image_size, image_size), dtype = 'float64')
    y_field, x_field = np.mgrid[:image_size, :image_size]
    half_w = subwindow_size // 2
    y_subfield, x_subfield = np.mgrid[:subwindow_size, :subwindow_size] - half_w
    positions = np.random.uniform(
        10,
        image_size - 10,
        size = (n_particles, 2)
    )
    factor = sigma * np.sqrt(2)
    for p_idx in range(n_particles):
        pos_vec = positions[p_idx, :]
        pos_int = pos_vec.astype('uint16')
        pos_float = pos_vec % 1
        sub_P = I0 * \
            (erf((y_subfield - pos_float[0] + 1) / factor) \
                - erf((y_subfield - pos_float[0]) / factor)) * \
            (erf((x_subfield - pos_float[1] + 1) / factor) \
                - erf((x_subfield - pos_float[1]) / factor)) / 4
        P[pos_int[0]-half_w : pos_int[0]+half_w+1, \
            pos_int[1]-half_w : pos_int[1]+half_w+1] += sub_P
    P += bg
    result = np.random.poisson(P)
    return result, positions 

    

