'''
mask.py

'''
import numpy as np 
from nd2reader import ND2Reader 
from scipy import io as sio 
import os
import sys
import matplotlib.pyplot as plt 
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector
import pandas as pd 
import tifffile 
import seaborn as sns
sns.set(style = 'ticks')

def onselect(verts):
    with open('_verts.txt', 'w') as o:
        for i, j in verts:
            o.write('%f\t%f\n' % (i, j))

def draw_mask_prompt(
    image_array,
    white_out_areas = None,
    user_prompt = None,
    vmax_mod = 1.0,
):
    '''
    Prompt the user to draw a mask on an image.

    args
        image_array     :   2D ndarray, the image
        white_out_areas :   2D ndarray of same shape as
                                *image_array*, areas to
                                white out in the image
        user_prompt     :   str, prompt for user
        vmax_mod        :   float, for intensity scaling

    returns
        binary 2D ndarray, the mask drawn by the user, and
        matplotlib.Path, the edge path of that mask

    '''
    N, M = image_array.shape 
    fig, ax = plt.subplots(figsize = (10, 10))
    plot_image = image_array.copy()
    if type(white_out_areas) == type(np.array([])):
        plot_image[white_out_areas > 0] = plot_image.max()
    ax.imshow(
        plot_image,
        cmap='gray',
        vmax = image_array.max() * vmax_mod
    )
    if type(user_prompt) == type(''):
        ax.annotate(
            user_prompt,
            (10, 10),
            color = 'w',
            fontsize = 20
        )
    lasso = LassoSelector(ax, onselect)
    plt.show(); plt.close()
    verts = np.asarray(pd.read_csv('_verts.txt', sep = '\t', header = None))
    path = Path(verts, closed = True)
    y, x = np.mgrid[:N, :M]
    points = np.transpose((x.ravel(), y.ravel()))
    mask = path.contains_points(points).reshape((N, M)).astype('uint16')
    return mask, path 

def draw_n_masks_prompt(
    image_array,
    n_masks,
    white_out_areas = None,
    user_prompt = None,
    vmax_mod = 1.0,
):
    if type(white_out_areas) != type(image_array):
        white_out_areas = np.zeros(image_array.shape, dtype = 'uint16')
    result = np.zeros(image_array.shape, dtype = 'uint16')
    result_paths = []
    c_idx = 0
    while c_idx < n_masks:
        mask, path = draw_mask_prompt(
            image_array,
            white_out_areas = white_out_areas,
            user_prompt = '%s\nMask %d/%d' % (user_prompt, c_idx+1, n_masks),
            vmax_mod = vmax_mod,
        )
        result_paths.append(path)
        result = result + mask * (c_idx + 1)
        white_out_areas = ((white_out_areas > 0) | (mask > 0)).astype('uint16')
        c_idx += 1
    return result, result_paths

def assign_trajectories_to_mask(
    mask_path,
    trajs,
    pixel_size_um_mask = 0.16,
):
    for traj_idx, traj in enumerate(trajs):
        positions = traj[0] / pixels_per_um_mask 
        in_mask = mask_path.contains_points(positions)
        raise NotImplementedError




# Implementation separation



def single_mask_trajectories(
    trajectory_file,
    mask_template_file,
    shape_y,
    shape_x,
    trajectory_units = 'um',
    take_col = 'one_loc_in_mask',
    make_summary_plot = False, 
    n_masks = 1
):
    # Figure out what kind of mask template file is provided

    # If ND2, make a maximum intensity projection
    if '.nd2' in mask_template_file:
        nd2_obj = ND2Reader(mask_template_file)
        n_frames = get_n_frames_nd2(nd2_obj)
        if n_frames > 1000:
            n_frames = 1000
        mask_template = nd2_obj.get_frame_2D(t = 0)
        for frame_idx in range(1, n_frames):
            next_frame = nd2_obj.get_frame_2D(t = frame_idx)
            replacements = np.nonzero(next_frame > mask_template)
            mask_template[replacements] = next_frame[replacements]
        target_mask_file = mask_template_file.replace('.nd2', '_mask_template.tif')
        tifffile.imsave(target_mask_file, mask_template.astype('uint16'))
        pixels_per_um_mask = 1 / 0.16
        reverse_xy = False 
    # If TIF, take the first frame
    elif '.tif' in mask_template_file:
        target_mask_file = mask_template_file
        pixels_per_um_mask = 1 / 0.16
        reverse_xy = False 
    # If locs.txt, make a density projection
    elif 'locs.txt' in mask_template_file:
        pixels_per_um_mask = 50
        f = pd.read_csv(mask_template_file, sep = '\t')
        if 'y_coord_pixels' in f.columns:
            mask_template = loc_density(
                f,
                (shape_y, shape_x),
                pixels_per_um_result = pixels_per_um_mask,
                convert_to_um = True,
                y_col = 'y_coord_pixels',
                x_col = 'x_coord_pixels',
            )
        elif 'y_coord_um' in f.columns:
            mask_template = loc_density(
                f,
                (shape_y, shape_x),
                pixels_per_um_result = pixels_per_um_mask,
                convert_to_um = False,
                y_col = 'y_coord_um',
                x_col = 'x_coord_um',
            )
        else:
            print('Localization CSV must contain either a `y_coord_pixels` or `y_coord_um` column')
            exit(1)
        target_mask_file = mask_template_file.replace('locs.txt', 'mask_template.tif')
        tifffile.imsave(target_mask_file, mask_template)
        reverse_xy = True  
    # If *Tracked.mat, overlay all of the individual localizations
    elif 'Tracked.mat' in mask_template_file:
        pixels_per_um_mask = 50
        f = extract_localizations_from_mat(mask_template_file)
        mask_template = loc_density(
            f,
            (shape_y, shape_x),
            pixels_per_um_result = pixels_per_um_mask
        )
        target_mask_file = mask_template_file.replace('Tracked.mat', 'mask_template.tif')
        tifffile.imsave(target_mask_file, mask_template)
        reverse_xy = True  
    else:
        print('Input %s not recognized' % mask_template_file)
        exit(1)

    mask, mask_path = draw_mask(
        target_mask_file,
        out_tif = '%s_trajectory_mask.tif' % \
            trajectory_file.replace('_Tracked.mat', ''),
        n_masks = n_masks,
    )

    assignments = assign_trajectories_to_mask(
        trajectory_file,
        mask_path,
        output_mat_file = '%s_%s_Tracked.mat' % (trajectory_file.replace('_Tracked.mat', ''), take_col),
        take_col = take_col,
        trajectory_units = trajectory_units,
        pixels_per_um_mask = pixels_per_um_mask,
        reverse_xy = reverse_xy 
    )
    print('%d/%d trajectories assigned to mask' % (assignments[take_col].sum(), len(assignments)))

    if make_summary_plot:
        fig, ax = plt.subplots(2, 3, figsize = (9, 6))
        colors_0 = sns.color_palette('viridis', 100)
        colors_1 = sns.color_palette('inferno', 100)

        target = find_xy_image(tifffile.imread(target_mask_file))
        traj_locs = extract_localizations_from_mat(trajectory_file)
        density = loc_density(
            traj_locs,
            (shape_y, shape_x),
        )
        trajs = sio.loadmat(trajectory_file)['trackedPar']
        if trajs.shape[0] == 1:
            trajs = trajs[0,:]

        ax[0,0].imshow(target[::-1,:], cmap = 'gray', vmax = target.mean() + 3.5 * target.std())
        ax[0,1].imshow(mask[::-1,:], cmap = 'gray')
        ax[0,2].imshow(density[::-1,:], cmap = 'gray', vmax = density.mean() + 3.5 * density.std())

        traj_to_plot = len(trajs)
        if traj_to_plot > 7000:
            traj_to_plot = 7000

        for traj_idx in range(traj_to_plot):
            traj = trajs[traj_idx]
            ax[1,0].plot(
                traj[0][:,1],
                traj[0][:,0],
                marker = '.',
                markersize = 2,
                linestyle = '-',
                linewidth = 0.5,
                color = colors_0[traj_idx % len(colors_0)],
            )
            if assignments.loc[traj_idx, take_col] == 1:
                ax[1,1].plot(
                    traj[0][:,1],
                    traj[0][:,0],
                    marker = '.',
                    markersize = 2,
                    linestyle = '-',
                    linewidth = 0.5,
                    color = colors_1[traj_idx % len(colors_1)],
                )
            else:
                ax[1,1].plot(
                    traj[0][:,1],
                    traj[0][:,0],
                    marker = '.',
                    markersize = 2,
                    linestyle = '-',
                    linewidth = 0.5,
                    color = colors_0[traj_idx % len(colors_0)],
                )
            sys.stdout.write('\tplotted %d/%d trajectories...\r' % (traj_idx + 1, len(trajs)))
            sys.stdout.flush()
        print('')
        ax[1,0].set_aspect('equal')
        ax[1,1].set_aspect('equal')

        ax[0,0].set_title('Mask template')
        ax[0,1].set_title('Mask')
        ax[0,2].set_title('Localization density')
        ax[1,0].set_title('All trajectories')
        ax[1,1].set_title('Masking')

        wrapup('%s_masking_summary_plot.png' % trajectory_file.replace('_Tracked.mat', ''))

    return assignments 

def loc_density(
    loc_dataframe,
    shape,
    y_col = 'y_coord_um',
    x_col = 'x_coord_um',
    convert_to_um = False,
    pixels_per_um_result = 50.0
):
    if convert_to_um:
        loc_dataframe['y_coord_um'] = loc_dataframe[y_col] * 0.16
        loc_dataframe['x_coord_um'] = loc_dataframe[x_col] * 0.16
    else:
        loc_dataframe['y_coord_um'] = loc_dataframe[y_col]
        loc_dataframe['x_coord_um'] = loc_dataframe[x_col]

    loc_dataframe[['y_coord_density', 'x_coord_density']] = \
        loc_dataframe[['y_coord_um', 'x_coord_um']] * pixels_per_um_result 

    out_shape_y = int(shape[0] * 0.16 * pixels_per_um_result)
    out_shape_x = int(shape[1] * 0.16 * pixels_per_um_result)

    kernel_size = 2 * int(pixels_per_um_result * 0.2) + 1
    density = np.zeros((out_shape_y, out_shape_x), dtype = 'float')

    kernel_y, kernel_x = np.mgrid[:kernel_size, :kernel_size]
    middle = int(kernel_size // 2)
    radius = kernel_size / 4
    r2 = radius ** 2
    kernel = np.exp(-((kernel_x - middle) ** 2 + (kernel_y - middle) ** 2) / (2 * r2))

    for loc_idx in loc_dataframe.index:
        y = int(round(loc_dataframe.loc[loc_idx, 'y_coord_density'], 0))
        x = int(round(loc_dataframe.loc[loc_idx, 'x_coord_density'], 0))
        try:
            density[y - middle : y + middle + 1, x - middle : x + middle + 1] += kernel 
        except ValueError:  #edge detection
            continue 

        sys.stdout.write('\tCompiled %d localizations...\r' % loc_idx)
        sys.stdout.flush()
    print('')

    return density 

def extract_localizations_from_mat(
    trajectory_mat_file,
    delimiter = '\t'
):
    trajs = sio.loadmat(trajectory_mat_file)['trackedPar']
    if trajs.shape[0] == 1:
        trajs = trajs[0,:]
    N_trajs = len(trajs)
    N_locs = sum([trajs[traj_idx][0].shape[0] for traj_idx in range(N_trajs)])
    loc_idx = 0
    result = np.zeros((N_locs, 6), dtype = 'float')
    for traj_idx in range(N_trajs):
        for local_idx in range(trajs[traj_idx][0].shape[0]):
            result[loc_idx, 0] = local_idx
            result[loc_idx, 1] = traj_idx
            result[loc_idx, 2] = trajs[traj_idx][1].flatten()[local_idx] - 1
            result[loc_idx, 3] = trajs[traj_idx][2].flatten()[local_idx]
            result[loc_idx, 4:] = trajs[traj_idx][0][local_idx, :]
            loc_idx += 1
    df = pd.DataFrame(result)
    df.columns = ['local_idx', 'traj_idx', 'frame_idx', 'time', 'y_coord_um', 'x_coord_um']
    df['local_idx'] = df['local_idx'].astype('uint16')
    df['traj_idx'] = df['traj_idx'].astype('uint16')
    df['frame_idx'] = df['frame_idx'].astype('uint16')

    return df 

def onselect(verts):
    with open('_verts.txt', 'w') as o:
        for i, j in verts:
            o.write('%f\t%f\n' % (i, j))

def draw_mask(
    tif_file,
    out_tif = None,
    n_masks = 1,
):
    image = tifffile.imread(tif_file)
    if len(image.shape) > 2:
        image = find_xy_image(image)
    if type(out_tif) == type(None):
        out_tif = tif_file.replace('.tif', '_single_mask.tif')

    y, x = np.mgrid[:image.shape[0], :image.shape[1]]
    points = np.transpose((x.ravel(), y.ravel()))
    mask = np.zeros(image.shape, dtype = 'uint16')

    for mask_idx in range(n_masks):
        fig, ax = plt.subplots(figsize = (9, 9))
        plot_image = image.copy()
        plot_image[np.where(mask)] = image.max()

        ax.imshow(plot_image, cmap = 'gray', vmax = image.mean() + 3 * image.std())
        lasso = LassoSelector(ax, onselect)
        plt.show(); plt.close()
        verts = np.asarray(pd.read_csv('_verts.txt', sep = '\t', header = None))
        path = Path(verts, closed=True)

        new_mask = path.contains_points(points).reshape(image.shape[0], \
            image.shape[1]).astype('uint16')

        mask += new_mask

    tifffile.imsave(out_tif, mask)
    return mask, path 

def wrapup(out_png):
    plt.tight_layout()
    plt.savefig(out_png, dpi = 800)
    plt.close()
    os.system('open %s' % out_png)

def get_n_frames_nd2(nd2_obj):
    frame_idx = 0
    while 1:
        try:
            frame = nd2_obj.get_frame_2D(t = frame_idx)
            frame_idx += 1
        except (KeyError, ValueError) as e2:
            return frame_idx 

def find_xy_image(image):
    '''
    For a multidimensional numpy array, take only the two 
    dimensions with the largest size.

    INPUT
        image       :   np.array with some shape

    RETURNS
        flattened_image, np.array with shape (N_max, M_max)

    '''
    if len(image.shape) == 2:
        return image
    shape = list(image.shape)
    midx = shape.index(min(shape))
    concat = \
        [':' for i in range(midx)] + \
        ['&'] + \
        [':' for i in range(midx + 1, len(shape))]
    idx_string = ','.join(concat)
    for j in range(shape[midx]):
        j_string = idx_string.replace('&', str(j))
        if eval('image[%s]' % j_string).any():
            return find_xy(eval('image[%s]' % j_string))
    return find_xy(eval('image[%s]' % idx_string.replace('&', '0')))

def assign_trajectories_to_mask(
    trajectory_file,
    mask_path,              #matplotlib.path.Path object
    output_csv = None,
    output_mat_file = None,
    take_col = 'one_loc_in_mask',
    trajectory_units = 'um',
    pixels_per_um_mask = 1 / 0.16,
    reverse_xy = True 
):
    trajs = sio.loadmat(trajectory_file)['trackedPar']

    if trajs.shape[0] == 1:
        trajs = trajs[0,:]

    if trajectory_units == 'pixels':
        for traj_idx in range(trajs.shape[0]):
            trajs[traj_idx][0] = trajs[traj_idx][0] * 0.16

    if reverse_xy:
        for traj_idx in range(trajs.shape[0]):
            trajs[traj_idx][0] = trajs[traj_idx][0][:, ::-1]

    n_trajs = trajs.shape[0]
    n_frames = max([trajs[traj_idx][1].max() for traj_idx in range(n_trajs)])
    assignments = np.zeros((n_trajs, 6), dtype = 'uint16')

    for traj_idx, traj in enumerate(trajs):
        positions = traj[0] * pixels_per_um_mask
        in_mask = mask_path.contains_points(positions)
        assignments[traj_idx, 0] = traj_idx
        assignments[traj_idx, 1] = int(in_mask.all())
        assignments[traj_idx, 2] = int(in_mask.any())
        assignments[traj_idx, 3] = int(in_mask[0])
        assignments[traj_idx, 4] = int(in_mask[-1])
        assignments[traj_idx, 5] = int((~in_mask).any() & (in_mask).any())

    result = pd.DataFrame(assignments, columns = [
        'traj_idx',
        'all_locs_in_mask',
        'one_loc_in_mask',
        'start_in_mask',
        'end_in_mask',
        'on_border',
    ])

    if output_csv != None:
        result.to_csv(output_csv, index = False)

    if output_mat_file != None:
        take_trajs = trajs[np.nonzero(result[take_col])]
        out_dict = {'trackedPar' : take_trajs}
        sio.savemat(output_mat_file, out_dict)

    return result 


def binary_mask_to_path(binary_mask):
    edge = mask_edge(binary_mask)
    points = np.asarray(np.nonzero(edge)).T 
    points = sort_points_into_polygon(points, tolerance_radius = 5.0)
    return Path(points, closed = True) 

def mask_edge(mask):
    '''
    Utility function; returns the list of vertices of a binary mask.

    INPUT
        mask    :   numpy.array, binary image mask

    RETURNS
    '''

    mask = mask.astype('bool')
    mask_00 = mask[:-1, :-1]
    mask_10 = mask[1: , :-1]
    mask_01 = mask[:-1, 1: ]
    edge_0 = np.zeros(mask.shape, dtype = 'bool')
    edge_1 = np.zeros(mask.shape, dtype = 'bool')

    edge_0[:-1, :-1] = np.logical_and(
        mask_00,
        ~mask_10
    )
    edge_1[1:, :-1] = np.logical_and(
        ~mask_00,
        mask_10 
    )
    horiz_mask = np.logical_or(edge_0, edge_1)
    edge_0[:, :] = 0
    edge_1[:, :] = 0
    edge_0[:-1, :-1] = np.logical_and(
        mask_00,
        ~mask_01
    )
    edge_1[:-1, 1:] = np.logical_and(
        ~mask_00,
        mask_01 
    )
    vert_mask = np.logical_or(edge_0, edge_1)
    return np.logical_or(horiz_mask, vert_mask)

def sort_points_into_polygon(points, tolerance_radius = 5.0):
    all_indices = np.arange(points.shape[0])
    remaining_pt_idxs = np.ones(points.shape[0], dtype = 'bool')
    result = np.zeros(points.shape[0], dtype = 'uint16')

    result_idx = 0
    current_idx = 0
    result[result_idx] = current_idx 
    remaining_pt_idxs[current_idx] = False 

    while remaining_pt_idxs.any():
        distances = np.sqrt(((points[current_idx, :] - \
            points[remaining_pt_idxs, :])**2).sum(axis = 1))
        within_tolerance = distances <= tolerance_radius
        if ~within_tolerance.any():
            return points[result[:result_idx], :]
        current_idx = all_indices[remaining_pt_idxs][within_tolerance]\
            [np.argmin(distances[within_tolerance])]
        result[result_idx] = current_idx 
        remaining_pt_idxs[current_idx] = False
        result_idx += 1

    return points[result, :]

