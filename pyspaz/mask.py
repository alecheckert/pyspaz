'''
mask.py

'''
import numpy as np 
from nd2reader import ND2Reader 
from scipy import io as sio 
import os
import sys
from tqdm import tqdm 
import matplotlib.pyplot as plt 
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector
import pandas as pd 
import tifffile 
import seaborn as sns
sns.set(style = 'ticks')

from pyspaz import spazio
from pyspaz import visualize

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

def assign_trajectories_to_mask_mat_format(
    mask_path,
    trajs,
    traj_cols,
    mask_col_name = None,
    mask_upsampling_factor = 1,
    pixel_size_um = 0.16,
): 
    '''
    Assign each trajectory to a mask. Assumes that
    trajectory units are in um and mask units are 
    in pixels.

    args
        mask_path               :   matplotlib.Path object, the mask
        trajs                   :   2D ndarray, trajectories
        mask_upsampling_factor  :   float, the factor by which 


    '''
    n_trajs = trajs.shape[0]
    n_cols = trajs.shape[1]
    new_trajs = np.zeros((n_trajs, n_cols+1), dtype = 'float64')

    for traj_idx, traj in enumerate(trajs):

        # Copy the trajectory information to the corresponding
        # columns of the new array
        for col_idx in range(n_cols):
            new_trajs[traj_idx, col_idx] = traj[col_idx]

        # Get the list of coordinates for this trajectory in pixels
        positions = traj[0] * mask_upsampling_factor / pixels_per_um

        # Figure out which positions lie within the mask
        in_mask = mask_path.contains_points(positions)

        # All points of the trajectory lie within the mask
        if in_mask.all():
            new_trajs[traj_idx, -1] = 1

        # Some points of the trajectory lie within the mask
        elif in_mask.any():
            if inside[0] and inside[-1]:
                new_trajs[traj_idx, -1] = 2
            elif inside[0] and ~inside[-1]:
                new_trajs[traj_idx, -1] = 3
            elif ~inside[0] and inside[-1]:
                new_trajs[traj_idx, -1] = 4
            else:
                new_trajs[traj_idx, -1] = 5

        # All points of the trajectory lie outside the mask
        else:
            new_trajs[traj_idx, -1] = 0

    # Update the trajectory column labels
    if mask_col_name == None:
        mask_col_name = 'mask'
    traj_cols.append(mask_col_name)

    return new_trajs, traj_cols

def assign_trajectories_to_mask(
    mask_path,
    trajs,
    mask_col_name,
    y_col = 'y_pixels',
    x_col = 'x_pixels',
):
    '''
    Given a set of trajectories in pandas.DataFrame format, add
    a new column that whether each localization is 
    inside or outside of the mask defined by *mask_path*.

    args
        mask_path                   :   matplotlib.Path
        trajs                       :   pandas.DataFrame
        mask_col_name               :   str, name of mask column in 
                                            the output dataframe
        y_col, x_col                :   str, the names of the columns
                                            in *trajs* containing the 
                                            localization pixel coordinates

    '''
    trajs[mask_col_name] = mask_path.contains_points(trajs[[x_col, y_col]])

def draw_n_masks_on_loc_density(
    locs,
    n_masks,
    metadata,
    out_tif = None,
    upsampling_factor = 10,
    y_col = 'y_pixels',
    x_col = 'x_pixels',
    vmax_mod = 0.5,
):
    '''
    Convenience function to draw a mask directly on 
    the localization density from a tracking movie.

    args
        locs        :   pandas.DataFrame, the localizations
        n_masks     :   int, the number of masks to draw
        metadata    :   dict, experiment metadata
        out_tif     :   str, output file if desired
        upsampling_factor   :   float, the subpixel density
                        at which to reconstruct the mask
        y_col       :   str, column corresponding to 
                            the y position
        x_col       :   str, column corresponding to 
                            the x position

    returns
        (2D ndarray, list of matplotlib.Path),

            the mask image and the 
            mask paths corresponding to each mask

    ''' 
    density = visualize.loc_density(
        locs,
        metadata,
        ax = None, 
        upsampling_factor = upsampling_factor,
        y_col = y_col,
        x_col = x_col,
        save_plot = False,
        kernel_width = 0.1,
        convert_to_um = False,
    )
    masks, mask_paths = draw_n_masks_prompt(
        density,
        n_masks,
        vmax_mod = vmax_mod, 
    )
    if out_tif != None:
        masks_wrapper = np.zeros((1, masks.shape[0], masks.shape[1]), dtype = masks.dtype)
        masks_wrapper[0,:,:] = masks 
        tifffile.imsave(out_tif, masks_wrapper)

    rescaled_mask_paths = []
    for mask_path in mask_paths:
        rescaled_mask_paths.append(
            rescale_path_pixel_size(
                mask_path, 
                1.0 / upsampling_factor
            )
        )

    return masks, rescaled_mask_paths
    

def mask_to_mask_path(
    mask_image,
    tolerance_radius = 10,
    plot = False,
    mask_upsampling_factor = 1.0
):
    '''
    Convert a binary mask image to a mask path, which can 
    be used to quickly determine whether points fall inside
    the mask.

    args
        mask_image              :   2D ndarray
        tolerance_radius        :   float, search radius for mask point
                                        reconnection
        plot                    :   bool, show the result
        mask_upsampling_factor  :   float, the number of mask pixels
                                        per real pixel

    returns
        matplotlib.Path, the mask Path object in terms of 
            camera pixels

    '''
    mask_path = Path(mask_to_oriented_edge(
        mask_image,
        mask_upsampling_factor = mask_upsampling_factor,
        tolerance_radius = tolerance_radius,
    ), closed = True)

    if plot:
        y_field, x_field = np.mgrid[:mask_image.shape[0], :mask_image.shape[1]]
        fig, ax = plt.subplots(figsize = (6, 6))
        points = np.asarray([y_field.flatten(), x_field.flatten()]).T 
        inside = mask_path.contains_points(points)
        ax.imshow(mask_image, cmap = 'gray')
        ax.plot(points[inside,1], points[inside,0], color = 'r', markersize = 1, marker = '.', linestyle = '')
        ax.plot(points[~inside,1], points[~inside,0], color = 'b', markersize = 1, marker = '.', linestyle = '')
        plt.show(); plt.close()

    return mask_path 

# 
# Utilities
#
def mask_to_oriented_edge(
    mask_image,
    mask_upsampling_factor = 1.0,
    tolerance_radius = 10
):
    '''
    Convert a binary mask image into a set of edge points
    that walk around the edge of the mask.

    args
        mask_image      :   2D binary ndarray
        tolerance_radius:   float, search radius for mask
                                edge point reconnection

    returns
        2D ndarray of shape (n_points, 2), the edge points
            of the mask

    '''
    return sort_points_into_polygon(
        np.asarray(mask_edge(mask_image).nonzero()).T / mask_upsampling_factor,
        tolerance_radius = tolerance_radius,
    )

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

def sort_points_into_polygon(
    points,
    tolerance_radius = 3
):
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

def rescale_path_pixel_size(
    matplotlib_path,
    rescaling_factor,
):
    return Path(matplotlib_path.vertices * rescaling_factor, closed = True)

def mask_membership(
    mask_interpolators,
    locs,
    y_col = 'y_pixels',
    x_col = 'x_pixels',
):
    '''
    Determine whether each localization lies within 
    a mask.

    args
        mask_interpolators      :   list of LinearMaskInterpolator or 
                                        SplineMaskInterpolator objects

        locs                    :   pandas.DataFrame

    returns
        2D ndarray of shape (n_locs, n_masks), bool
        
    '''
    n_masks = len(mask_interpolators)
    n_locs = len(locs)
    in_mask = np.zeros((n_masks, n_locs), dtype = 'bool')
    for frame_idx, frame_locs in tqdm(locs.groupby('frame_idx')):
        in_mask[:, frame_locs.index] = np.asarray([mask_interpolators[i].contains_points(frame_locs[[y_col, x_col]], \
            frame_idx) for i in range(n_masks)])
    return in_mask.T 




