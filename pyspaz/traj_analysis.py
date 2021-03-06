'''
traj_analysis.py
'''
# Numerical tools
import numpy as np 
from scipy.optimize import curve_fit 
from scipy.special import erf, gamma, gammainc, gammaincc
from scipy.integrate import quad 

# Dataframes
import pandas as pd 

# Reading *Tracked.mat format trajectories
from scipy import io as sio 

# I/O
import os
import sys
import tifffile

# File-finding
from glob import glob 

# Plotting
import matplotlib.pyplot as plt 

# Hard copies
from copy import copy 

# pyspaz functions
from pyspaz import spazio
from pyspaz import mask
from pyspaz import utils 
from pyspaz import visualize 

# Progress bar
from tqdm import tqdm 

def add_locs_per_frame(locs, frame_col = 'frame_idx'):
    '''
    Add a column to a pandas.DataFrame that shows the
    number of localizations in each frame.

    args
        locs        :   pandas.DataFrame, localizations / trajectories
        frame_col   :   str, column in *trajs* with the frame index

    returns
        pandas.DataFrame with the new 'locs_per_frame' column

    '''
    if 'locs_per_frame' in locs.columns:
        locs = locs.drop('locs_per_frame', axis = 1)

    return locs.join(locs.groupby(frame_col).size().rename('locs_per_frame'), on=frame_col)

def add_trajectory_length(trajs, traj_col='traj_idx'):
    """
    Calculate trajectory length for each trajectory in a 
    dataframe and add as a column to the dataframe.

    args
    ----
        trajs :  pandas.DataFrame
        traj_col :  str

    returns
    -------
        pandas.DataFrame

    """
    if 'traj_len' in trajs.columns:
        trajs = trajs.drop('traj_len', axis = 1)

    return trajs.join(
        trajs.groupby(traj_col).size().rename('traj_len'), on = traj_col
    )

def truncate_trajs(trajs, max_traj_len, remove_unassigned=True):
    """
    Truncate trajectories up to a specific number of 
    localizations.

    args
    ----
        trajs :  pandas.DataFrame
        max_traj_len :  int
        remove_unassigned :  bool, remove localizations
            with traj_idx 0 (unassigned to any trajectory)

    returns
    -------
        pandas.DataFrame, truncated trajectories

    """
    _t = trajs.copy()
    if remove_unassigned: _t = _t[_t['traj_idx'] > 0]
    _t['one'] = 1
    _t['loc_idx_by_traj'] = _t.groupby('traj_idx')['one'].cumsum()
    _t = _t[_t['loc_idx_by_traj'] <= max_traj_len]
    return _t.drop('one', axis = 1)

def concat_trajs(*trajs, traj_col='traj_idx'):
    """
    Concatenate a set of localization dataframes while
    changing the `traj_idx` index so that each trajectory
    has a unique index.

    args
    ----
        *trajs :  set of pandas.DataFrame, output of 
            `track.track_locs` or similar

    returns
    -------
        pandas.DataFrame, the concatenated trajectoreis

    """
    n = len(trajs)

    # Iterative concatenate each dataframe to the first
    # while incrementing the trajectory index
    out = trajs[0].assign(dataframe_idx=0)
    c_idx = out[traj_col].max() + 1

    for i in range(1, n):

        # Get the next set of trajectories and keep track of the
        # origin dataframe
        new = trajs[i].assign(dataframe_idx=i)

        # Trajectory indices of -1 indicate a reconnection error;
        # leave these untouched
        new.loc[new[traj_col] >= 0, traj_col] += c_idx

        # Increment the total number of trajectories
        c_idx = new[traj_col].max() + 1

        # Concatenate trajectories
        out = pd.concat([out, new], ignore_index=True, sort=False)

    return out

def displacement_df(loc_df, compute_angles=True):
    """
    Convert a dataframe of localizations into a dataframe of 
    displacements. This is similar to the input dataframe, 
    except it contains the additional columns `y_delta` and
    `x_delta` that give the direction of motion over the 
    NEXT frame interval.

    Because it has no next frame interval, the last loc
    of each trajectory is dropped.

    Parameters
    ----------
        loc_df :  pandas.DataFrame, for instance the output
            of track.track_locs

    Returns
    -------
        pandas.DataFrame

    """
    # Work with a copy of the original dataframe as we 
    # transform it into a displacement dataframe
    disps = loc_df.copy()

    # Remove all single localizations
    disps = add_trajectory_length(disps)
    disps = disps[disps['traj_len'] > 1]

    # Remove localizations that were not connected by
    # the tracking algorithm (these have index 0)
    disps = disps[disps['traj_idx'] > 0]

    # Reindex
    disps.index = np.arange(len(disps))

    # Take difference
    disps['y_delta'] = -disps.groupby('traj_idx')['y_pixels'].diff(-1)
    disps['x_delta'] = -disps.groupby('traj_idx')['x_pixels'].diff(-1)

    # Exclude the last position of each trajectory
    disps = disps[~pd.isnull(disps['y_delta'])]

    # If desired, also compute the angle of each displacement
    # on the half plane (between 0 and np.pi)
    if compute_angles:
        disps['r'] = np.sqrt(disps['y_delta']**2 + disps['x_delta']**2)
        disps['yr'] = disps['y_delta'] / disps['r']
        disps['xr'] = disps['x_delta'] / disps['r']
        disps['theta'] = np.arccos(disps['xr'])

        # For all angles > pi, invert through the origin
        cond_0 = disps['yr'] < 0.0
        disps.loc[cond_0.index, 'theta'] = np.pi - disps.loc[cond_0.index, 'theta']

    return disps 

def assign_loc_idx_in_traj(trajs):
    """
    For each trajectory, assign a separate localization 
    index to each localization: 0 for the first localization
    in that trajectory, 1 for the second localization in 
    that trajectory, and so on. This is stored in the 
    `loc_idx_in_traj` column.

    Parameters
    ----------
        trajs : pd.DataFrame

    Returns
    -------
        pandas.DataFrame

    """
    trajs['loc_idx'] = np.arange(len(trajs))
    trajs.index = np.arange(len(trajs))
    D = np.asarray(trajs[['loc_idx', 'traj_idx', 'frame_idx']])
    D = D[np.lexsort((D[:,2], D[:,1]))]
    traj_indices = np.unique(D[:,1])
    loc_idx_in_traj = np.zeros(D.shape[0], dtype='int64')
    for traj_idx in tqdm(traj_indices):
        in_traj = D[:,1] == traj_idx 
        loc_idx_in_traj[in_traj] = np.arange(in_traj.sum())
    loc_idx_in_traj = pd.Series(loc_idx_in_traj, index=D[:,0])
    trajs['loc_idx_in_traj'] = loc_idx_in_traj
    return trajs 


def fast_jump_lengths_unbinned(trajs, max_r=5.0, time_delays=4, pixel_size_um=0.16):
    """
    Compute jump lengths for a set of trajectories without
    gaps and return the raw displacements without binning.

    Parameters
    ----------
        trajs :  pandas.DataFrame
        max_r :  float
        time_delays :  int
        pixel_size_um :  float

    Returns
    -------
        dict{ time_delay : 1D ndarray }, the set of
            radial displacements for each frame interval

    """
    # Filter out unassigned localizations
    subtrajs = trajs[trajs["traj_idx"] >= 1]

    # Get trajectory lengths
    subtrajs = add_trajectory_length(subtrajs)

    # Take only trajectories with multiple points
    subtrajs = subtrajs[subtrajs["traj_len"] > 1]

    # Format as ndarray for speed
    subtrajs = np.asarray(subtrajs[["frame_idx", "traj_idx", "y_pixels", "x_pixels"]])

    # Convert pixels to um
    subtrajs[:, 2:] *= pixel_size_um

    # Sort by trajectory -> frame
    subtrajs = subtrajs[np.lexsort((subtrajs[:, 0], subtrajs[:, 1]))]

    # Determine the maximum number of frame gaps present
    diff = subtrajs[1:, :] - subtrajs[:-1, :]
    allowed_gaps = max([0, diff[diff[:, 1] == 0, 0].max() - 1])

    # For each time delay, calculate the histogram of displacements
    out = {t: np.array([]) for t in range(1, time_delays + 1)}
    for t in range(1, time_delays + 1):
        for g in range(2 * (int(allowed_gaps) + 1) + 1):

            # Get displacements
            diff = subtrajs[(t + g) :, :] - subtrajs[: -(t + g), :]

            # Take only displacements originating from the same trajectory
            # and from the correct frame gap
            diff = diff[(diff[:, :2] == np.array([t, 0])).all(1), :]

            # Compute radial displacements
            out[t] = np.concatenate((out[t], np.sqrt((diff[:, 2:] ** 2).sum(1))))

    return out

def fast_jump_lengths(trajs, max_r=5.0, n_bins=5000, time_delays=4, pixel_size_um=0.16):
    """
    Compile histograms of jump lengths for a set of trajectories.

    Parameters
    ----------
        trajs :  pandas.DataFrame, locs with `traj_idx` col
        max_r :  float, max jump length to consider
        n_bins :  int, number of bins between 0 and max_r
        time_delays :  int. Each time delay is compiled as 
                a separate distribution
        pixel_size_um :  float

    Returns
    -------
        (
            2D ndarray of shape (n_bins, time_delays), the 
                histograms;
            1D ndarray of shape (n_bins + 1,), the edges of
                each jump length bin
        )
    
    """
    # Get the raw displacements
    jump_lengths = fast_jump_lengths_unbinned(
        trajs, max_r=max_r, time_delays=time_delays, pixel_size_um=pixel_size_um
    )

    # Make the binning array
    bin_edges = np.linspace(0, max_r, n_bins + 1)

    # Compile histograms
    radial_disp_histograms = np.empty((n_bins, time_delays), dtype="int64")
    for t in range(1, time_delays + 1):
        radial_disp_histograms[:, t - 1] = np.histogram(jump_lengths[t], bins=bin_edges)[0]

    return radial_disp_histograms, bin_edges

def fast_jump_lengths_split(
    trajs,
    max_r=5.0,
    n_bins=5000,
    time_delays=4,
    pixel_size_um=0.16,
):
    """
    Wrapper for fast_jump_lengths() on very large datasets which 
    are difficult to compute in a single binning step. This splits
    the trajectories into groups of 1000 and then compiles the 
    results for each group, accumulating as we go.

    Parameters
    ----------
        dfs :  pandas.DataFrame objects, the trajectories
        kwargs :  to fast_jump_lengths()

    Returns
    -------
        (
            2D ndarray of shape (n_bins, time_delays), the 
                histograms;
            1D ndarray of shape (n_bins + 1,), the edges of
                each jump length bin
        )
    
    """
    # Get rid of unassigned localizations
    trajs = trajs[trajs['traj_idx'] > 0]

    # Add a new column representing the absolute trajectory index
    traj_indices = np.unique(trajs['traj_idx'])
    new_traj_indices = np.arange(len(traj_indices))
    trajs['new_traj_idx'] = trajs['traj_idx'].apply(
        lambda x : new_traj_indices[np.argwhere(traj_indices==x)[0,0]]
    )

    # Split the problem into groups of 1000 trajectories
    trajs['group_idx'] = trajs['traj_idx'] % 1000

    result = np.zeros((n_bins, time_delays), dtype='int64')
    for group_idx, group_trajs in tqdm(trajs.groupby('group_idx')):
        group_result, bin_edges = fast_jump_lengths(
            group_trajs, max_r=max_r, n_bins=n_bins,
            time_delays=time_delays,
            pixel_size_um=pixel_size_um,
        )
        result = result + group_result.copy()

    return result, bin_edges 





# def fast_jump_lengths_split(*dfs, max_r=5.0, n_bins=5000,
#     time_delays=4, pixel_size_um=0.16):
#     """
#     Wrapper for fast_jump_lengths() on very large datasets which 
#     are difficult to compute in a single binning step.

#     Parameters
#     ----------
#         dfs :  pandas.DataFrame objects, the trajectories
#         kwargs :  to fast_jump_lengths()

#     Returns
#     -------
#         (
#             2D ndarray of shape (n_bins, time_delays), the 
#                 histograms;
#             1D ndarray of shape (n_bins + 1,), the edges of
#                 each jump length bin
#         )
    
#     """
#     result = np.zeros((n_bins, time_delays), dtype='float64')
#     for df_idx, df in tqdm(enumerate(dfs)):
#         r_disps, bin_edges = fast_jump_lengths(
#             df, max_r=max_r, n_bins=n_bins,
#             time_delays=time_delays, pixel_size_um=pixel_size_um
#         )
#         result = result + r_disps 
#     return result, bin_edges  



def traj_mask_membership(locs, mask_col, traj_mask_col, rule = 'any', traj_col = 'traj_idx'):
    '''
    Given a dataframe with locs annotated by mask membership,
    classify each trajectory as belonging to the mask or not.

    args
        locs            :   pandas.DataFrame, traj-annotated locs
        mask_col        :   str, column in *locs* indicating loc
                                mask membership
        traj_mask_col   :   str, column in *locs* to add with the 
                                mask membership for trajectories
        rule            :   str, either 'any', 'all', 'first', 'last',
                                or 'both'
        traj_col        :   str, column in *locs* indicating trajectory
                                index

    returns
        pandas.DataFrame with the new column

    '''
    # Check if dataframe does not contain any trajectories that
    # cross the boundary
    if ~(locs.groupby(traj_col)[mask_col].any() & ~locs.groupby(traj_col)[mask_col].all()).any():
        return locs 

    if rule == 'any':
        return locs.join(locs.groupby(traj_col)[mask_col].any().rename(traj_mask_col), on=traj_col)
    elif rule == 'all':
        return locs.join(locs.groupby(traj_col)[mask_col].all().rename(traj_mask_col), on=traj_col)
    elif rule == 'first':
        return locs.join(locs.groupby(traj_col)[mask_col].first().rename(traj_mask_col), on=traj_col)
    elif rule == 'last':
        return locs.join(locs.groupby(traj_col)[mask_col].last().rename(traj_mask_col), on=traj_col)
    elif rule == 'both':
        return locs.join((locs.groupby(traj_col)[mask_col].any() & ~locs.groupby(traj_col)[mask_col].all()).rename(traj_mask_col), on=traj_col)
    else:
        raise RuntimeError('traj_analysis.traj_mask_membership: rule must be any, all, first, last, or both')

def compile_displacements(
    trajs,
    n_gaps = 0,
    n_dt = 4,
    max_disp = 5.0,
    n_bins = 5001,
    pixel_size_um = 0.16,
    start_frame = 0,
    dim = 2,
    condition_col = None,
    max_traj_len = None,
):
    '''
    Convenience function to compile displacements from trajectories
    in one of several formats.

    *trajs* may be:

        - a dataframe of localizations assigned to trajectories
          in the `traj_idx` column

        - a .TXT file with the `traj_idx` column

        - a .trajs or .locs file (TXT with tracking metadata,
          readable by spazio.load_locs)

        - a *Tracked.mat file (MATLAB trajectory format)

        - a list of files of the types above

        - a directory containing *.locs, *.trajs, or *Tracked.mat
          files

    args
        trajs               :   one of the inputs above

        n_gaps              :   int, the number of gaps allowed
                                during tracking

        n_dt                :   int, the number of time delays to 
                                compute for the radial displacement
                                histograms. For instance, if n_dt = 4,
                                then will compute dt = 0.00548 out to
                                dt = 0.00548 * 4 = 0.02192

        max_disp            :   float, the maximum displacement to 
                                compile in um

        n_bins              :   int, the number of distinct radial
                                displacement bins to compute

        pixel_size_um       :   float, size of camera pixels in um

        start_frame         :   int, exclude all trajectories before
                                this

        dim                 :   int, the number of spatial dimensions
                                over which to compute displacements 
                                (1 or 2)

        condition_col       :   str, a boolean column in the input
                                dataframe. Take only trajectories that
                                have this column == True. Only meaningful
                                when the input is dataframe-like (not
                                *Tracked.mat format)

    returns
        (
            2D ndarray of shape (n_bins - 1, n_dt), the radial
                displacement histograms;
            1D ndarray of shape (n_bins,), the edges of the radial
                displacement bins
        )

    '''
    if isinstance(trajs, str):

        # Input is a directory
        if os.path.isdir(trajs):

            # Get the list of files in those directories
            tracked_mat_file_list = glob("%s/*Tracked.mat" % trajs)
            traj_file_list = glob("%s/*.trajs" % trajs)

            # Compile displacements from the *Tracked.mat files in that
            # directory
            if len(tracked_mat_file_list) > 0:
                radial_disp_histograms, bin_edges = compile_displacements_tracked_mat(
                    trajs,
                    n_bins = n_bins,
                    max_displacement = max_disp,
                    n_dt = n_dt,
                    max_traj_len = max_traj_len,
                )
                return radial_disp_histograms, bin_edges

            # Compile displacements from the *.trajs files that 
            # directory
            elif len(traj_file_list) > 0:
                radial_disp_histograms, bin_edges = compile_displacements_directory(
                    traj_file_list,
                    n_bins = n_bins,
                    max_disp = max_disp,
                    time_delays = n_dt,
                    max_traj_len = max_traj_len,
                    traj_col = 'traj_idx',
                    y_col = 'y_pixels',
                    x_col = 'x_pixels',
                    dim = dim,
                    start_frame = start_frame,
                    condition_col = condition_col,
                )
                return radial_disp_histograms, bin_edges

        # Input is a file
        elif os.path.isfile(trajs):

            if 'Tracked.mat' in trajs:
                radial_disp_histograms, bin_edges = compile_displacements_tracked_mat(
                    trajs,
                    n_bins = n_bins,
                    max_displacement = max_disp,
                    n_dt = n_dt,
                    max_traj_len = max_traj_len,
                )
                return radial_disp_histograms, bin_edges 

            elif '.trajs' in trajs:
                locs, metadata = spazio.load_locs(trajs)
                radial_disp_histograms, bin_edges = compile_displacements_trajs(
                    locs,
                    n_bins = n_bins,
                    max_disp = max_disp,
                    time_delays = n_dt,
                    max_traj_len = max_traj_len,
                    traj_col = 'traj_idx',
                    y_col = 'y_pixels',
                    x_col = 'x_pixels',
                    dim = dim,
                    start_frame = start_frame,
                    condition_col = condition_col,
                )
                return radial_disp_histograms, bin_edges

        else:
            raise RuntimeError('traj_analysis.compile_displacements: string input %s not recognized as a directory or file path' % trajs)

    # Input is not a path
    else:

        # Input is a dataframe
        if isinstance(trajs, pd.DataFrame):
            radial_disp_histograms, bin_edges = compile_displacements_trajs(
                trajs,
                n_bins = n_bins,
                max_disp = max_disp,
                time_delays = n_dt,
                max_traj_len = max_traj_len,
                traj_col = 'traj_idx',
                y_col = 'y_pixels',
                x_col = 'x_pixels',
                dim = dim,
                start_frame = start_frame,
                condition_col = condition_col,
            )

        # Does not recognize other inputs
        else:
            raise RuntimeError('traj_analysis.compile_displacements: input must either be a file path or a pandas.DataFrame')




def compile_displacements_trajs(
    trajs,
    n_gaps = 0,
    time_delays = 4,
    traj_col = 'traj_idx',
    frame_col = 'frame_idx',
    y_col = 'y_pixels',
    x_col = 'x_pixels',
    max_disp = 5.0,
    n_bins = 5001,
    pixel_size_um = 0.16,
    start_frame = 0,
    dim = 2,
    condition_col = None,
    max_traj_len = None,
):
    '''
    Compile radial displacements from a given dataframe into
    radial displacement histograms.

    args
        trajs       :   pandas.DataFrame, localizations annotated
                            by trajectory membership

        n_gaps      :   int, the number of gaps allowed during the
                            tracking step. In the current version,
                            it only matters whether this is zero or
                            nonzero.

        time_delays :   int, the number of time delays to consider.
                            Each time delay will compile a separate
                            displacement histogram corresponding 
                            to that delay.

        traj_col    :   str, the column in *trajs* with the trajectory
                            annotation

        frame_col   :   str, the column in *trajs* with the frame index
                            corresponding to each localization

        y_col, x_col:   str, columns in *trajs* corresponding to the    
                            localization position in pixels

        max_disp    :   float, maximum displacement to consider (um)

        n_bins      :   int, the number of bins in the displacement
                            histogram

        pixel_size_um   :   float, the size of the camera pixels in 
                                um

        dim             :    int, dimensionality of displacements,
                                either 1 or 2 

    returns
    (
        2D ndarray of shape (n_bins-1, time_delays), the displacement
            histograms, and

        1D ndarray of shape (n_bins,), the bin
            edges used to construct the histograms
    )

    '''
    # Make sure the user has passed a dataframe that contains the
    # necessary information
    required_cols = [traj_col, frame_col, y_col, x_col]
    if any([c not in trajs.columns for c in required_cols]):
        raise RuntimeError('traj_analysis.compile_displacements: input dataframe must contain the columns %s' % ', '.join(required_cols))

    # Can only compile displacements in 1 or 2 dimensions
    assert dim in [1, 2]

    # Make the sequence of radial displacement bins
    bin_edges = np.linspace(0, max_disp, n_bins)
    histograms = np.zeros((n_bins-1, time_delays), dtype = 'int64')

    # Convert loc coordinates from pixels to um
    df = trajs.assign(y_um = trajs[y_col] * pixel_size_um, x_um = trajs[x_col] * pixel_size_um)

    # Filter out unconnected localizations
    df = df[~pd.isnull(df['traj_idx'])]
    df = df.loc[df['traj_idx'] > 0]

    # Take only displacements after the start frame
    df = df.loc[df['frame_idx'] >= start_frame]

    # If desired, filter out long trajectories
    if not (max_traj_len is None):
        df = df.join(df.groupby('traj_idx').size().rename('traj_len'), on = 'traj_idx')
        df = df[df['traj_len'] <= max_traj_len]

    if not (condition_col is None):
        df = df.loc[df[condition_col]]

    # If there are no gaps in tracking, then pandas.DataFrame.diff()
    # does all the work for us
    if n_gaps == 0:
        for t in tqdm(range(1, time_delays + 1)):
            disps = df.groupby(traj_col)[['y_um', 'x_um']].diff(t)
            disps = disps[~pd.isnull(disps['y_um'])]

            if dim == 2:
                r_disps = np.sqrt((disps ** 2).sum(1))
            elif dim == 1:
                r_disps = np.abs(disps).flatten()

            histo, _edges = np.histogram(r_disps, bins = bin_edges)
            histograms[:, t-1] = histo.copy()

        return histograms, bin_edges

    # Otherwise need to deal with gaps
    else:
        # Collect all of the missing frames and add them as NaNs to
        # the dataframe
        dead_indices = []
        for traj_idx, traj in df.groupby(traj_col):
            min_frame = traj[frame_col].min()
            max_frame = traj[frame_col].max()
            if len(traj) != (max_frame-min_frame+1):
                for frame_idx in [j for j in range(min_frame, max_frame+1) if j not in traj[frame_col].tolist()]:
                    dead_indices.append((traj_idx, frame_idx))

        # Make a new dataframe that includes the gap frames, which will throw
        # NaNs upon pandas.DataFrame.diff()
        gap_df = pd.DataFrame(dead_indices, columns = [traj_col, frame_col])
        trajs_with_gaps = pd.concat([df, gap_df], ignore_index=True, sort=False)
        trajs_with_gaps = trajs_with_gaps.sort_values(by = frame_col)
        trajs_with_gaps.index = np.arange(len(trajs_with_gaps))

        # Compile the radial displacement histograms
        for t in range(1, time_delays + 1):
            disps = trajs_with_gaps.groupby(traj_col)[['y_um', 'x_um']].diff(t)
            disps = disps[~pd.isnull(disps['y_um'])]
            r_disps = np.sqrt((disps**2).sum(1))

            histo, _edges = np.histogram(r_disps, bins = bin_edges)
            histograms[:, t-1] = histo.copy()

        return histograms, bin_edges

def compile_displacements_directory(
    loc_file_list,
    **kwargs,
):
    '''
    Compile radial displacement historams for a set of 
    TXT files containing localization/tracking information.

    args
        loc_file_list   :   list of str, the paths to the TXT
                                files containing localization 
                                info. If a directory str is passed
                                instead, will run on all *.locs
                                files in that directory.

        **kwargs        :   for compile_displacements()
    
    returns
    (
        2D ndarray of shape (n_bins-1, time_delays), the displacement
            histograms, and

        1D ndarray of shape (n_bins,), the bin
            edges used to construct the histograms
    )

    '''
    # If the user passes a directory name for *loc_file_list*,
    # run on all of the *.locs files in that directory
    if type(loc_file_list) == type('') and os.path.isdir(loc_file_list):
        dir_name = copy(loc_file_list)
        loc_file_list = glob("%s/*.locs" % dir_name)

    locs, metadata = spazio.load_locs(loc_file_list[0])
    c_traj_idx = 0
    for i in tqdm(range(1, len(loc_file_list))):
        locs_2, metadata_2 = spazio.load_locs(loc_file_list[i])
        locs_2 = locs_2[locs_2['traj_idx'] > 0]
        n_trajs = locs_2['traj_idx'].max()
        locs_2['traj_idx'] = locs_2['traj_idx'] + c_traj_idx
        c_traj_idx += n_trajs 

        locs = pd.concat([locs, locs_2], ignore_index = True, sort = False)

    if ('condition_col' in kwargs) and not (kwargs['condition_col'] is None):
        locs = traj_mask_membership(locs, kwargs['condition_col'], 'traj_in_mask')
        kwargs['condition_col'] = 'traj_in_mask'
    
    locs = locs.join(locs.groupby('traj_idx').size().rename('traj_len'), on = 'traj_idx')
    locs = locs[locs['traj_len'] > 1]
    locs.index = np.arange(len(locs))

    # Compile radial displacement histograms
    histograms, bin_edges = compile_displacements_trajs(
        locs,
        **kwargs,
    )
    return histograms, bin_edges


def compile_displacements_tracked_mat(
    tracked_mat_file_or_directory,
    n_bins = 5001,
    max_displacement = 5.0,  #um
    n_dt = 4,      #number of delays at which to calculate displacements
    max_traj_len = None,
):
    if os.path.isdir(tracked_mat_file_or_directory):
        file_list = glob("%s/*Tracked.mat" % tracked_mat_file_or_directory)
    elif os.path.isfile(tracked_mat_file_or_directory):
        file_list = [tracked_mat_file_or_directory]
    else:
        print('input %s not recognized' % tracked_mat_file_or_directory)
        exit(1)

    bin_sequence = np.linspace(0, max_displacement, n_bins)
    displacements = np.zeros((n_dt, n_bins-1), dtype = 'uint32')

    for file_idx, file_name in enumerate(file_list):
        trajs = sio.loadmat(file_name)['trackedPar']
        if trajs.shape[0] == 1:
            trajs = trajs[0,:]

        for traj_idx in range(trajs.shape[0]):
            traj = trajs[traj_idx]
            if max_traj_len != None:
                traj[0] = traj[0][:max_traj_len, :]
                traj_frames = traj[1].flatten()[:max_traj_len]
            else:
                traj_frames = traj[1].flatten()

            if traj[0].shape[0] == 1:
                continue 
            for dt in range(1, n_dt + 1):
                for loc_idx in range(traj[0].shape[0]):
                    loc_frame = traj_frames[loc_idx]
                    if (loc_frame + dt) in traj_frames:
                        connect_idx = (traj_frames == (loc_frame + dt)).nonzero()[0][0]
                        displacement = traj[0][connect_idx, :] - traj[0][loc_idx, :]
                        r = np.sqrt((displacement ** 2).sum())
                        if r > max_displacement:
                            pass 
                        else:
                            bin_idx = (r <= bin_sequence).nonzero()[0][0] - 1
                            displacements[dt-1, bin_idx] += 1

    return displacements.T, bin_sequence 


def fit_radial_disp_model(
    radial_disp_histograms,
    bin_edges,
    delta_t,
    model,
    initial_guess_bounds = None,
    initial_guess_n = None,
    fit_bounds = None,
    plot = True,
    out_png = None,
    pdf_plot_max_r = 1.0,
    figsize_mod = 1.0,
    exp_bin_size = 0.04,
    **model_kwargs,
):
    '''
    Fit displacement histograms to a model.

    args
        radial_disp_histograms      :   2D ndarray of shape (n_bins-1, n_time_delays),
                                            the radial displacement histograms

        bin_edges                   :   1D ndarray of shape (n_bins,), the 
                                            edges of each radial displacement bin

        delta_t                     :   float, the interval between frames in 
                                            seconds

        initial_guess_bounds        :   2-tuple of 1D ndarray, the lower and upper
                                            bounds for the initial guesses. For instance:
                                            (np.array([0.0, 0.0, 0.5]), np.array([1.0, 0.05, 20.0]))

        initial_guess_n             :   tuple of int, the number of initial guesses
                                            for each model fit parameter

        fit_bounds                  :   2-tuple of 1D ndarray, the lower and upper
                                            bounds for parameter estimation                     

        model                       :   str, the model's key

        model_kwargs                :   for the model function

    If any of initial_guess_bounds, initial_guess_n, or fit_bounds is None,
    then program chooses a default set of parameters for these variables
    according to DEFAULTS.

    returns
        1D ndarray, the fit parameters for the model

    '''
    # Get the function corresponding to the model
    try:
        model_f = MODELS[model]['cdf']
    except KeyError:
        raise RuntimeError('traj_analysis.fit_radial_disp_model: model %s ' \
            'not supported; available models are %s' % (model, ', '.join(MODELS)))

    # Get default fitting/guess bounds if the user doesn't
    # provide them
    if type(initial_guess_bounds) == type(None):
        initial_guess_bounds = DEFAULTS[model]['initial_guess_bounds']

    if type(initial_guess_n) == type(None):
        initial_guess_n = DEFAULTS[model]['initial_guess_n']

    if type(fit_bounds) == type(None):
        fit_bounds = DEFAULTS[model]['fit_bounds']

    # Format input data for the independent/response variable
    # format of the scipy.optimize.curve_fit function. Here,
    # r_dt is a 2D ndarray of shape (n_points, 2) and 
    # ext_cdf is a 1D ndarray of shape (n_points,)
    r_dt, ext_cdf = _format_radial_disp_cdf(
        radial_disp_histograms,
        bin_edges,
        delta_t,
    )
    # plt.plot(np.arange(ext_cdf.shape[0]), ext_cdf); plt.show(); plt.close()

    # Fit data to model
    popt, pcov = _fit_from_initial_guesses(
        r_dt,
        ext_cdf,
        model_f,
        model_kwargs,
        initial_guess_bounds = initial_guess_bounds,
        initial_guess_n = initial_guess_n,
        fit_bounds = fit_bounds,
    )

    # Plot, if desired
    if plot:

        # Get the model PDF
        model_f_pdf = MODELS[model]['pdf']
        n_time_delays = radial_disp_histograms.shape[1]
        model_pdf = np.zeros((5000, n_time_delays), dtype = 'float64')
        model_cdf = np.zeros((5000, n_time_delays), dtype = 'float64')
        model_bin_centers = np.zeros((5000, 2), dtype = 'float64')
        model_bin_centers[:,0] = (bin_edges[:-1] + bin_edges[1:])/2
        for dt_idx in range(n_time_delays):
            model_bin_centers[:,1] = (dt_idx + 1) * delta_t 
            model_pdf[:,dt_idx] = model_f_pdf(
                model_bin_centers,
                *popt,
                **model_kwargs,
            )
            model_cdf[:,dt_idx] = model_f(
                model_bin_centers,
                *popt, 
                **model_kwargs,
            )

        # Plot all model PDFs with the data
        if type(out_png) == type(''):
            out_png_pdf = '%s_pdf.png' % os.path.splitext(out_png)[0]
            out_png_cdf = '%s_cdf.png' % os.path.splitext(out_png)[0]
        else:
            out_png_pdf = None 
            out_png_cdf = None 

        visualize.plot_cdf_with_model_from_histogram(
            radial_disp_histograms,
            bin_edges,
            model_cdf,
            model_bin_centers[:,0],
            delta_t,
            color_palette = 'magma',
            figsize_mod = figsize_mod,
            out_png = out_png_cdf,
        )

        #fig, ax = plt.subplots(4, 1, figsize = (2.8, 0.9 * 4))
        visualize.plot_pdf_with_model_from_histogram(
            radial_disp_histograms,
            bin_edges,
            model_pdf,
            model_bin_centers[:,0],
            delta_t,
            max_r = pdf_plot_max_r,
            exp_bin_size = exp_bin_size,
            figsize_mod = figsize_mod,
            out_png = out_png_pdf,
            #ax = ax,
        )
        
    return popt, pcov, MODELS[model]['par_names']

# 
# Fitting utilities
#
def _format_radial_disp_cdf(
    radial_disp_histograms,
    bin_edges,
    delta_t,
    fit_until = None,
):
    # Get the number of timepoint and radial distance bins
    n_bins, n_dt = radial_disp_histograms.shape

    # Check that the number of bins matches the input
    assert bin_edges.shape[0]-1 == n_bins 

    # Take the cumulative sum of each histogram to get the CDF
    cdfs = np.zeros((n_dt, n_bins), dtype = 'float64')
    for dt_idx in range(n_dt):
        cdfs[dt_idx, :] = np.cumsum(radial_disp_histograms[:, dt_idx])
        cdfs[dt_idx, :] = cdfs[dt_idx, :] / cdfs[dt_idx, -1]

    # Truncate the CDFs if desired
    if not (fit_until is None):
        cdfs = cdfs[:fit_]

    # Make the ext_cdf vector, which represents the response
    # variable for every tuple (r_bin, dt)
    ext_cdf = np.zeros(n_dt * n_bins, dtype = 'float64')
    ext_cdf[:] = cdfs.flatten()

    # Make the r_dt array, which represents the independent 
    # variable (r_bin, dt). Here, r_dt[2,0] would correspond
    # to the r_bin (right bin edge) of the second observation,
    # while r_dt[2,1] would correspond to the time delay of the
    # second observation. The tuple r_dt[2,:] is the independent
    # variable corresponding to the response variable ext_cdf[2]
    bin_size = bin_edges[1] - bin_edges[0]
    dt_indices, r_bins_left = np.mgrid[:n_dt, :n_bins]
    dt_indices = (dt_indices + 1) * delta_t 
    r_bins_left = r_bins_left * bin_size 
    r_bins_right = r_bins_left + bin_size 
    r_bins_center = r_bins_left + bin_size / 2

    r_dt = np.zeros((n_dt * n_bins, 2), dtype = 'float64')
    r_dt[:,0] = r_bins_right.flatten()
    r_dt[:,1] = dt_indices.flatten()

    # Return the independent and response variables
    return r_dt, ext_cdf 


def _fit_from_initial_guesses(
    r_dt,
    ext_cdf,
    model_function,
    model_kwargs,
    initial_guess_bounds = None,
    initial_guess_n = None,
    fit_bounds = None,
    verbose = True,
):
    '''
    Generate an array of initial guesses and do iterative
    fitting of the model parameters from each initial guess.

    args
        r_dt                :   2D ndarray of shape (n_points, 2),
                                    the right bin edges and frame interval
                                    corresponding to each data point

        ext_cdf             :   1D ndarray of shape (n_points,), the 
                                    response variable (CDF)

        model_function      :   function, the model function. Should take
                                    r_dt and fit pars and output an 
                                    approximation to ext_cdf

        initial_guess_bounds:   list of 2-tuple, the (lower, upper) bounds
                                    for each model fit par

        initial_guess_n     :   list of int, the number of initial guesses
                                    for each parameter to use

        fit_bounds          :   (1D ndarray, 1D ndarray), the lower and
                                    upper fitting bounds for each 
                                    parameter, passed to scipy.optimize.curve_fit's
                                    bounds parameter

    returns
        (popt, pcov), the optimal fit parameters and the corresponding
            covariance matrix

    '''
    # Generate the array of initial guesses
    m = len(initial_guess_bounds[0])
    lower_bounds = np.array(initial_guess_bounds[0])
    upper_bounds = np.array(initial_guess_bounds[1])
    initial_guesses = utils.cartesian_product(
        *(np.linspace(
            lower_bounds[i],
            upper_bounds[i],
            initial_guess_n[i]
        ) for i in range(m))
    )
    n_guesses = initial_guesses.shape[0]

    # For each initial guess, keep track of the sum of squared errors
    sum_sq_err = np.zeros(n_guesses, dtype = 'float64')

    # Perform fitting, starting at each initial guess
    def _model(r_dt, *args):
        return model_function(r_dt, *args, **model_kwargs)

    for g_idx, guess in tqdm(enumerate(initial_guesses)):
        popt, pcov = curve_fit(
            _model,
            r_dt,
            ext_cdf,
            bounds = fit_bounds,
            p0 = guess,
        )
        sum_sq_err[g_idx] = ((_model(r_dt, *popt) - ext_cdf)**2).sum()

    if verbose: print(list(sum_sq_err))

    # Identify the guess with the minimum sum of squares
    m_idx = np.argmin(sum_sq_err)
    guess = initial_guesses[m_idx, :]

    # Get the corresponding fit parameters for that initial guess
    popt, pcov = curve_fit(
        model_function,
        r_dt,
        ext_cdf,
        bounds = fit_bounds,
        p0 = guess,
    )

    # Return the result
    return popt, pcov 

# 
# Model functions
#
def cdf_two_state_brownian_zcorr(
    r_dt,
    f_bound,
    d_free,
    d_bound,
    loc_error = 0.035,
    delta_z = 0.7,
):
    '''
    CDF for the 2D radial displacements of a population of
    Brownian particles with two diffusion coefficients, *d_free* and
    *d_bound*, that are observed in a thin axial detection slice. The fraction
    of particles in the slower-moving state is given by *f_bound*, and
    particles are assumed not to convert between the states.

    This version uses a type of correction for the loss of particles
    due to axial diffusion out of the detection slice that is more
    suited to tracking with gaps.

    args
        r_dt        :   2D ndarray of shape (n_points, 2), the
                            (r, dt) points at which to evaluate
                            the CDF

        f_bound     :   float, fraction of molecules in the slower-moving
                            state

        d_free      :   float, diffusion coefficient for the fast state
                            in um^2 s^-1

        d_bound     :   float, diffusion coefficient for the slow state
                            in um^2 s^-1

        loc_error   :   float, localization error in um

        delta_z     :   float, thickness of the axial detection slice in um

    returns
        1D ndarray of shape (n_points,), the CDF at the 
            input points

    '''
    # 4 D t for the bound state 
    var2_0 = 2 * (2 * d_bound * r_dt[:,1] + loc_error**2)

    # 4 D t for the free state
    var2_1 = 2 * (2 * d_free * r_dt[:,1] + loc_error**2)

    # Squared radial displacement
    r2 = r_dt[:,0] ** 2

    # Correction for loss of free particles
    half_z = delta_z / 2

    def _fraction_lost(z0, pos_var):
        ''' pos_var : e.g. 4 D t '''
        return 0.5 * (gammaincc(0.5, (half_z-z0)**2 / pos_var) + \
            gammaincc(0.5, (half_z+z0)**2 / pos_var))

    unique_dts = np.unique(r_dt[:,1])
    f1_corr = np.zeros(r_dt.shape[0], dtype='float64')
    for unique_dt in unique_dts:
        f_lost = quad(
            _fraction_lost,
            -half_z,
            half_z,
            args = (var2_1[r_dt[:,1]==unique_dt][0])
        )[0] / delta_z 
        f1_corr[r_dt[:,1] == unique_dt] = (1-f_bound)*(1-f_lost) / (1-(1-f_bound)*f_lost)

    part_0 = np.exp(-r2 / var2_0)
    part_1 = np.exp(-r2 / var2_1)
    return 1 - (1-f1_corr)*part_0 - f1_corr*part_1

def pdf_two_state_brownian_zcorr(
    r_dt,
    f_bound,
    d_free,
    d_bound,
    loc_error = 0.035,
    delta_z = 0.7,
): 
    '''
    Density function for the 2D radial displacements of a population of
    Brownian particles with two diffusion coefficients, *d_free* and
    *d_bound*, that are observed in a thin axial detection slice. The fraction
    of particles in the slower-moving state is given by *f_bound*, and
    particles are assumed not to convert between the states.

    This version uses a type of correction for the loss of particles
    due to axial diffusion out of the detection slice that is more
    suited to tracking with gaps.

    args
        r_dt        :   2D ndarray of shape (n_points, 2), the
                            (r, dt) points at which to evaluate
                            the PDF

        f_bound     :   float, fraction of molecules in the slower-moving
                            state

        d_free      :   float, diffusion coefficient for the fast state
                            in um^2 s^-1

        d_bound     :   float, diffusion coefficient for the slow state
                            in um^2 s^-1

        loc_error   :   float, localization error in um

        delta_z     :   float, thickness of the axial detection slice in um

    returns
        1D ndarray of shape (n_points,), the PDF at the 
            input points

    '''
    # 4 D t for bound state
    var2_0 = 2 * (2 * d_bound * r_dt[:,1] + (loc_error ** 2))

    # 4 D t for free state
    var2_1 = 2 * (2 * d_free * r_dt[:,1] + (loc_error ** 2))

    # Squared radial displacement
    r2 = r_dt[:,0] ** 2

    # Correction for loss of free particles
    half_z = delta_z / 2

    def _fraction_lost(z0, pos_var):
        ''' pos_var : e.g. 4 D t '''
        return 0.5 * (gammaincc(0.5, (half_z-z0)**2 / pos_var) + \
            gammaincc(0.5, (half_z+z0)**2 / pos_var))

    unique_dts = np.unique(r_dt[:,1])
    f1_corr = np.zeros(r_dt.shape[0], dtype='float64')
    for unique_dt in unique_dts:
        f_lost = quad(
            _fraction_lost,
            -half_z,
            half_z,
            args = (var2_1[r_dt[:,1]==unique_dt][0])
        )[0] / delta_z 
        f1_corr[r_dt[:,1] == unique_dt] = (1 - f_bound) * (1 - f_lost) / (1 - (1 - f_bound) * f_lost)

    # Write out the PDF for each state, then add together
    # to get the two-state PDF
    pdf_0 = r_dt[:,0] * np.exp(-r2 / var2_0) / (0.5 * var2_0)
    pdf_1 = r_dt[:,0] * np.exp(-r2 / var2_1) / (0.5 * var2_1)

    return (1-f1_corr)*pdf_0 + f1_corr*pdf_1 

def cdf_two_state_brownian_zcorr_gapless(
    r_dt,
    f_bound,
    d_free,
    d_bound,
    loc_error = 0.035,
    delta_z = 0.7,
):
    '''
    CDF for the 2D radial displacements of a population of
    Brownian particles with two diffusion coefficients, *d_free* and
    *d_bound*, that are observed in a thin axial detection slice. The fraction
    of particles in the slower-moving state is given by *f_bound*, and
    particles are assumed not to convert between the states.

    This version uses a type of correction for the loss of particles
    due to axial diffusion out of the detection slice that is more
    suited to tracking without gaps.

    args
        r_dt        :   2D ndarray of shape (n_points, 2), the
                            (r, dt) points at which to evaluate
                            the CDF

        f_bound     :   float, fraction of molecules in the slower-moving
                            state

        d_free      :   float, diffusion coefficient for the fast state
                            in um^2 s^-1

        d_bound     :   float, diffusion coefficient for the slow state
                            in um^2 s^-1

        loc_error   :   float, localization error in um

        delta_z     :   float, thickness of the axial detection slice in um

    returns
        1D ndarray of shape (n_points,), the CDF at the 
            input points

    '''
    # 4 D t for the bound state 
    var2_0 = 2 * (2 * d_bound * r_dt[:,1] + loc_error**2)

    # 4 D t for the free state
    var2_1 = 2 * (2 * d_free * r_dt[:,1] + loc_error**2)

    # Squared radial displacement
    r2 = r_dt[:,0] ** 2

    # Correction for loss of free particles
    half_z = delta_z / 2

    def _fraction_lost(z0, pos_var):
        ''' pos_var : e.g. 4 D t '''
        return 0.5 * (gammaincc(0.5, (half_z-z0)**2 / pos_var) + \
            gammaincc(0.5, (half_z+z0)**2 / pos_var))

    unique_dts = np.unique(r_dt[:,1])
    f1_corr = np.zeros(r_dt.shape[0], dtype='float64')
    min_dt = unique_dts.min()
    f_remain_base = 1.0 - quad(
        _fraction_lost,
        -half_z,
        half_z,
        args = (2 * (2 * d_free * min_dt)),
    )[0] / delta_z 
    for unique_dt in unique_dts:
        f_remain = f_remain_base ** int(round((unique_dt / min_dt), 0)) 
        f1_corr[r_dt[:,1] == unique_dt] = (1-f_bound)*f_remain / (1-(1-f_bound)*(1-f_remain))

    part_0 = np.exp(-r2 / var2_0)
    part_1 = np.exp(-r2 / var2_1)
    return 1 - (1-f1_corr)*part_0 - f1_corr*part_1

def pdf_two_state_brownian_zcorr_gapless(
    r_dt,
    f_bound,
    d_free,
    d_bound,
    loc_error = 0.035,
    delta_z = 0.7,
):
    '''
    Density function for the 2D radial displacements of a population of
    Brownian particles with two diffusion coefficients, *d_free* and
    *d_bound*, that are observed in a thin axial detection slice. The fraction
    of particles in the slower-moving state is given by *f_bound*, and
    particles are assumed not to convert between the states.

    This version uses a type of correction for the loss of particles
    due to axial diffusion out of the detection slice that is more
    suited to tracking without gaps.

    args
        r_dt        :   2D ndarray of shape (n_points, 2), the
                            (r, dt) points at which to evaluate
                            the PDF

        f_bound     :   float, fraction of molecules in the slower-moving
                            state

        d_free      :   float, diffusion coefficient for the fast state
                            in um^2 s^-1

        d_bound     :   float, diffusion coefficient for the slow state
                            in um^2 s^-1

        loc_error   :   float, localization error in um

        delta_z     :   float, thickness of the axial detection slice in um

    returns
        1D ndarray of shape (n_points,), the PDF at the 
            input points

    '''
    # 4 D t + err for bound state
    var2_0 = 2 * (2 * d_bound * r_dt[:,1] + (loc_error ** 2))

    # 4 D t + err for free state
    var2_1 = 2 * (2 * d_free * r_dt[:,1] + (loc_error ** 2))

    # Squared radial displacement
    r2 = r_dt[:,0] ** 2

    # Correction for loss of free particles
    half_z = delta_z / 2

    def _fraction_lost(z0, pos_var):
        ''' pos_var : e.g. 4 D t '''
        return 0.5 * (gammaincc(0.5, (half_z-z0)**2 / pos_var) + \
            gammaincc(0.5, (half_z+z0)**2 / pos_var))

    unique_dts = np.unique(r_dt[:,1])
    f1_corr = np.zeros(r_dt.shape[0], dtype='float64')
    min_dt = unique_dts.min()
    f_remain_base = 1.0 - quad(
        _fraction_lost,
        -half_z,
        half_z,
        args = (2 * (2 * d_free * min_dt)),
    )[0] / delta_z 
    for unique_dt in unique_dts:
        f_remain = f_remain_base ** int(round((unique_dt / min_dt), 0)) 
        f1_corr[r_dt[:,1] == unique_dt] = (1-f_bound)*f_remain / (1-(1-f_bound)*(1-f_remain))

    # Write out the PDF for each state, then add together
    # to get the two-state PDF
    pdf_0 = r_dt[:,0] * np.exp(-r2 / var2_0) / (0.5 * var2_0)
    pdf_1 = r_dt[:,0] * np.exp(-r2 / var2_1) / (0.5 * var2_1)

    return (1-f1_corr)*pdf_0 + f1_corr*pdf_1 

def cdf_two_state_brownian(
    r_dt,
    f_bound,
    d_free,
    d_bound,
    loc_error = 0.035,
):
    # 4 D t for the bound state 
    var2_0 = 2 * (2 * d_bound * r_dt[:,1] + loc_error**2)

    # 4 D t for the free state
    var2_1 = 2 * (2 * d_free * r_dt[:,1] + loc_error**2)

    # Squared radial displacement
    r2 = r_dt[:,0] ** 2

    part_0 = np.exp(-r2 / var2_0)
    part_1 = np.exp(-r2 / var2_1)
    return 1 - f_bound*part_0 - (1-f_bound)*part_1

def pdf_two_state_brownian(
    r_dt,
    f_bound,
    d_free,
    d_bound,
    loc_error = 0.035,
    delta_z = 0.7,
):
    # 4 D t + err for bound state
    var2_0 = 2 * (2 * d_bound * r_dt[:,1] + (loc_error ** 2))

    # 4 D t + err for free state
    var2_1 = 2 * (2 * d_free * r_dt[:,1] + (loc_error ** 2))

    # Squared radial displacement
    r2 = r_dt[:,0] ** 2

    # Write out the PDF for each state, then add together
    # to get the two-state PDF
    pdf_0 = r_dt[:,0] * np.exp(-r2 / var2_0) / (0.5 * var2_0)
    pdf_1 = r_dt[:,0] * np.exp(-r2 / var2_1) / (0.5 * var2_1)

    return f_bound*pdf_0 + (1-f_bound)*pdf_1 

def cdf_one_state_brownian(
    r_dt,
    d,
    loc_error = 0.035,
):
    '''
    Cumulative distribution function for the 2D radial displacements
    of a Brownian walker with diffusion coefficient *d*.

    args
        r_dt        :   2D ndarray of shape (n_points, 2), the
                            (r, dt) points at which to evaluate
                            the CDF

        d           :   float, diffusion coefficient in um^2 s^-1

        loc_error   :   float, localization error in um

    returns
        1D ndarray of shape (n_points,), the CDF at the 
            input points

    '''
    # 4 D t + err
    var2 = 2 * (2 * d * r_dt[:,1] + loc_error**2)

    # Squared radial displacement
    r2 = r_dt[:,0]**2

    return 1.0 - np.exp(-r2 / var2)

def pdf_one_state_brownian(
    r_dt,
    d,
    loc_error = 0.035,
):
    '''
    Density function for the 2D radial displacements of a Brownian
    walker with diffusion coefficient *d*. 

    args
        r_dt        :   2D ndarray of shape (n_points, 2), the
                            (r, dt) points at which to evaluate
                            the PDF

        d           :   float, diffusion coefficient in um^2 s^-1

        loc_error   :   float, localization error in um

    returns
        1D ndarray of shape (n_points,), the PDF at the 
            input points

    '''
    # 4 D t + err
    var2 = 2 * (2 * d * r_dt[:,1] + loc_error**2)

    # Squared radial displacement
    r2 = r_dt[:,0]**2

    return r_dt[:,0] * np.exp(-r2 / var2) / (0.5 * var2)

def cdf_one_state_brownian_on_fractal(
    r_dt,
    d,
    df,
    loc_error = 0.035,
):
    '''
    Cumulative distribution function for the 2D radial 
    displacements of a Brownian particle diffusing on a
    2D fractal with fractal dimension *df*.

    args
        r_dt        :   2D ndarray of shape (n_points, 2), the
                            (r, dt) points at which to evaluate
                            the CDF

        d           :   float, diffusion coefficient in um^2 s^-1

        df          :   float, fractal dimension

        loc_error   :   float, localization error in um

    returns
        1D ndarray of shape (n_points,), the CDF at the 
            input points

    '''
    var2 = 2 * (2 * d * r_dt[:,1] + loc_error ** 2)
    r2 = r_dt[:,0] ** 2
    return gammainc(float(df)/2, r2/var2)

def pdf_one_state_brownian_on_fractal(
    r_dt,
    d,
    df,
    loc_error = 0.035,
):
    '''
    Density function for the 2D radial displacements of a
    Brownian particle diffusing on a 2D fractal with fractal
    dimension *df*.

    args
        r_dt        :   2D ndarray of shape (n_points, 2), the
                            (r, dt) points at which to evaluate
                            the PDF

        d           :   float, diffusion coefficient in um^2 s^-1

        df          :   float, fractal dimension

        loc_error   :   float, localization error in um

    returns
        1D ndarray of shape (n_points,), the PDF at the 
            input points

    '''
    var2 = 2 * (2 * d * r_dt[:,1] + loc_error ** 2)
    r2 = r_dt[:,0]**2

    return ( 2 * np.power(r_dt[:,0], df-1.0) * np.exp(-r2 / var2) ) / \
        ( np.power(var2, float(df) / 2) * gamma(float(df) / 2) )

def cdf_two_state_brownian_on_fractal(
    r_dt,
    f0,
    d0,
    d1,
    df,
    loc_error = 0.04,
    delta_z = 0.7,
):
    # positional variances of the two states
    var2_0 = 2 * (2 * d0 * r_dt[:,1] + loc_error**2)
    var2_1 = 2 * (2 * d1 * r_dt[:,1] + loc_error**2)

    # spectral dimension
    spec_dim = float(df) / 2 

    # squared displacement
    r2 = r_dt[:,0]**2

    # correction for loss of free particles
    half_z = delta_z / 2

    def _fraction_lost(z0, pos_var):
        ''' pos_var : e.g. 4 D t '''
        return 0.5 * (gammaincc(spec_dim/2, (half_z-z0)**2 / pos_var) + \
            gammaincc(spec_dim/2, (half_z+z0)**2 / pos_var))

    unique_dts = np.unique(r_dt[:,1])
    f1_corr = np.zeros(r_dt.shape[0], dtype = 'float64')
    for unique_dt in unique_dts:
        f_lost = quad(
            _fraction_lost,
            -half_z,
            half_z,
            args = (var2_1[r_dt[:,1]==unique_dt][0]),
        )[0] / delta_z 
        f1_corr[r_dt[:,1] == unique_dt] = (1 - f0) * (1 - f_lost) / (1 - (1 - f0) * f_lost)

    # The cdf of each state in terms of regularized lower incomplete
    # gamma functions
    cdf_state_0 = (1-f1_corr) * gammainc(1.0, r2 / var2_0)
    #cdf_state_0 = (1-f1_corr) * gammainc(spec_dim, r2 / var2_0)
    cdf_state_1 = f1_corr * gammainc(spec_dim, r2 / var2_1)

    return cdf_state_0 + cdf_state_1 

def pdf_two_state_brownian_on_fractal(
    r_dt,
    f0,
    d0,
    d1,
    df,
    loc_error = 0.04,
    delta_z = 0.7,
):
    # positional variances of the two states
    var2_0 = 2 * (2 * d0 * r_dt[:,1] + loc_error**2)
    var2_1 = 2 * (2 * d1 * r_dt[:,1] + loc_error**2)

    # spectral dimension
    spec_dim = float(df) / 2 

    # squared displacement
    r2 = r_dt[:,0]**2

    # correction for loss of free particles
    half_z = delta_z / 2

    def _fraction_lost(z0, pos_var):
        ''' pos_var : e.g. 4 D t '''
        return 0.5 * (gammaincc(spec_dim/2, (half_z-z0)**2 / pos_var) + \
            gammaincc(spec_dim/2, (half_z+z0)**2 / pos_var))

    unique_dts = np.unique(r_dt[:,1])
    f1_corr = np.zeros(r_dt.shape[0], dtype = 'float64')
    for unique_dt in unique_dts:
        f_lost = quad(
            _fraction_lost,
            -half_z,
            half_z,
            args = (var2_1[r_dt[:,1]==unique_dt][0]),
        )[0] / delta_z 
        f1_corr[r_dt[:,1] == unique_dt] = (1 - f0) * (1 - f_lost) / (1 - (1 - f0) * f_lost)

    # The pdf of each state in terms of regularized lower incomplete
    # gamma functions
    r_df_min_1 = np.power(r_dt[:,0], float(df)-1)
    pdf_state_0 = (1-f1_corr) * r_dt[:,0] * np.exp(-r2 / var2_0) / (0.5 * var2_0)
    #pdf_state_0 = (1-f1_corr) * 2 * r_df_min_1 * np.exp(-r2/var2_0) / (np.power(var2_0, spec_dim) * gamma(spec_dim))
    pdf_state_1 = f1_corr * 2 * r_df_min_1 * np.exp(-r2/var2_1) / (np.power(var2_1, spec_dim) * gamma(spec_dim))

    return pdf_state_0 + pdf_state_1 

def cdf_one_state_subdiffusion_on_fractal(
    r_dt,
    d,
    df,
    dw,
    loc_error = 0.035,
):
    dw = float(dw)
    var2 = np.power(2 * (2 * d * r_dt[:,1] + loc_error**2), 1.0/(dw-1))
    factor = dw / (dw - 1)
    r_factor = np.power(r_dt[:,0], factor)
    return gammainc(float(df)/dw, r_factor / var2)

def pdf_one_state_subdiffusion_on_fractal(
    r_dt,
    d,
    df,
    dw,
    loc_error = 0.035,
):
    dw = float(dw)
    factor = dw / (dw - 1)
    r_df_min_1 = np.power(r_dt[:,0], float(df)-1)
    r_factor = np.power(r_dt[:,0], factor)

    var2_pre = 2 * (2 * d * r_dt[:,1] + loc_error**2)

    var2 = np.power(var2_pre, 1.0/(dw-1))
    norm_1 = np.power(var2_pre, float(df) / dw)
    return factor * r_df_min_1 * np.exp(-r_factor / var2) / (norm_1 * gamma(df / factor))

def cdf_two_state_subdiffusion_on_fractal_zcorr(
    r_dt,
    f0,
    d0,
    d1,
    df,
    dw,
    loc_error = 0.047,
    delta_z = 0.7,
):
    '''
    Guyer model
    '''
    dw = float(dw)
    factor = dw / (dw - 1)
    r_df_min_1 = np.power(r_dt[:,0], float(df)-1)
    r_factor = np.power(r_dt[:,0], factor)
    var2_0 = 2 * (2 * d0 * r_dt[:,1] + loc_error**2)
    var2_1 = 2 * (2 * d1 * r_dt[:,1] + loc_error**2)
    var2_1_renorm_1 = np.power(var2_1, 1.0/(dw-1))

    # Correction for loss of free particles
    half_z = delta_z / 2
    spec_dim = df / factor 

    def _fraction_lost(z0, pos_var):
        return 0.5 * (gammaincc(spec_dim/2, np.power(half_z-z0, factor) / pos_var) + \
            gammaincc(spec_dim/2, np.power(half_z+z0, factor) / pos_var))

    unique_dts = np.unique(r_dt[:,1])
    f1_corr = np.zeros(r_dt.shape[0], dtype = 'float64')
    for unique_dt in unique_dts:
        f_lost = quad(
            _fraction_lost,
            -half_z,
            half_z,
            args = (var2_1_renorm_1[r_dt[:,1] == unique_dt][0]),
        )[0] / delta_z 
        f1_corr[r_dt[:,1] == unique_dt] = (1 - f0) * (1 - f_lost) / (1 - (1 - f0) * f_lost)

    cdf_state_0 = (1-f1_corr) * gammainc(1.0, r_dt[:,0]**2 / var2_0)
    cdf_state_1 = f1_corr * gammainc(spec_dim, r_factor / var2_1_renorm_1)

    return cdf_state_0 + cdf_state_1     

def pdf_two_state_subdiffusion_on_fractal_zcorr(
    r_dt,
    f0,
    d0,
    d1,
    df,
    dw,
    loc_error = 0.047,
    delta_z = 0.7,
):
    '''
    Guyer model
    '''
    dw = float(dw)
    factor = dw / (dw - 1)
    r_df_min_1 = np.power(r_dt[:,0], float(df)-1)
    r_factor = np.power(r_dt[:,0], factor)
    var2_0 = 2 * (2 * d0 * r_dt[:,1] + loc_error**2)
    var2_1 = 2 * (2 * d1 * r_dt[:,1] + loc_error**2)
    var2_1_renorm_0 = np.power(var2_1, float(df)/dw)
    var2_1_renorm_1 = np.power(var2_1, 1.0/(dw-1))

    # Correction for loss of free particles
    half_z = delta_z / 2
    spec_dim = df / factor 

    def _fraction_lost(z0, pos_var):
        return 0.5 * (gammaincc(spec_dim/2, np.power(half_z-z0, factor) / pos_var) + \
            gammaincc(spec_dim/2, np.power(half_z+z0, factor) / pos_var))

    unique_dts = np.unique(r_dt[:,1])
    f1_corr = np.zeros(r_dt.shape[0], dtype = 'float64')
    for unique_dt in unique_dts:
        f_lost = quad(
            _fraction_lost,
            -half_z,
            half_z,
            args = (var2_1_renorm_1[r_dt[:,1] == unique_dt][0]),
        )[0] / delta_z 
        f1_corr[r_dt[:,1] == unique_dt] = (1 - f0) * (1 - f_lost) / (1 - (1 - f0) * f_lost)

    pdf_state_0 = (1-f1_corr) * r_dt[:,0] * np.exp(-(r_dt[:,0]**2) / var2_0) / (0.5 * var2_0)
    pdf_state_1 = f1_corr * factor * r_df_min_1 * np.exp(-r_factor / var2_1_renorm_1) / (var2_1_renorm_0 * gamma(spec_dim))

    return pdf_state_0 + pdf_state_1

def cdf_two_state_subdiffusion_on_fractal(
    r_dt,
    f0,
    d0,
    d1,
    df,
    dw,
    loc_error = 0.054,
):
    dw = float(dw)
    factor = dw / (dw - 1)
    spec_dim = df / factor 

    r_df_min_1 = np.power(r_dt[:,0], float(df)-1)
    r_factor = np.power(r_dt[:,0], factor)
    var2_0 = 2 * (2 * d0 * r_dt[:,1] + loc_error**2)
    var2_1 = 2 * (2 * d1 * r_dt[:,1] + loc_error**2)
    var2_1_renorm_0 = np.power(var2_1, float(df)/dw)
    var2_1_renorm_1 = np.power(var2_1, 1.0/(dw-1))
    cdf_state_0 = f0 * gammainc(1.0, r_dt[:,0]**2 / var2_0)
    cdf_state_1 = (1-f0) * gammainc(spec_dim, r_factor / var2_1_renorm_1)

    return cdf_state_0 + cdf_state_1 

def pdf_two_state_subdiffusion_on_fractal(
    r_dt,
    f0,
    d0,
    d1,
    df,
    dw,
    loc_error = 0.054,
):
    dw = float(dw)
    factor = dw / (dw - 1)
    spec_dim = df / factor 

    r_df_min_1 = np.power(r_dt[:,0], float(df)-1)
    r_factor = np.power(r_dt[:,0], factor)
    var2_0 = 2 * (2 * d0 * r_dt[:,1] + loc_error**2)
    var2_1 = 2 * (2 * d1 * r_dt[:,1] + loc_error**2)
    var2_1_renorm_0 = np.power(var2_1, float(df)/dw)
    var2_1_renorm_1 = np.power(var2_1, 1.0/(dw-1))
    pdf_state_0 = f0 * r_dt[:,0] * np.exp(-(r_dt[:,0]**2) / var2_0) / (0.5 * var2_0)
    pdf_state_1 = (1-f0) * factor * r_df_min_1 * np.exp(-r_factor / var2_1_renorm_1) / (var2_1_renorm_0 * gamma(spec_dim))

    
    return pdf_state_0 + pdf_state_1 

def cdf_two_state_oshaughnessy(
    r_dt,
    f0,
    d0,
    d1,
    df,
    dw,
    loc_error = 0.04,
    delta_z = 0.7,
):
    '''
    STILL UNDER CONSTRUCTION
    '''
    dim = float(df) / dw
    r0_exp_dw = np.power(r_dt[:,0], dw)
    var0_2 = (dw**2) * d0 * r_dt[:,1] + 0.5 * loc_error**2
    var1_2 = (dw**2) + d1 * r_dt[:,1] + 0.5 * loc_error**2

    half_z = delta_z / 2

    def _fraction_lost(z0, pos_var):
        return 0.5 * (gammaincc(dim, np.power((half_z-z0), dw) / pos_var) + \
            gammaincc(dim, np.power((half_z+z0), dw) / pos_var))

    unique_dts = np.unique(r_dt[:,1])
    f1_corr = np.zeros(r_dt.shape[0], dtype = 'float64')
    for unique_dt in unique_dts:
        f_lost = quad(
            _fraction_lost,
            -half_z,
            half_z,
            args = (var1_2[r_dt[:,1] == unique_dt][0])
        )[0] / delta_z
        f1_corr[r_dt[:,1] == unique_dt] = (1 - f0) * (1 - f_lost) / (1 - (1 - f0) * f_lost)

    cdf_state_0 = (1-f1_corr) * gammainc(dim, r0_exp_dw / var0_2)
    cdf_state_1 = f1_corr * gammainc(dim, r0_exp_dw / var1_2)

    return cdf_state_0 + cdf_state_1 

def pdf_two_state_oshaughnessy(
    r_dt,
    f0,
    d0,
    d1,
    df,
    dw,
    loc_error = 0.04,
    delta_z = 0.7,
):
    '''
    STILL UNDER CONSTRUCTION
    '''
    dim = float(df) / dw
    r0_exp_dw = np.power(r_dt[:,0], dw)
    r0_exp_df_min_1 = np.power(r_dt[:,0], df-1)

    var0_2 = (dw**2) * d0 * r_dt[:,1] + 0.5 * loc_error**2
    var1_2 = (dw**2) + d1 * r_dt[:,1] + 0.5 * loc_error**2

    half_z = delta_z / 2

    def _fraction_lost(z0, pos_var):
        return 0.5 * (gammaincc(dim, np.power((half_z-z0), dw) / pos_var) + \
            gammaincc(dim, np.power((half_z+z0), dw) / pos_var))

    unique_dts = np.unique(r_dt[:,1])
    f1_corr = np.zeros(r_dt.shape[0], dtype = 'float64')
    for unique_dt in unique_dts:
        f_lost = quad(
            _fraction_lost,
            -half_z,
            half_z,
            args = (var1_2[r_dt[:,1] == unique_dt][0])
        )[0] / delta_z
        f1_corr[r_dt[:,1] == unique_dt] = (1 - f0) * (1 - f_lost) / (1 - (1 - f0) * f_lost)

    pdf_state_0 = (1-f1_corr) * dw * r0_exp_df_min_1 * np.exp(-r0_exp_dw / var0_2) / (np.power(var0_2, dim) * gamma(dim))
    pdf_state_1 = f1_corr * dw * r0_exp_df_min_1 * np.exp(-r0_exp_dw / var1_2) / (np.power(var1_2, dim) * gamma(dim))

    return pdf_state_0 + pdf_state_1 

def cdf_three_state_brownian(
    r_dt,
    f0,
    f1,
    d0,
    d1,
    d2,
    loc_error = 0.04,
):
    le2 = loc_error ** 2
    var2_0 = 2 * (2 * d0 * r_dt[:,1] + le2)
    var2_1 = 2 * (2 * d1 * r_dt[:,1] + le2)
    var2_2 = 2 * (2 * d2 * r_dt[:,1] + le2)

    r2 = r_dt[:,0] ** 2

    return 1 - (f0*np.exp(-r2/var2_0) + f1*np.exp(-r2/var2_1) + (1.0-f0-f1)*np.exp(-r2/var2_2))

def pdf_three_state_brownian(
    r_dt,
    f0,
    f1,
    d0,
    d1,
    d2,
    loc_error = 0.04,
):
    le2 = loc_error ** 2
    var2_0 = 2 * (2 * d0 * r_dt[:,1] + le2)
    var2_1 = 2 * (2 * d1 * r_dt[:,1] + le2)
    var2_2 = 2 * (2 * d2 * r_dt[:,1] + le2)
    r2 = r_dt[:,0] ** 2
    pdf_state_0 = r_dt[:,0] * np.exp(-r2/var2_0) / (0.5*var2_0)
    pdf_state_1 = r_dt[:,0] * np.exp(-r2/var2_1) / (0.5*var2_1)
    pdf_state_2 = r_dt[:,0] * np.exp(-r2/var2_2) / (0.5*var2_2)
    return f0*pdf_state_0 + f1*pdf_state_1 + (1.0-f0-f1)*pdf_state_2


# Models available for comparison
MODELS = {
    'two_state_brownian_zcorr' : {
        'cdf' : cdf_two_state_brownian_zcorr,
        'pdf' : pdf_two_state_brownian_zcorr,
        'par_names' : ['fraction_bound', 'diff_coef_free', 'diff_coef_bound'],
    },
    'two_state_brownian_zcorr_gapless' : {
        'cdf' : cdf_two_state_brownian_zcorr_gapless,
        'pdf' : pdf_two_state_brownian_zcorr_gapless,
        'par_names' : ['fraction_bound', 'diff_coef_free', 'diff_coef_bound'],
    },
    'one_state_brownian' : {
        'cdf' : cdf_one_state_brownian,
        'pdf' : pdf_one_state_brownian,
        'par_names' : ['diff_coef'],
    },
    'one_state_brownian_on_fractal' : {
        'cdf' : cdf_one_state_brownian_on_fractal,
        'pdf' : pdf_one_state_brownian_on_fractal,
        'par_names' : ['diff_coef', 'fractal_dim'],
    },
    'two_state_brownian_on_fractal' : {
        'cdf' : cdf_two_state_brownian_on_fractal,
        'pdf' : pdf_two_state_brownian_on_fractal,
        'par_names' : ['fraction_bound', 'diff_coef_bound', 'diff_coef_free', 'fractal_dim'],
    },
    'one_state_subdiffusion_on_fractal' : {
        'cdf' : cdf_one_state_subdiffusion_on_fractal,
        'pdf' : pdf_one_state_subdiffusion_on_fractal,
        'par_names' : ['diff_coef', 'fractal_dim', 'walk_dim'],
    },
    'two_state_subdiffusion_on_fractal_zcorr' : {
        'cdf' : cdf_two_state_subdiffusion_on_fractal_zcorr,
        'pdf' : pdf_two_state_subdiffusion_on_fractal_zcorr,
        'par_names' : ['fraction_bound', 'diff_coef_bound', 'diff_coef_free', 'fractal_dim', 'walk_dim'],
    },
    'two_state_subdiffusion_on_fractal_zcorr' : {
        'cdf' : cdf_two_state_subdiffusion_on_fractal,
        'pdf' : pdf_two_state_subdiffusion_on_fractal,
        'par_names' : ['fraction_bound', 'diff_coef_bound', 'diff_coef_free', 'fractal_dim', 'walk_dim'],
    },
    'two_state_oshaughnessy' : {
        'cdf' : cdf_two_state_oshaughnessy,
        'pdf' : pdf_two_state_oshaughnessy,
        'par_names' : ['fraction_bound', 'diff_coef_bound', 'diff_coef_free', 'fractal_dim', 'walk_dim'],
    },
    'three_state_brownian' : {
        'cdf' : cdf_three_state_brownian,
        'pdf' : pdf_three_state_brownian,
        'par_names' : ['fraction_bound', 'fraction_diff_free_1', 'diff_coef_bound', 'diff_coef_free_1', 'diff_coef_free2'],
    },
    'two_state_brownian' : {
        'cdf' : cdf_two_state_brownian,
        'pdf' : pdf_two_state_brownian,
        'par_names' : ['fraction_bound', 'diff_coef_bound', 'diff_coef_free'],
    },
}

# Default fitting bounds for each model, if the user doesn't
# specify
DEFAULTS = {
    'two_state_brownian_zcorr' : {
        'initial_guess_bounds' : (
            np.array([0, 0.2, 0.0]),
            np.array([1.0, 20.0, 0.01]),
        ),
        'initial_guess_n' : (5, 5, 1),
        'fit_bounds' : (
            np.array([0.0, 0.2, 0.0]),
            np.array([1.0, 50.0, 0.02]),
        ),
    },
    'two_state_brownian_zcorr_gapless' : {
        'initial_guess_bounds' : (
            np.array([0, 0.2, 0.0]),
            np.array([1.0, 20.0, 0.05]),
        ),
        'initial_guess_n' : (5, 5, 1),
        'fit_bounds' : (
            np.array([0.0, 0.2, 0.0]),
            np.array([1.0, 50.0, 0.05]),
        ),
    },
    'one_state_brownian' : {
        'initial_guess_bounds' : (
            np.array([0]),
            np.array([30.0]),
        ),
        'initial_guess_n' : np.array([20]),
        'fit_bounds' : (
            np.array([0.0]),
            np.array([100.0]),
        ),
    },
    'one_state_brownian_on_fractal' : {
        'initial_guess_bounds' : (
            np.array([0, 0.5]),
            np.array([30.0, 3.0]),
        ),
        'initial_guess_n' : np.array([5, 10]),
        'fit_bounds' : (
            np.array([0.0, 0.5]),
            np.array([100.0, 10.0]),
        ),
    },
    'two_state_brownian_on_fractal' : {
        'initial_guess_bounds' : (
            np.array([0.0, 0.0, 0.2, 0.5]),
            np.array([1.0, 0.01, 100.0, 3.0]),
        ),
        'initial_guess_n' : np.array([4, 1, 3, 4]),
        'fit_bounds' : (
            np.array([0.0, 0.0, 0.2, 0.5]),
            np.array([1.0, 0.1, 100.0, 3.0]),
        ),
    },
    'one_state_subdiffusion_on_fractal' : {
        'initial_guess_bounds' : (
            np.array([0, 0.5, 0.5]),
            np.array([30.0, 3.0, 4.0]),
        ),
        'initial_guess_n' : np.array([5, 5, 5]),
        'fit_bounds' : (
            np.array([0.0, 0.5, 0.5]),
            np.array([100.0, 5.0, 5.0]),
        ),
    },
    'two_state_subdiffusion_on_fractal_zcorr' : {
        'initial_guess_bounds' : (
            np.array([0.1, 0.01, 2.0, 1.0, 2.0]),
            np.array([0.5, 0.05, 20.0, 2.0, 3.0]),
        ),
        'initial_guess_n' : np.array([2, 1, 2, 2, 2]),
        'fit_bounds' : (
            np.array([0.0, 0.0, 0.2, 0.5, 2.0]),
            np.array([1.0, 0.05, 100.0, 4.0, 6.0]),
        ),
    },
    'two_state_subdiffusion_on_fractal' : {
        'initial_guess_bounds' : (
            np.array([0.1, 0.01, 2.0, 1.0, 2.0]),
            np.array([0.5, 0.05, 20.0, 2.0, 3.0]),
        ),
        'initial_guess_n' : np.array([2, 1, 2, 2, 2]),
        'fit_bounds' : (
            np.array([0.0, 0.0, 0.2, 0.5, 2.0]),
            np.array([1.0, 0.05, 100.0, 4.0, 6.0]),
        ),
    },
    'two_state_oshaughnessy' : {
        'initial_guess_bounds' : (
            np.array([0.1, 0.01, 2.0, 1.0, 2.0]),
            np.array([0.5, 0.05, 20.0, 2.0, 3.0]),
        ),
        'initial_guess_n' : np.array([2, 1, 2, 2, 2]),
        'fit_bounds' : (
            np.array([0.0, 0.0, 0.2, 0.5, 2.0]),
            np.array([1.0, 0.05, 100.0, 4.0, 6.0]),
        ),
    },
    'three_state_brownian' : {
        'initial_guess_bounds' : (
            np.array([0.0, 0.0, 0.01, 0.1, 1.0]),
            np.array([0.5, 0.5, 0.01, 10.0, 10.0]),
        ),
        'initial_guess_n' : np.array([2, 2, 1, 2, 2]),
        'fit_bounds' : (
            np.array([0.0, 0.0, 0.0, 0.1, 1.0]),
            np.array([1.0, 1.0, 0.1, 10.0, 100.0]),
        ),
    },
    'two_state_brownian' : {
        'initial_guess_bounds' : (
            np.array([0, 0.2, 0.0]),
            np.array([1.0, 20.0, 0.05]),
        ),
        'initial_guess_n' : (5, 5, 1),
        'fit_bounds' : (
            np.array([0.0, 0.2, 0.0]),
            np.array([1.0, 50.0, 0.1]),
        ),
    },
}

















