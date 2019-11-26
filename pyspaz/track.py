'''
track.py

'''
# Numerical essentials
import numpy as np 

# File reading / saving
import pandas as pd 
import os 
import sys

# For writing MATLAB output
from scipy import io as sio

# For getting a list of input files
import glob
from glob import glob

# Hungarian algorithm
from munkres import Munkres 
m = Munkres()

# Making hard copies of trajectories
import random
from copy import copy 
from time import time

# Various utilities
from . import utils
from . import spazio 

# Progress bar
from tqdm import tqdm 

def track_locs_directory(
    directory_name,
    out_dir = None,
    d_max = 20.0,
    d_bound_naive = 0.1,
    out_file = None,
    search_exp_fac = 3,
    pixel_size_um = 0.16,
    frame_interval_sec = 0.00548,
    min_int = 0.0,
    max_blinks = 0,
    k_return_from_blink = 1.0,
    y_int = 0.5,
    y_diff = 0.9,
    start_frame = None,
    stop_frame = None,
):
    '''
    Track all of the *.locs files in a given directory.

    If *out_dir* is None, the resulting *Tracked.mat files
    are placed in the same directory as the localizations.

    '''
    loc_files = glob("%s/*.locs" % directory_name)
    if out_dir == None:
        out_files = [i.replace('.locs', '_Tracked.mat') \
            for i in loc_files]
    else:
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        out_files = ['%s/%s' % (out_dir, \
            i.split('/')[-1].replace('.locs', '_Tracked.mat')) \
            for i in loc_files]

    for f_idx, fname in enumerate(loc_files):
        track_locs(
            fname,
            out_file = out_files[f_idx],
            d_max = d_max,
            d_bound_naive = d_bound_naive,
            search_exp_fac = search_exp_fac,
            pixel_size_um = pixel_size_um,
            frame_interval_sec = frame_interval_sec,
            min_int = min_int,
            max_blinks = max_blinks,
            k_return_from_blink = k_return_from_blink,
            y_int = y_int,
            y_diff = y_diff,
            start_frame = start_frame,
            stop_frame = stop_frame,
        )

def track_locs_hi_conf(
    locs_file,
    out_file = None,
    d_max = 20.0,
    search_exp_fac = 3,
    frame_interval_sec = 0.00548,
    pixel_size_um = 0.16,
    max_blinks = 0,
    min_int = 0.0,
    start_frame = None,
    stop_frame = None,
):
    # Determine the search radius, the maximum allowed distnace
    # between a trajectory and a connecting localization
    sig2_free = 2.0 * d_max * frame_interval_sec / (pixel_size_um ** 2)
    search_radius = search_exp_fac * np.sqrt(np.pi * 0.5 * sig2_free)
    search_radius_sq = search_radius ** 2

    locs, metadata = spazio.load_locs(locs_file)
    columns = ['frame_idx', 'y_pixels', 'x_pixels', 'I0', 'bg', 'llr_detection']
    if not all([c in locs.columns for c in columns]):
        raise RuntimeError("Localization CSV must contains columns %s" % ', '.join(columns))

    # Truncate localizations to begin and end at the 
    # desired frame interval for tracking
    if start_frame != None:
        locs = locs[locs['frame_idx'] >= start_frame]
    else:
        start_frame = 0
    if stop_frame != None:
        locs = locs[locs['frame_idx'] <= stop_frame]
    n_frames = int(locs['frame_idx'].max())

    # Convert the important columns to an ndarray for faster indexing
    locs = np.asarray(locs[columns])

    # Initiate trajectories from each localization in the first frame
    if start_frame != None:
        frame_idx = start_frame 
    else:
        frame_idx = 0
    frame_locs = locs[locs[:,0] == frame_idx, :]
    active_trajectories = [Trajectory(frame_locs[i, :], 0.0, (0,1)) \
        for i in range(frame_locs.shape[0])]

    new_trajectories = []
    completed_trajectories = []

    # Go through the frames, reconnecting trajectories with 
    # localizations as we go
    for frame_idx in tqdm(range(start_frame + 1, n_frames + 1)):

        # Get all locs in this frame
        frame_locs = locs[locs[:,0] == frame_idx, :]

        # Get the size of the frame-wide tracking problem
        n_trajs = len(active_trajectories)
        n_locs = frame_locs.shape[0]

        # If there are no active trajectories, consider starting
        # some from the localizations if they pass the minimum
        # intensity threshold. Set these as active trajectories
        # and go on to the next frame.
        if n_trajs == 0:
            for loc_idx in range(n_locs):
                loc = frame_locs[loc_idx, :]
                if loc[3] >= min_int:
                    new_trajectory = Trajectory(loc, 0.0, (0, 1))
                    new_trajectories.append(new_trajectory)
            active_trajectories = copy(new_trajectories)
            new_trajectories = []
            continue 

        # If there are no localizations, set all of the trajectories
        # into blink and move onto the next frame
        if n_locs == 0:
            for traj_idx, trajectory in enumerate(active_trajectories):

                # Increment this trajectory's blink counter
                trajectory.n_blinks += 1

                # Terminate this trajectory if it has too many blinks
                if trajectory.n_blinks > max_blinks:
                    completed_trajectories.append(trajectory)

                # Otherwise deal with it in the next frame
                else:
                    new_trajectories.append(trajectory)

            active_trajectories = copy(new_trajectories)
            new_trajectories = []
            continue 

        # Otherwise, we have some number of trajectories and localizations
        # in the frame. First, get the adjacency matrix.
        sq_radial_distances = utils.sq_radial_distance_array(
            np.asarray([traj.positions[-1, :] for traj in active_trajectories]),
            frame_locs[:, 1:3],
        )
        adjacency_matrix = (sq_radial_distances <= search_radius_sq).astype('uint16')

        # Break the assignment problem into subproblems based on adjacency
        subgraphs, y_index_lists, x_index_lists, trajs_without_locs, locs_without_trajs = \
            utils.connected_components(adjacency_matrix)

        # Deal with trajectories without localizations in their search radii
        for traj_idx in trajs_without_locs:

            # Get the corresponding trajectory from the pool of running
            # trajectories
            trajectory = active_trajectories[traj_idx]

            # Increment the blink counter of this trajectory
            trajectory.n_blinks += 1

            # If this trajectory has too many blinks, terminate it
            if trajectory.n_blinks > max_blinks:
                completed_trajectories.append(trajectory)
            else:
                new_trajectories.append(trajectory)

        # Deal with localizations that do not lie within the search radius
        # of any trajectory, starting new trajectories if they pass the
        # intensity threshold
        for loc_idx in locs_without_trajs:
            loc = frame_locs[loc_idx, :]
            if loc[3] >= min_int:
                new_trajectory = Trajectory(loc, 0.0, (0, 1))
                new_trajectories.append(new_trajectory)

        # The remainder of the assignment subproblems have some number
        # of trajectories and some number of localizations
        for subgraph_idx, subgraph in enumerate(subgraphs):

            # If there is only one trajectory and only one localization,
            # the assignment is unambiguous
            if subgraph.shape == (1, 1):

                # Get the corresponding trajectory by looking up its 
                # index in the assignment subproblem
                traj_idx = y_index_lists[subgraph_idx][0]
                trajectory = active_trajectories[traj_idx]

                # Find the corresponding localization
                loc_idx = x_index_lists[subgraph_idx][0]
                loc = frame_locs[loc_idx, :]

                # Add the localization to the trajectory
                trajectory.add_loc(loc, (1, 1))

                # Copy to the trajectory list for the next frame
                new_trajectories.append(trajectory)

            # Otherwise we have some ambiguity: multiple trajectories
            # and/or localizations. Kill all ambiguity in the problem
            # and start again.
            else:

                # Terminate each trajectory in this subproblem
                traj_indices = y_index_lists[subgraph_idx]
                for traj_idx in traj_indices:
                    completed_trajectories.append(active_trajectories[traj_idx])

                # For each localization, start a new trajectory if it
                # passes the intensity threshold
                for loc_idx in x_index_lists[subgraph_idx]:
                    loc = frame_locs[loc_idx, :]
                    if loc[3] >= min_int:
                        new_trajectory = Trajectory(loc, 0.0, (0, 1))
                        new_trajectories.append(new_trajectory)

        # Pass any active trajectories to be considered for reconnection
        # in the next frame.
        active_trajectories = new_trajectories
        new_trajectories = []

    # Wrap up any trajectories that are still running in the last frame
    for traj_idx, trajectory in enumerate(active_trajectories):
        completed_trajectories.append(copy(trajectory))

    # Make output filename
    if out_file == None:
        out_file = '%s_Tracked.mat' % locs_file.replace('.locs', '')

    # Add tracking parameters to metadata
    for k in metadata.keys():
        if metadata[k] == None:
            metadata[k] = 'None'

    metadata['locs_used_for_tracking'] = locs_file
    metadata['d_max'] = d_max
    metadata['d_bound_naive'] = 'None' 
    metadata['search_exp_fac'] = search_exp_fac
    metadata['frame_interval_sec'] = frame_interval_sec
    metadata['pixel_size_um'] = pixel_size_um
    metadata['min_int'] = min_int
    metadata['max_blinks'] = max_blinks
    metadata['k_return_from_blink'] = 'None'
    metadata['y_int'] = 'None'
    metadata['y_diff'] = 'None' 
    metadata['tracking_start_frame'] = str(start_frame)
    metadata['tracking_stop_frame'] = str(stop_frame)

    # Save to file
    spazio.save_trajectory_obj_to_mat(
        out_file,
        completed_trajectories,
        metadata,
        frame_interval_sec,
        pixel_size_um = pixel_size_um,
        convert_pixel_to_um = True,
    )

    # Return (trajs, metadata, traj_cols)
    return spazio.load_trajs(out_file)

OUTPUT_FORMATS = [
    'mat',
    'txt',
]
ALGORITHM_TYPES = [
    'full',
    'conservative',
    'equal_size',
]

def track_locs(
    loc_file,
    out_file = None,
    d_max = 20.0,
    d_bound_naive = 0.1,
    search_exp_fac = 3,
    pixel_size_um = 0.16,
    frame_interval_sec = 0.00548,
    min_int = 0.0,
    max_blinks = 0,
    k_return_from_blink = 1.0,
    y_int = 0.5,
    y_diff = 0.9,
    start_frame = None,
    stop_frame = None,
    output_format = 'txt',
    algorithm_type = 'full',
):
    # Check that user has specified compatible file formats 
    if output_format not in OUTPUT_FORMATS:
        raise RuntimeError('track.track_locs: output format %s not supported; use one of %s' % (output_format, ', '.join(OUTPUT_FORMATS)))

    if type(out_file) == type(''):
        if output_format == 'mat' and '.mat' not in out_file:
            raise RuntimeError('track.track_locs: out file %s not compatible with output format .mat' % out_file)
        elif output_format == 'txt' and not ('.trajs' in out_file or '.txt' in out_file):
            raise RuntimeError('track.track_locs: out file %s not compatible with output format .txt' % out_file)

    # Check that user has supplied an available reconnection algorithm
    if algorithm_type not in ALGORITHM_TYPES:
        raise RuntimeError('track.track_locs: algorithm type %s not supported; use one of %s' % (algorithm_type, ', '.join(ALGORITHM_TYPES)))

    # Calculate the two-dimensional radial variance for 
    # a particle traveling at *d_max* or *d_bound_naive*
    sig2_free = 2.0 * d_max * frame_interval_sec / (pixel_size_um ** 2)
    sig2_bound_naive = 2.0 * d_bound_naive * frame_interval_sec / (pixel_size_um ** 2)

    # Determine the search radius, the maximum allowed distance
    # between a trajectory and a connecting localization.
    search_radius = search_exp_fac * np.sqrt(np.pi * d_max * \
        frame_interval_sec / (pixel_size_um**2))

    # Square the search radius, so we don't have to take square 
    # roots of every single radial displacement
    search_radius_sq = search_radius ** 2

    # Read the localization csv and make sure we have all of the 
    # right columns
    locs_df, metadata = spazio.load_locs(loc_file)
    columns = ['frame_idx', 'y_pixels', 'x_pixels', 'I0', 'bg', 'llr_detection']
    if not all([c in locs_df.columns for c in columns]):
        raise RuntimeError("Localization CSV must contains columns %s" % ', '.join(columns))

    # Truncate the localizations to begin and end at the 
    # desired frame interval for tracking
    if start_frame != None:
        locs_df = locs_df[locs_df['frame_idx'] >= start_frame]
    if stop_frame != None:
        locs_df = locs_df[locs_df['frame_idx'] <= stop_frame]
    n_frames = int(locs_df['frame_idx'].max())

    # Convert the important columns to numpy.array, which has
    # a quicker accession
    locs = np.asarray(locs_df[columns])

    # Get the spot intensity mean and variance, for the intensity
    # reconnection test
    mean_spot_intensity = locs[:,3].mean()
    var_spot_intensity = locs[:,3].var()

    # Initiate trajectories from each localization in the
    # first frame
    if start_frame == None:
        start_frame = 0
    frame_idx = start_frame

    if stop_frame == None:
        stop_frame = n_frames - 1

    frame_locs = locs[locs[:,0] == frame_idx, :]
    active_trajectories = [Trajectory(frame_locs[i, :], sig2_bound_naive, (0,1)) \
        for i in range(frame_locs.shape[0])]

    # Save trajectories as we go in three lists:
    #    active_trajectories holds trajectories to be considered
    #       for reconnection in the current frame
    #active_trajectories = []

    #    new_trajectories holds trajectories that have already been
    #       dealt with in this frame. They'll live on into the next
    #       frame.
    new_trajectories = []

    #    completed_trajectories holds any trajectories that have
    #       been terminated. These are our results.
    completed_trajectories = []

    # Iterate through the rest of the frames, saving trajectories
    # as we go.
    for frame_idx in tqdm(range(start_frame + 1, n_frames + 1)):

        # Get all localizations in this frame
        frame_locs = locs[locs[:,0] == frame_idx, :]

        # Get the size of the tracking problem
        n_trajs = len(active_trajectories)
        n_locs = frame_locs.shape[0]

        # If there are no active trajectories, consider starting
        # some from the localizations if they pass the minimum
        # intensity threshold. Set these as active trajectories
        # and go on to the next frame
        if n_trajs == 0:
            for loc_idx in range(n_locs):
                loc = frame_locs[loc_idx, :]
                if loc[3] >= min_int:
                    new_trajectory = Trajectory(loc, sig2_bound_naive, (0,1))
                    new_trajectories.append(new_trajectory)
            active_trajectories = copy(new_trajectories)
            new_trajectories = []
            continue

        # If there are no localizations, set all of the trajectories
        # into blink and move onto the next frame
        if n_locs == 0:
            for traj_idx, trajectory in enumerate(active_trajectories):

                # Increment this trajectory's blink counter
                trajectory.n_blinks += 1

                # Terminate the trajectory if it has too many blinks
                if trajectory.n_blinks > max_blinks:
                    completed_trajectories.append(trajectory)

                # Otherwise deal with it in the next frame
                else:
                    new_trajectories.append(trajectory)

            active_trajectories = copy(new_trajectories)
            new_trajectories = []
            continue 

        # Otherwise, we have some number of trajectories and localizations.
        # First, compute the squared radial distance between the last
        # known position of each trajectory and each new localization
        sq_radial_distances = utils.sq_radial_distance_array(
            np.asarray([traj.positions[-1, :] for traj in active_trajectories]),
            frame_locs[:,1:3],
        )

        # Take only reconnections that are within the search radius.
        within_search_radius = (sq_radial_distances <= search_radius_sq).astype('uint16')

        # Break the assignment problem into subproblems based on adjacency
        subgraphs, y_index_lists, x_index_lists, trajs_without_locs, locs_without_trajs = \
            utils.connected_components(within_search_radius)

        # Deal with trajectories without localizations in their search radii
        for traj_idx in trajs_without_locs:

            # Get the corresponding trajectory from the pool of
            # running trajectories
            trajectory = active_trajectories[traj_idx]

            # Increment the blink counter of this trajectory
            trajectory.n_blinks += 1

            # If this trajectory has too many blinks, terminate it
            if trajectory.n_blinks > max_blinks:
                completed_trajectories.append(trajectory)
            else:
                new_trajectories.append(trajectory)

        # Deal with localizations that do not lie within the search radius of
        # any trajectory, starting new trajectories if they pass the intensity threshold
        for loc_idx in locs_without_trajs:
            loc = frame_locs[loc_idx, :]
            if loc[3] >= min_int:
                new_trajectory = Trajectory(loc, sig2_bound_naive, (0,1))
                new_trajectories.append(new_trajectory)

        # The remainder of the assignment subproblems have some number of 
        # trajectories and some number of localizations
        for subgraph_idx, subgraph in enumerate(subgraphs):

            # If only one trajectory and only one localization, the assignment
            # is unambiguous
            if subgraph.shape == (1, 1):

                # Get the corresponding trajectory by looking up its index
                # in the assignment subproblem
                traj_idx = y_index_lists[subgraph_idx][0]
                trajectory = active_trajectories[traj_idx]

                # Find the corresponding localization
                loc_idx = x_index_lists[subgraph_idx][0]
                loc = frame_locs[loc_idx, :]

                # Add the localization to the trajectory
                trajectory.add_loc(loc, (1,1))

                # Copy to the trajectories for the next frame
                new_trajectories.append(trajectory)

            # Otherwise we have some ambiguity: multiple trajectories and/or
            # localizations. Pass the problem to the *assign* function, which
            # solves individual subgraph problems with the Hungarian algorithm
            else:
                if algorithm_type == 'full':
                    running_trajectories, finished_trajectories = assign_full(
                        [active_trajectories[i] for i in y_index_lists[subgraph_idx]],
                        frame_locs[x_index_lists[subgraph_idx], :],
                        mean_spot_intensity,
                        var_spot_intensity,
                        frame_interval_sec = frame_interval_sec,
                        sig2_bound_naive = sig2_bound_naive,
                        sig2_free = sig2_free,
                        k_return_from_blink = k_return_from_blink,
                        y_int = y_int,
                        y_diff = y_diff,
                        max_blinks = max_blinks,
                    )
                    for trajectory in running_trajectories:
                        new_trajectories.append(copy(trajectory))
                    for trajectory in finished_trajectories:
                        completed_trajectories.append(copy(trajectory))

                # Throw away all potentially ambiguous reconnections
                elif algorithm_type == 'conservative':

                    # Terminate each trajectory in this subproblem
                    traj_indices = y_index_lists[subgraph_idx]
                    for traj_idx in traj_indices:
                        completed_trajectories.append(active_trajectories[traj_idx])

                    # For each localization, start a new trajectory if it
                    # passes the intensity threshold
                    for loc_idx in x_index_lists[subgraph_idx]:
                        loc = frame_locs[loc_idx, :]
                        if loc[3] >= min_int:
                            new_trajectory = Trajectory(loc, 0.0, (0, 1))
                            new_trajectories.append(new_trajectory)

                # Only solve subproblems with n_trajs == n_locs 
                elif algorithm_type == 'equal_size':
                    raise NotImplementedError

        # Pass any active trajectories to be considered for reconnection in
        # the next frame
        active_trajectories = new_trajectories
        new_trajectories = []

    # Wrap up any trajectories that are still running at the last frame
    for traj_idx, trajectory in enumerate(active_trajectories):
        completed_trajectories.append(copy(trajectory))

    # Add tracking parameters to metadata
    for k in metadata.keys():
        if metadata[k] == None:
            metadata[k] = 'None'

    metadata['locs_used_for_tracking'] = loc_file
    metadata['d_max'] = d_max
    metadata['d_bound_naive'] = d_bound_naive
    metadata['search_exp_fac'] = search_exp_fac
    metadata['frame_interval_sec'] = frame_interval_sec
    metadata['pixel_size_um'] = pixel_size_um
    metadata['min_int'] = min_int
    metadata['max_blinks'] = max_blinks
    metadata['k_return_from_blink'] = k_return_from_blink
    metadata['y_int'] = y_int
    metadata['y_diff'] = y_diff 
    metadata['tracking_start_frame'] = str(start_frame)
    metadata['tracking_stop_frame'] = str(stop_frame)

    # Save to file
    if output_format == 'mat':
        if out_file == None:
            out_file = '%s_Tracked.mat' % loc_file.replace('.locs', '')
        spazio.save_trajectory_obj_to_mat(
            out_file,
            completed_trajectories,
            metadata,
            frame_interval_sec,
            pixel_size_um = pixel_size_um,
            convert_pixel_to_um = True,
        )
    elif output_format == 'txt':
        if out_file == None:
            out_file = '%s.trajs' % loc_file.replace('.locs', '')
        spazio.save_trajectory_obj_to_txt(
            out_file,
            completed_trajectories,
            metadata,
            frame_interval_sec,
            pixel_size_um = pixel_size_um,
        )

    # Return (trajs, metadata, traj_cols)
    return spazio.load_trajs(out_file)

def assign_full(
    trajectories,
    localizations,
    mean_spot_intensity,
    var_spot_intensity,
    frame_interval_sec = 0.00548,
    sig2_bound_naive = 0.0584,
    sig2_free = 8.77,
    k_return_from_blink = 1.0,
    y_int = 0.5,
    y_diff = 0.9,
    max_blinks = 0,
):
    '''
    args
        trajectories            :   list of class Trajectory()
        localizations           :   np.array of shape (N_locs, 10),
                                    the localizations and their 
                                    associated information. See
                                    below for the required format.
        mean_spot_intensity     :   float, the mean estimated spot
                                    intensity across the whole 
                                    population of spots
        var_spot_intensity      :   float, the variance in estimated
                                    spot intensity across the whole
                                    population of spots
        frame_interval_sec      :   float
        sig2_bound_naive        :   float, naive estimate for
                                    2 D_{bound} dt. For instance,
                                    if D_{bound} = 0.1 um^2 s^-1,
                                    then sigma_bound_naive = 
                                    2 * D_{bound} * dt / (0.16**2)
                                    ~= 0.0584
        sig2_free               :   float, the same parameter but
                                    for the free population
        k_return_from_blink     :   float, the rate constant governing
                                    the return-from-blink Poisson process
                                    in frames^-1
        y_int                   :   float between 0.0 and 1.0
        y_diff                  :   float between 0.0 and 1.0
                                

    Format of localizations:

        localizations[loc_idx, 0]   :   global loc idx [not used]
        localizations[loc_idx, 1]   :   frame idx [not used]
        localizations[loc_idx, 2]   :   fitted y position
        localizations[loc_idx, 3]   :   fitted x position
        localizations[loc_idx, 4]   :   detection intensity (alpha)
        localizations[loc_idx, 5]   :   detection noise variance (sig2)
        localizations[loc_idx, 6]   :   result_ok [not used]
        localizations[loc_idx, 7]   :   detected y position (int) [not used]
        localizations[loc_idx, 8]   :   detected x position (int) [not used]
        localizations[loc_idx, 9]   :   variance of subwindow for detection
                                        (equivalent to mle variance under
                                        detection hypothesis H0)
        localizations[loc_idx, 10]  :   fitted I0 [not used]
        localizations[loc_idx, 11]  :   fitted bg [not used]
        

    Note that localizations should be given as the slice of a larger
    dataframe that contains localizations that should be considered for
    detection. That is, it should probably correspond to a single frame_idx.
    But, you know, your funeral.

    '''
    # Get the size of the subproblem - the number of competing trajectories
    # and/or localizations
    n_trajs = len(trajectories)
    n_locs = localizations.shape[0]

    # The size of the log-likelihood matrix is n_trajs + n_locs, which
    # allows for localizations to start new trajectories or old trajectories
    # to end
    n_dim = n_trajs + n_locs

    # Instantiate the matrix of log-likelihoods. 
    LL = np.zeros((n_dim, n_dim), dtype = 'float64')

    # If i >= n_trajs and j < n_locs, then LL[i,j] is the cost of starting
    # a new trajectory. This is set to the log-likelihood ratio for detection.
    for j in range(n_locs):
        #LL[n_trajs:, j] = localizations[j, 5]
        LL[n_trajs:, j] = -10.0

    # If i < n_trajs and j >= n_locs, then LL[i,j] is the cost of putting
    # a trajectory into blink. This is set to a fixed value.
    LL[:, n_locs:] = -50

    # If i < n_trajs and j < n_locs, then LL[i,j] corresponds to the log-likelihood
    # of some traj-loc reconnection. This is the sum of three terms that depend
    # on the traj-loc distance, the intensity of the localization, and the 
    # blinking status of the trajectory.

    # First, compute the intensity law, which is the same for all trajectories
    P_int = np.ones(localizations.shape[0], dtype = 'float64') * \
        (1 - y_int) / mean_spot_intensity
    P_int += y_int * np.exp(-(localizations[:,3]-mean_spot_intensity)**2/(2*var_spot_intensity) \
        / (np.sqrt(2 * np.pi * var_spot_intensity)))
    P_int[P_int <= 0.0] = 1.0
    L_int = np.log(P_int)

    # Iterate through the trajectories for the diffusion/blinking contributions
    for traj_idx, trajectory in enumerate(trajectories):

        # Return-from-blink penalty
        L_blink = -k_return_from_blink * trajectory.n_blinks + np.log(k_return_from_blink)

        # Displacement likelihood; varies from -5 to 0 and strongly favors closer
        # reconnections
        squared_r = utils.sq_radial_distance(trajectory.positions[-1,:], localizations[:,1:3])
        r = np.sqrt(squared_r)

        # If there are enough past displacements, estimate the radial displacement
        # variance for the bound state. Else set it to its default, naive_sig2_bound.
        sig2_bound = trajectory.calculate_sig2_bound()

        # Adjust the expected diffusion for gaps in the trajectory
        sig2_bound = sig2_bound * (1 + trajectory.n_blinks)
        traj_sig2_free = sig2_free * (1 + trajectory.n_blinks)

        # Get the relative probabilities of diffusing to each of the
        # eligible localizations
        P_disp_bound = (r / sig2_bound) * np.exp(-squared_r / (2 * sig2_bound))
        P_disp_free = (r / traj_sig2_free) * np.exp(-squared_r / (2 * traj_sig2_free))
        L_diff = np.log(y_diff*P_disp_bound + (1-y_diff)*P_disp_free)

        # Assign to the corresponding positions in the log-likelihood array
        LL[traj_idx, :n_locs] = L_blink + L_diff + L_int 

    # The Hungarian algorithm will find the minimum path through its matrix
    # argument, so take the negative of the log likelihoods
    LL = LL * -1 

    # Find the trajectory-localization assignment that maximizes the total
    # log likelihoods
    assignments = m.compute(LL)
    assign_traj = [i[0] for i in assignments]
    assign_locs = [i[1] for i in assignments]

    # Deal with the aftermath, keeping track of which trajectories
    # have finished and which are still running
    active_trajectories = []
    finished_trajectories = []

    # For each trajectory, either set it into blink or match it with the
    # corresponding localization
    for traj_idx, trajectory in enumerate(trajectories):

        # Trajectory did not match with a localization: increment its
        # blink counter and terminate if necessary
        if assign_locs[traj_idx] >= n_locs:
            trajectory.n_blinks += 1
            if trajectory.n_blinks > max_blinks:
                finished_trajectories.append(trajectory)
            else:
                active_trajectories.append(trajectory)

        # Trajectory matched with a localization; update its info
        else:
            loc_idx = assign_locs[traj_idx]
            loc = localizations[loc_idx, :]
            trajectory.add_loc(loc, (n_trajs, n_locs))

            # Pass the trajectory on for reconnection in the next frame
            active_trajectories.append(trajectory)

    # For each unpaired localization, start a new trajectory
    for traj_idx in range(n_trajs, n_dim):

        # Get the loc info
        loc_idx = assign_locs[traj_idx]

        # Index corresponds to a real localization
        if loc_idx < n_locs:
            loc = localizations[loc_idx, :]
            trajectory = Trajectory(loc, sig2_bound_naive, (n_trajs, n_locs))

            # Pass the trajectory for reconnection in the next frame
            active_trajectories.append(trajectory)

    return active_trajectories, finished_trajectories


class Trajectory(object):
    '''
    A convenience object used internally in the
    tracking program.

    To create an instance, pass *locs*, a np.array
    with the following information:

        locs[0] : frame index
        locs[1] : y position (pixels)
        locs[2] : x position (pixels)
        locs[3] : fitted PSF intensity (photons)
        locs[4] : fitted BG intensity (photons)
        locs[5] : log likelihood ratio for detection

    *naive_sig2* is the best guess for the `bound`
    radial displacement variance, lacking prior 
    information for this trajectory.

    *subproblem_shape* is a 2-tuple (int, int), the
        number of competing trajectories and localizations
        for that localization.

        (0, non-zero) -> a new trajectory with no surrounding trajectories
        (non-zero, 0) -> termination (should not occur)
        (1, 1)        -> unambiguous 1:1 traj:loc assignment
        (n, m)        -> n trajectories competing for m localizations

    '''
    def __init__(
        self,
        loc,
        naive_sig2,
        subproblem_shape,
    ):
        self.positions = np.asarray([loc[1:3]])
        self.frames = [int(loc[0])]
        self.mle_I0 = [loc[3]]
        self.mle_bg = [loc[4]]
        self.llr_detect = [loc[5]]
        self.n_blinks = 0
        self.naive_sig2 = naive_sig2
        self.subproblem_shapes = [subproblem_shape]

    def add_loc(self, loc, subproblem_shape):
        # Update the positions array: extend the numpy array by 1
        # index to accommodate the new localization
        new_pos = np.zeros((self.positions.shape[0]+1, 2), \
            dtype = self.positions.dtype)
        new_pos[:-1, :] = self.positions
        new_pos[-1, :] = loc[1:3]
        self.positions = new_pos.copy()

        # Update the other attributes
        self.frames.append(int(loc[0]))
        self.mle_I0.append(loc[3])
        self.mle_bg.append(loc[4])
        self.llr_detect.append(loc[5])

        # Set the blink counter back to 0
        self.n_blinks = 0

        # Record the size of the tracking subproblem
        self.subproblem_shapes.append(subproblem_shape)


    def calculate_sig2_bound(self):
        # If the trajectory has a past history, use this to estimate
        # its diffusion coefficient using the Brownian MSD and from that
        # the expected radial displacement variance.
        n_pos = self.positions.shape[0]
        if n_pos > 1:
            delta_frames = np.array(self.frames[1:]) - np.array(self.frames[:-1])
            sq_disp_per_frame = ((self.positions[1:, :] - \
                self.positions[:-1, :])**2).sum(axis=1) / delta_frames
            sig2_bound = sq_disp_per_frame.mean() / 2
        else:
            sig2_bound = self.naive_sig2

        # Adjust for the expected movement during a gap frame
        sig2_bound = sig2_bound * (1 + self.n_blinks)

        return sig2_bound



