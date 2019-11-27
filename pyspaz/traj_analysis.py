'''
traj_analysis.py
'''
# Numerical tools
import numpy as np 

# Dataframes
import pandas as pd 

# I/O
import os
import sys
import tifffile

# Hard copies
from copy import copy 

# pyspaz functions
from pyspaz import spazio
from pyspaz import mask

# Models available for comparison
MODELS = {
    'two_state_brownian_zcorr' = two_state_brownian_zcorr,
}

def compile_displacements(
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

    # Make the sequence of radial displacement bins
    bin_edges = np.linspace(0, max_disp, n_bins)
    histograms = np.zeros((n_bins-1, time_delays), dtype = 'int64')

    # Convert loc coordinates from pixels to um
    df = trajs.assign(y_um = trajs[y_col] * pixel_size_um, x_um = trajs[x_col] * pixel_size_um)

    # If there are no gaps in tracking, then pandas.DataFrame.diff()
    # does all the work for us
    if n_gaps == 0:
        for t in range(1, time_delays + 1):
            disps = df.groupby(traj_col)[['y_um', 'x_um']].diff(t)
            disps = disps[~pd.isnull(disps['y_um'])]
            r_disps = np.sqrt((disps ** 2).sum(1))

            histo, _edges = np.histogram(r_disps, bins = bin_edges)
            histograms[:, t-1] = histo.copy()

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
    TXT files containing loclization/tracking information.

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
    if os.path.isdir(loc_file_list):
        dir_name = copy(loc_file_list)
        loc_file_list = ['%s/%s' % (dir_name, i) for i in os.listdir(dir_name)]

    # Check that all of the files exist
    for fname in loc_file_list:
        spazio.check_file_exists(fname)

    # Put all of the localizations into one dataframe
    df = pd.concat(
        [pd.read_csv(fname, sep='\t').assign(file_idx=i) for \
            i, fname in enumerate(loc_file_list)],
        ignore_index = True, sort = False,
    )

    # Compile radial displacement histograms
    histograms, bin_edges = compile_displacements(
        df,
        **kwargs,
    )
    return histograms, bin_edges 

def fit_radial_disp_model(
    radial_disp_histograms,
    bin_edges,
    model,
    plot = True,
    **model_kwargs,
):
    '''
    Fit displacement histograms to a model.

    args
        radial_disp_histograms      :   2D ndarray of shape (n_bins-1, n_time_delays),
                                            the radial displacement histograms

        bin_edges                   :   1D ndarray of shape (n_bins,), the 
                                            edges of each radial displacement bin

        model                       :   str, the model's key

        model_kwargs                :   for the model function

    returns
        1D ndarray, the fit parameters for the model

    '''
    # 
    # Get the function corresponding to the model
    try:
        model_f = MODELS[model]
    except KeyError:
        raise RuntimeError('traj_analysis.fit_radial_disp_model: model %s ' \
            'not supported; available models are %s' % (model, ', '.join(MODELS)))

    # Fit data to the model
    popt = model_f(
        radial_disp_histograms,
        bin_edges,
        **model_kwargs,
    )

    # Plot the result
    if plot:
        plot_radial_disps_with_model(
            radial_disp_histograms,
            bin_edges,
            model_f,
        )

    return popt 

def two_state_brownian_zcorr(
    radial_disp_histograms,
    bin_edges,
    initial_guess_bounds = ((0, 1), (0, 0.1), (0.2, 50.0)),
    n_initial_guesses_per_parameter = 5,
):
    # Calculate the empirical distribution function for each
    # displacement histogram
    cdfs = np.zeros((n_bins-1, n_dt), dtype = 'float64')
    for dt_idx in range(n_dt):
        cdfs[dt_idx, :] = np.cumsum(displacements[:, dt_idx])
        cdfs[dt_idx, :] = cdfs[dt_idx, :] / cdfs[dt_idx, -1]

    # Make the ext_cdf vector, which represents the response
    # variable for every tuple (r_bin, dt)
    ext_cdf = np.zeros((n_bins-1) * n_dt, dtype = 'float64')
    ext_cdf[:] = cdfs.flatten()

    # Make the r_dt array, which represents the independent 
    # variable (r_bin, dt). Here, r_dt[2,0] would correspond
    # to the r_bin (right bin edge) of the second observation,
    # while r_dt[2,1] would correspond to the time delay of the
    # second observation. The tuple r_dt[2,:] is the independent
    # variable corresponding to the response variable ext_cdf[2]
    bin_size = bin_edges[1] - bin_edges[0]
    n_bins = bin_edges.shape[0]

    r_bins_left, dt_indices = np.mgrid[:(n_bins-1), :n_dt]
    dt_indices = (dt_indices + 1) * dt 
    r_bins_left = r_bins_left * bin_size 
    r_bins_right = r_bins_left + bin_size 
    r_bins_center = r_bins_left + bin_size / 2

    # Generate an array of initial guesses
    lower_bounds = np.array([initial_guess_bounds[i][0] for i in range(3)])
    upper_bounds = np.array([initial_guess_bounds[i][1] for i in range(3)])
    initial_guesses = utils.cartesian_product(
        *(np.linspace(
            lower_bounds[i],
            upper_bounds[i],
            n_initial_guesses_per_parameter
        ) for i in range(3))
    )
    n_guesses = initial_guesses.shape[0]

    # For each initial guess, keep track of the sum of squared errors
    sum_sq_err = np.zeros(n_guesses, dtype = 'float64')

    # Perform fitting, starting at each initial guess
    for g_idx, guess in enumerate(initial_guesses):
        popt, pcov = curve_fit(
            rad_disp_cdf,
            r_dt,
            ext_cdf,
            bounds = (lower_bounds, upper_bounds),
            p0 = guess,
        )
        sum_sq_err[g_idx] = ((rad_disp_cdf(r_dt, *popt) - ext_cdf)**2).sum()

    # Identify the guess with the lowest squared error and refit 
    # from that one
    raise NotImplementedError




















