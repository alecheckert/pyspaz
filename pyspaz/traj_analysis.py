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

# I/O
import os
import sys
import tifffile

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

        return histograms, bin_edges, df 

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

        return histograms, bin_edges, trajs_with_gaps 

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
    histograms, bin_edges, df = compile_displacements(
        df,
        **kwargs,
    )
    return histograms, bin_edges, df

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
    pdf_plot_max_r = 2.0,
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

        visualize.plot_pdf_with_model_from_histogram(
            radial_disp_histograms,
            bin_edges,
            model_pdf,
            model_bin_centers[:,0],
            delta_t,
            max_r = pdf_plot_max_r,
            exp_bin_size = 0.02,
            figsize_mod = 1.0,
            out_png = out_png_pdf,
        )

        visualize.plot_cdf_with_model_from_histogram(
            radial_disp_histograms,
            bin_edges,
            model_cdf,
            model_bin_centers[:,0],
            delta_t,
            color_palette = 'magma',
            figsize_mod = 1.0,
            out_png = out_png_cdf,
        )

    return popt 

# 
# Fitting utilities
#
def _format_radial_disp_cdf(
    radial_disp_histograms,
    bin_edges,
    delta_t,
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
    lower_bounds = np.array(initial_guess_bounds[0])
    upper_bounds = np.array(initial_guess_bounds[1])
    initial_guesses = utils.cartesian_product(
        *(np.linspace(
            lower_bounds[i],
            upper_bounds[i],
            initial_guess_n[i]
        ) for i in range(3))
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
    CDF for the radial displacements of a two-state Brownian
    particle without state transitions. Only displacements
    in a thin axial slice of thickness *delta_z* are observed.

    args
        r_dt                :   2D ndarray of shape (n_points, 2),
                                    the right radial displacement bins
                                    and frame intervals for each data point

        f_bound             :   float, fraction of molecules in 
                                    the slower-moving (`bound`) state

        d_free              :   float, diffusion coefficient for 
                                    free state in um^2 s^-1

        d_bound             :   float, diffusion coefficient for
                                    bound state in um^2 s^-1

        loc_error           :   float, localization error in um

        delta_z             :   float, the thickness of the axial
                                    detection/observation slice in um

    returns
        1D ndarray, the model CDF at the points indicated by 
            r_dt

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
    # 2 D t for bound state
    var2_0 = 2 * (2 * d_bound * r_dt[:,1] + (loc_error ** 2))

    # 2 D t for free state
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


# Models available for comparison
MODELS = {
    'two_state_brownian_zcorr' : {
        'cdf' : cdf_two_state_brownian_zcorr,
        'pdf' : pdf_two_state_brownian_zcorr,
    },
}

# Default fitting bounds for each model, if the user doesn't
# specify
DEFAULTS = {
    'two_state_brownian_zcorr' : {
        'initial_guess_bounds' : (
            np.array([0, 0.2, 0.0]),
            np.array([1.0, 20.0, 0.05]),
        ),
        'initial_guess_n' : (5, 1, 5),
        'fit_bounds' : (
            np.array([0.0, 0.2, 0.0]),
            np.array([1.0, 50.0, 0.05]),
        ),
    }
}

















