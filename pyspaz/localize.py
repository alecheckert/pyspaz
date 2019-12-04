'''
localize.py
'''
# Convenient file finder
from glob import glob 

# Get numerical essentials
import numpy as np

# For convolutions in the detection step
import scipy.ndimage as ndi 
from scipy.ndimage import uniform_filter

# Quick erfs for the PSF definition
from scipy.special import erf 

# For file reading / writing
from . import spazio 
import pandas as pd 

# For showing our work
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import axes3d

# Frame counter
from tqdm import tqdm
import time 

# The radial symmetry method involves a possible divide-by-zero that
# is subsequently corrected. The divide-by-zero warnings are 
# temporarily disabled for this step.
import warnings

# Opening the result files in the test_detect() and test_localize() programs
import os

# Utility functions for localization
from . import utils 

# # Parallelization
import dask 
# from dask.distributed import Client, progress 
# from dask.diagnostics import ProgressBar 
# client = Client(n_workers = 8, threads_per_worker = 1)


def radial_symmetry(psf_image):
    '''
    Use the radial symmetry method to estimate the center
    y and x coordinates of a PSF.

    This method was originally conceived by
    Parasarathy R Nature Methods 9, pgs 724–726 (2012).

    args
        image   :   2D np.array (ideally small and symmetric,
                    e.g. 9x9 or 13x13), the image of the PSF.
                    Larger frames relative to the size of the PSF
                    will reduce the accuracy of the estimation.

    returns
        np.array([y_estimate, x_estimate]), the estimated
            center of the PSF in pixels, relative to
            the corner of the image frame.

    '''
    # Get the size of the image frame and build
    # a set of pixel indices to match
    N, M = psf_image.shape
    N_half = N // 2
    M_half = M // 2
    ym, xm = np.mgrid[:N-1, :M-1]
    ym = ym - N_half + 0.5
    xm = xm - M_half + 0.5 
    
    # Calculate the diagonal gradients of intensities across each
    # corner of 4 pixels
    dI_du = psf_image[:N-1, 1:] - psf_image[1:, :M-1]
    dI_dv = psf_image[:N-1, :M-1] - psf_image[1:, 1:]
    
    # Smooth the image to reduce the effect of noise, at the cost
    # of a little resolution
    fdu = uniform_filter(dI_du, 3)
    fdv = uniform_filter(dI_dv, 3)
    
    dI2 = (fdu ** 2) + (fdv ** 2)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m = -(fdv + fdu) / (fdu - fdv)
        
    # For pixel values that blow up, instead set them to a very
    # high float
    m[np.isinf(m)] = 9e9
    
    b = ym - m * xm

    sdI2 = dI2.sum()
    ycentroid = (dI2 * ym).sum() / sdI2
    xcentroid = (dI2 * xm).sum() / sdI2
    w = dI2 / np.sqrt((xm - xcentroid)**2 + (ym - ycentroid)**2)

    # Correct nan / inf values
    w[np.isnan(m)] = 0
    b[np.isnan(m)] = 0
    m[np.isnan(m)] = 0

    # Least-squares analytical solution to the point of 
    # maximum radial symmetry, given the slopes at each
    # edge of 4 pixels
    wm2p1 = w / ((m**2) + 1)
    sw = wm2p1.sum()
    smmw = ((m**2) * wm2p1).sum()
    smw = (m * wm2p1).sum()
    smbw = (m * b * wm2p1).sum()
    sbw = (b * wm2p1).sum()
    det = (smw ** 2) - (smmw * sw)
    xc = (smbw*sw - smw*sbw)/det
    yc = (smbw*smw - smmw*sbw)/det

    # Adjust coordinates so that they're relative to the
    # edge of the image frame
    yc = (yc + (N + 1) / 2.0) - 1
    xc = (xc + (M + 1) / 2.0) - 1

    # Add 0.5 pixel shift to get back to the original indexing.
    # This is not necessarily the desired behavior, so I've
    # commented out this.
    # fit_vector = np.array([yc, xc]) + 0.5
    fit_vector = np.array([yc, xc])

    return fit_vector 

def detect_frame(
    image,
    sigma = 1.0,
    window_size = 9,
    detect_threshold = 20.0,
    plot = False,
):
    image = image.astype('float64')
    N, M = image.shape 
    half_w = window_size // 2

    # Compute the kernels for spot detection
    g = utils.gaussian_model(sigma, window_size)
    gc = g - g.mean() 
    gaussian_kernel = utils.expand_window(gc, N, M)
    gc_rft = np.fft.rfft2(gaussian_kernel)

    # Compute some required normalization factors for spot detection
    n_pixels = window_size ** 2
    Sgc2 = (gc ** 2).sum()

   # Perform the convolutions required for the LL detection test
    A = uniform_filter(image, window_size) * n_pixels
    B = uniform_filter(image**2, window_size) * n_pixels
    im_rft = np.fft.rfft2(image)
    C = np.fft.ifftshift(np.fft.irfft2(im_rft * gc_rft))

    # Calculate the likelihood of a spot in each pixel,
    # and set bad values to 1.0 (i.e. no chance of detection)
    L = 1 - (C**2) / (Sgc2*(B - (A**2)/float(n_pixels)))
    L[:half_w,:] = 1.0
    L[:,:half_w] = 1.0
    L[-(1+half_w):,:] = 1.0
    L[:,-(1+half_w):] = 1.0
    L[L <= 0.0] = 0.001

    # Calculate log likelihood of the presence of a spot
    LL = -(n_pixels / 2) * np.log(L)

    # Find pixels that pass the detection threshold
    detections = LL > detect_threshold 

    # For each detection that consists of adjacent pixels,
    # take the local maximum
    peaks = utils.local_max_2d(LL) & detections

    # Find the coordinates of the detections
    detected_positions = np.asarray(np.nonzero(peaks)).T + 1
    
    # Copy the detection information to the result array
    n_detect = detected_positions.shape[0]
    result = np.zeros((n_detect, 3), dtype = 'float64')
    result[:, :2] = detected_positions.copy()
    result[:, 2] = LL[np.nonzero(peaks)]

    if plot:
        fig, ax = plt.subplots(2, 3, figsize = (9, 6))
        ax[0,0].imshow(image, cmap = 'gray')
        ax[0,1].imshow(C, cmap = 'gray')
        ax[0,2].imshow(LL, cmap = 'gray')
        ax[1,0].imshow(detections, cmap = 'gray')
        ax[1,1].imshow(peaks, cmap = 'gray')
        plot_image = image.copy()
        for detect_idx in range(n_detect):
            y, x = detected_positions[detect_idx, :]
            for i in range(-2, 3):
                plot_image[y+i,x] = plot_image.max()
                plot_image[y,x+i] = plot_image.max()
        ax[1,2].imshow(plot_image, cmap = 'gray')

        ax[0,0].set_title('Original')
        ax[0,1].set_title('Conv kernel')
        ax[0,2].set_title('Log likelihood of spot')
        ax[1,0].set_title('> threshold')
        ax[1,1].set_title('Peaks')
        ax[1,2].set_title('Original + detections')
        plt.show(); plt.close()

    return result 

def localize_frame(
    image,
    sigma = 1.0,
    window_size = 9,
    detect_threshold = 20.0,
    plot_detect = False,
    plot_localize = False,
    camera_bg = 470,
    camera_gain = 110,
    initial_guess = 'radial_symmetry',
    max_iter = 20,
    convergence_crit = 1.0e-4,
    divergence_crit = 1000.0,
    enforce_negative_definite = False,
    damp = 0.2,
    calculate_error = False,
):
    image = image.astype('float64')

    ## Run spot detection 
    detected_positions = detect_frame(
        image,
        sigma = sigma,
        window_size = window_size, 
        detect_threshold = detect_threshold,
        plot = plot_detect,
    )
    n_detect = detected_positions.shape[0]

    ## Run subpixel localization
    # Precompute some required factors for localization
    n_pixels = window_size ** 2
    half_w = window_size // 2
    y_field, x_field = np.mgrid[:window_size, :window_size]
    sig2_2 = 2 * (sigma ** 2)
    sqrt_sig2_2 = np.sqrt(sig2_2)
    sig2_pi_2 = np.pi * sig2_2
    sqrt_sig2_pi_2 = np.sqrt(sig2_pi_2)
    
    # Initialize the vector of PSF model parameters. We'll
    # always use the following index scheme:
    #    pars[0]: y center of the PSF
    #    pars[1]: x center of the PSF
    #    pars[2]: I0, the number of photons in the PSF
    #    pars[3]: Ibg, the number of BG photons per pixel
    init_pars = np.zeros(4, dtype = 'float64')
    pars = np.zeros(4, dtype = 'float64')
    
    # Gradient of the log-likelihood w/ respect to each of the four parameters
    grad = np.zeros(4, dtype = 'float64')
    
    # Hessian of the log-likelihood
    H = np.zeros((4, 4), dtype = 'float64')
    
    # Store the results of localization in a large numpy.array. 
    locs = np.zeros((n_detect, 10), dtype = 'float64')

    # Keep track of information about each localization in a result
    # array.
    if calculate_error:
        result = np.zeros((n_detect, 14), dtype = 'float64')
    else:
        result = np.zeros((n_detect, 10), dtype = 'float64')
    result[:,:3] = detected_positions.copy()

    if plot_localize:
        q = np.sqrt(n_detect)
        if q % 1.0 == 0.0:
            q = int(q)
        else:
            q = int(np.ceil(q))
        fig, ax = plt.subplots(q, q, figsize = (6, 6))

    for d_idx in range(n_detect):

       # Get a small subwindow surrounding that detection
        detect_y = int(detected_positions[d_idx, 0])
        detect_x = int(detected_positions[d_idx, 1])
        psf_image = (image[
            detect_y - half_w : detect_y + half_w + 1,
            detect_x - half_w : detect_x + half_w + 1
        ] - camera_bg) / camera_gain
        psf_image[psf_image < 0.0] = 0.0
    
        # If the image is not square (edge detection), set the error
        # column to True and move on
        if psf_image.shape[0] != psf_image.shape[1]:
            result[d_idx, 7] = True 
            continue

        # Make the initial parameter guesses.

        # Guess by radial symmetry (usually the best)
        if initial_guess == 'radial_symmetry':
            init_pars[0], init_pars[1] = radial_symmetry(psf_image)
            init_pars[3] = np.array([
                psf_image[0,:-1].mean(),
                psf_image[-1,1:].mean(),
                psf_image[1:,0].mean(),
                psf_image[:-1,-1].mean()
            ]).mean()

        # Guess by centroid method
        elif initial_guess == 'centroid':
            init_pars[0] = (y_field * psf_image).sum() / psf_image.sum()
            init_pars[1] = (x_field * psf_image).sum() / psf_image.sum()
            init_pars[3] = np.array([
                psf_image[0,:-1].mean(),
                psf_image[-1,1:].mean(),
                psf_image[1:,0].mean(),
                psf_image[:-1,-1].mean()
            ]).mean()

        # Simply place the guess in the center of the window
        elif initial_guess == 'window_center':
            init_pars[0] = window_size / 2
            init_pars[1] = window_size / 2
            init_pars[3] = psf_image.min()

        # Other initial guess methods are not implemented
        else:
            raise NotImplementedError

        # Use the model specification to guess I0
        max_idx = np.argmax(psf_image)
        max_y_idx = max_idx // window_size
        max_x_idx = max_idx % window_size 
        init_pars[2] = psf_image[max_y_idx, max_x_idx] - init_pars[3]
        E_y_max = 0.5 * (erf((max_y_idx + 0.5 - init_pars[0]) / sqrt_sig2_2) - \
                erf((max_y_idx - 0.5 - init_pars[0]) / sqrt_sig2_2))
        E_x_max = 0.5 * (erf((max_x_idx + 0.5 - init_pars[1]) / sqrt_sig2_2) - \
                erf((max_x_idx - 0.5 - init_pars[1]) / sqrt_sig2_2))
        init_pars[2] = init_pars[2] / (E_y_max * E_x_max)

        # If the I0 guess looks crazy (usually because there's a bright pixel
        # on the edge), make a more conservative guess by integrating the
        # inner ring of pixels
        if (np.abs(init_pars[2]) > 1000) or (init_pars[2] < 0.0):
            init_pars[2] = psf_image[half_w-1:half_w+2, half_w-1:half_w+2].sum()

        # Set the current parameter set to the initial guess. Hold onto the
        # initial guess so that if MLE diverges, we have a fall-back.
        pars[:] = init_pars.copy()

        # Keep track of the current number of iterations. When this exceeds
        # *max_iter*, then the iteration is terminated.
        iter_idx = 0

        # Continue iterating until the maximum number of iterations is reached, or
        # if the convergence criterion is reached
        update = np.ones(4, dtype = 'float64')
        while (iter_idx < max_iter) and any(np.abs(update[:2]) > convergence_crit):

            #Calculate the PSF model under the current parameter set
            E_y = 0.5 * (erf((y_field+0.5-pars[0])/sqrt_sig2_2) - \
                erf((y_field-0.5-pars[0])/sqrt_sig2_2))
            E_x = 0.5 * (erf((x_field+0.5-pars[1])/sqrt_sig2_2) - \
                erf((x_field-0.5-pars[1])/sqrt_sig2_2))
            model = pars[2] * E_y * E_x + pars[3]

            # Avoid divide-by-zero errors
            nonzero = (model > 0.0)

            # Calculate the derivatives of the model with respect
            # to each of the four parameters
            du_dy = ((pars[2] / (2 * sqrt_sig2_pi_2)) * (
                np.exp(-(y_field-0.5-pars[0])**2 / sig2_2) - \
                np.exp(-(y_field+0.5-pars[0])**2 / sig2_2)
            )) * E_x
            du_dx = ((pars[2] / (2 * sqrt_sig2_pi_2)) * (
                np.exp(-(x_field-0.5-pars[1])**2 / sig2_2) - \
                np.exp(-(x_field+0.5-pars[1])**2 / sig2_2)
            )) * E_y 
            du_dI0 = E_y * E_x 
            du_dbg = np.ones((window_size, window_size), dtype = 'float64')

            # Calculate the gradient of the log-likelihood at the current parameter vector
            J_factor = (psf_image[nonzero] - model[nonzero]) / model[nonzero]
            grad[0] = (du_dy[nonzero] * J_factor).sum()
            grad[1] = (du_dx[nonzero] * J_factor).sum()
            grad[2] = (du_dI0[nonzero] * J_factor).sum()
            grad[3] = (du_dbg[nonzero] * J_factor).sum()

            # Calculate the Hessian
            H_factor = psf_image[nonzero] / (model[nonzero]**2)
            H[0,0] = (-H_factor * du_dy[nonzero]**2).sum()
            H[0,1] = (-H_factor * du_dy[nonzero]*du_dx[nonzero]).sum()
            H[0,2] = (-H_factor * du_dy[nonzero]*du_dI0[nonzero]).sum()
            H[0,3] = (-H_factor * du_dy[nonzero]).sum()
            H[1,1] = (-H_factor * du_dx[nonzero]**2).sum()
            H[1,2] = (-H_factor * du_dx[nonzero]*du_dI0[nonzero]).sum()
            H[1,3] = (-H_factor * du_dx[nonzero]).sum()
            H[2,2] = (-H_factor * du_dI0[nonzero]**2).sum()
            H[2,3] = (-H_factor * du_dI0[nonzero]).sum()
            H[3,3] = (-H_factor).sum()

            # Use symmetry to complete the Hessian.
            H[1,0] = H[0,1]
            H[2,0] = H[0,2]
            H[3,0] = H[0,3]
            H[2,1] = H[1,2]
            H[3,1] = H[1,3]
            H[3,2] = H[2,3]

            # Invert the Hessian, stably
            Y = np.diag([1.0e-5, 1.0e-5, 1.0e-5, 1.0e-5])
            if enforce_negative_definite:
                while 1:
                    try:
                        pivots = get_pivots(H - Y)
                        if (pivots > 0.0).any():
                            Y *= 10
                            continue
                        else:
                            H_inv = np.linalg.inv(H - Y)
                            break
                    except (ZeroDivisionError, np.linalg.linalg.LinAlgError):
                        Y *= 10
                        continue
            else:
                while 1:
                    try:
                        H_inv = np.linalg.inv(H - Y)
                        break
                    except (ZeroDivisionError, np.linalg.linalg.LinAlgError):
                        Y *= 10
                        continue

            # Get the update vector, the change in parameters
            update = -H_inv.dot(grad) * damp 

            # Update the parameters
            pars = pars + update 
            iter_idx += 1

        # If the estimate has diverged, fall back to the initial guess
        if any(np.abs(update[:2]) >= divergence_crit):
            pars = init_pars
        if  ~check_pos_inside_window(pars[:2], window_size, edge_tolerance = 2):
            pars = init_pars
        else:
            # Correct for the half-pixel indexing error that results from
            # implicitly assigning intensity values to the corners of pixels
            # in the PSF definition
            pars[:2] = pars[:2] + 0.5

        # Give the resulting y, x coordinates in terms of the whole image
        pars[0] = pars[0] + detect_y - half_w
        pars[1] = pars[1] + detect_x - half_w

        # Save the parameter vector for this localization
        result[d_idx, 3:7] = pars.copy()

        # Save the image variance, which will become useful in the tracking
        # step
        result[d_idx, 8] = psf_image.var()

        # Make an estimate for the error in the parameter estimation 
        # by inverting the observed information matrix, acquired
        # from the Jacobian
        if calculate_error:

            # The Jacobian is a 4-by-n_pixels matrix of the derivatives
            # of each model parameter at each pixel
            jac = np.array((du_dy, du_dx, du_dI0, du_dbg))
            J_factor = (psf_image[nonzero] - model[nonzero]) / model[nonzero]
            _grad = (jac[:,nonzero] * J_factor).sum(1)
            jac_div_sqrt_model = jac[:,nonzero] / np.sqrt(model[nonzero])

            # Observed information matrix I
            I = jac_div_sqrt_model.dot(jac_div_sqrt_model.T)
            try:
                err = np.diagonal(np.linalg.inv(I))
            except np.linalg.LinAlgError as e:
                err = np.full(len(I), np.nan)

            result[d_idx, 9:13] = np.sqrt(err)

        if plot_localize:
            if q == 1:
                ax.imshow(psf_image, cmap = 'gray')
                ax.plot(
                    [pars[1] - detect_x + half_w - 0.5],
                    [pars[0] - detect_y + half_w - 0.5],
                    marker = '.',
                    markersize = 10,
                    color = 'r',
                )
            else:
                ax[d_idx//q, d_idx%q].imshow(
                    psf_image, cmap = 'gray',
                )
                ax[d_idx//q, d_idx%q].plot(
                    [pars[1] - detect_x + half_w - 0.5],
                    [pars[0] - detect_y + half_w - 0.5],
                    marker = '.',
                    markersize = 10,
                    color = 'r',
                )

    if plot_localize:
        plt.show(); plt.close()

    # Format the result as a dataframe
    result_df = pd.DataFrame(result, columns = [
        'y_detect_pixels',
        'x_detect_pixels',
        'llr_detection',
        'y_pixels',
        'x_pixels',
        'I0',
        'bg',
        'error',
        'subwindow_variance',
        'empty'
    ])

    return result_df


def detect_and_localize_file(
    file_name,
    sigma = 1.0,
    out_txt = None,
    save = True,
    window_size = 9,
    detect_threshold = 20.0,
    damp = 0.2,
    camera_bg = 470,
    camera_gain = 110,
    max_iter = 20,
    plot = False,
    initial_guess = 'radial_symmetry',
    convergence_crit = 3.0e-4,
    divergence_crit = 1000.0,
    max_locs = 1000000,
    enforce_negative_definite = False,
    verbose = True,
    calculate_error = True,
    start_frame = None,
    stop_frame = None,
    frames_to_do = None,
    progress_bar = True,
):
    '''
    Detect and localize Gaussian spots in every frame of a single
    molecule tracking movie in either ND2 or TIF format.
    
    args
        file_name: str, a single ND2 or TIF file
        
        sigma: float, the expected standard deviation of the Gaussian spot

        out_txt: str, the name of a file to save the localizations, if desired.
            If a file path, will save the results to that file.
            If a directory, will save output files in that directory.
            If None, autogenerates the output file name.

        save: bool, save output to file
        
        window_size: int, the width of the window to use for spot detection
            and localization
        
        detect_threshold: float, the threshold in the log-likelihood image to 
            use when calling a spot
        
        damp: float, the factor by which to damp the update vector at each iteration
        
        camera_bg: float, the background level on the camera
        
        camera_gain: float, the grayvalues/photon conversion for the camera
        
        max_iter: int, the maximum number of iterations to execute before escaping
        
        plot: bool, show each step of the result for illustration
        
        initial_guess: str, the method to use for the initial guess. The currently
            implemented options are `radial_symmetry`, `centroid`, and `window_center`
        
        convergence: float, the criterion on the update vector for y and x when
            the algorithm can stop iterating
            
        divergence_crit: float, the criterion on the update vector for y and x
            when the algorithm should abandon MLE and default to a simpler method
            
        max_locs: int, the size of the localizations array to instantiate. This should
            be much greater than the number of expected localizations in the movie

        enforce_negative_definite : bool, whether to force the Hessian to be 
            negative definite by iteratively testing for negative definiteness
            by LU factorization, then subtracting successively larger ridge terms.
            If False, the method will only add ridge terms if numpy throws a
            linalg.linAlgError when trying to the invert the Hessian.

        verbose : bool, show the user the current progress 

    returns
        pandas.DataFrame with the localization results for this movie.
     
    '''
    # Make the file reader object
    reader = spazio.ImageFileReader(file_name)
    
    # Get the frame size and the number of frames
    N, M, n_frames = reader.get_shape()

    # Compute the kernels for spot detection
    g = utils.gaussian_model(sigma, window_size)
    gc = g - g.mean() 
    gaussian_kernel = utils.expand_window(gc, N, M)
    gc_rft = np.fft.rfft2(gaussian_kernel)

    # Compute some required normalization factors for spot detection
    n_pixels = window_size ** 2
    Sgc2 = (gc ** 2).sum()
    
    # Precompute some required factors for localization
    half_w = window_size // 2
    y_field, x_field = np.mgrid[:window_size, :window_size]
    sig2_2 = 2 * (sigma ** 2)
    sqrt_sig2_2 = np.sqrt(sig2_2)
    sig2_pi_2 = np.pi * sig2_2
    sqrt_sig2_pi_2 = np.sqrt(sig2_pi_2)
    
    # Initialize the vector of PSF model parameters. We'll
    # always use the following index scheme:
    #    pars[0]: y center of the PSF
    #    pars[1]: x center of the PSF
    #    pars[2]: I0, the number of photons in the PSF
    #    pars[3]: Ibg, the number of BG photons per pixel
    init_pars = np.zeros(4, dtype = 'float64')
    pars = np.zeros(4, dtype = 'float64')
    
    # Gradient of the log-likelihood w/ respect to each of the four parameters
    grad = np.zeros(4, dtype = 'float64')
    
    # Hessian of the log-likelihood
    H = np.zeros((4, 4), dtype = 'float64')
    
    # Store the results of localization in a large numpy.array. 
    if calculate_error:
        locs = np.zeros((max_locs, 14), dtype = 'float64')
        columns = [
            'detect_y_pixels',
            'detect_x_pixels',
            'y_pixels',
            'x_pixels',
            'I0',
            'bg',
            'error',
            'subwindow_variance',
            'frame_idx',
            'llr_detection',
            'err_y_pixels',
            'err_x_pixels',
            'err_I0',
            'err_bg',
        ]
    else:   
        locs = np.zeros((max_locs, 10), dtype = 'float64')
        columns = [
            'detect_y_pixels',
            'detect_x_pixels',
            'y_pixels',
            'x_pixels',
            'I0',
            'bg',
            'error',
            'subwindow_variance',
            'frame_idx',
            'llr_detection',
        ]

    # If start_frame is None, start at the first frame
    if start_frame == None:
        start_frame = 0

    # If stop_frame is None, stop at the last frame
    if stop_frame == None:
        stop_frame = n_frames - 1

    # Current localization index
    c_idx = 0
    
    # Iterate through the frames
    if type(frames_to_do) != type(np.array([])):
        frames_to_do = list(range(start_frame, stop_frame + 1))

    for frame_idx in tqdm(frames_to_do, disable = ~progress_bar):
        
        # Get the image corresponding to this frame from the reader
        image = reader.get_frame(frame_idx).astype('float64')
        
        # Perform the convolutions required for the LL detection test
        A = uniform_filter(image, window_size) * n_pixels
        B = uniform_filter(image**2, window_size) * n_pixels
        im_rft = np.fft.rfft2(image)
        C = np.fft.ifftshift(np.fft.irfft2(im_rft * gc_rft))

        # Calculate the likelihood of a spot in each pixel,
        # and set bad values to 1.0 (i.e. no chance of detection)
        L = 1 - (C**2) / (Sgc2*(B - (A**2)/float(n_pixels)))
        L[:half_w,:] = 1.0
        L[:,:half_w] = 1.0
        L[-(1+half_w):,:] = 1.0
        L[:,-(1+half_w):] = 1.0
        L[L <= 0.0] = 0.001

        # Calculate log likelihood of the presence of a spot
        LL = -(n_pixels / 2) * np.log(L)

        # Find pixels that pass the detection threshold
        detections = LL > detect_threshold 

        # For each detection that consists of adjacent pixels,
        # take the local maximum
        peaks = utils.local_max_2d(LL) & detections

        # Find the coordinates of the detections
        detected_positions = np.asarray(np.nonzero(peaks)).T + 1
        
        # Copy the detection information to the result array
        n_detect = detected_positions.shape[0]
        locs[c_idx : c_idx+n_detect, :2] = detected_positions.copy()
        
        # Save the frame index corresponding to these detections
        locs[c_idx : c_idx+n_detect, 8] = frame_idx 

        # Save the log-likelihood ratio of detection, which will become
        # useful in the tracking step
        locs[c_idx : c_idx+n_detect, 9] = LL[np.nonzero(peaks)]
        
        # For each detection, run subpixel localization
        for d_idx in range(n_detect):
            
            # Get a small subwindow surrounding that detection
            detect_y = detected_positions[d_idx, 0]
            detect_x = detected_positions[d_idx, 1]
            psf_image = (image[
                detect_y - half_w : detect_y + half_w + 1,
                detect_x - half_w : detect_x + half_w + 1
            ] - camera_bg) / camera_gain
            psf_image[psf_image < 0.0] = 0.0
        
            # If the image is not square (edge detection), set the error
            # column to True and move on
            if psf_image.shape[0] != psf_image.shape[1]:
                locs[d_idx, 6] = True 
                c_idx += 1
                continue


            # Make the initial parameter guesses.

            # Guess by radial symmetry (usually the best)
            if initial_guess == 'radial_symmetry':
                init_pars[0], init_pars[1] = radial_symmetry(psf_image)
                init_pars[3] = np.array([
                    psf_image[0,:-1].mean(),
                    psf_image[-1,1:].mean(),
                    psf_image[1:,0].mean(),
                    psf_image[:-1,-1].mean()
                ]).mean()

            # Guess by centroid method
            elif initial_guess == 'centroid':
                init_pars[0] = (y_field * psf_image).sum() / psf_image.sum()
                init_pars[1] = (x_field * psf_image).sum() / psf_image.sum()
                init_pars[3] = np.array([
                    psf_image[0,:-1].mean(),
                    psf_image[-1,1:].mean(),
                    psf_image[1:,0].mean(),
                    psf_image[:-1,-1].mean()
                ]).mean()

            # Simply place the guess in the center of the window
            elif initial_guess == 'window_center':
                init_pars[0] = window_size / 2
                init_pars[1] = window_size / 2
                init_pars[3] = psf_image.min()

            # Other initial guess methods are not implemented
            else:
                raise NotImplementedError

            # Use the model specification to guess I0
            max_idx = np.argmax(psf_image)
            max_y_idx = max_idx // window_size
            max_x_idx = max_idx % window_size 
            init_pars[2] = psf_image[max_y_idx, max_x_idx] - init_pars[3]
            E_y_max = 0.5 * (erf((max_y_idx + 0.5 - init_pars[0]) / sqrt_sig2_2) - \
                    erf((max_y_idx - 0.5 - init_pars[0]) / sqrt_sig2_2))
            E_x_max = 0.5 * (erf((max_x_idx + 0.5 - init_pars[1]) / sqrt_sig2_2) - \
                    erf((max_x_idx - 0.5 - init_pars[1]) / sqrt_sig2_2))
            init_pars[2] = init_pars[2] / (E_y_max * E_x_max)

            # If the I0 guess looks crazy (usually because there's a bright pixel
            # on the edge), make a more conservative guess by integrating the
            # inner ring of pixels
            if (np.abs(init_pars[2]) > 1000) or (init_pars[2] < 0.0):
                init_pars[2] = psf_image[half_w-1:half_w+2, half_w-1:half_w+2].sum()

            # Set the current parameter set to the initial guess. Hold onto the
            # initial guess so that if MLE diverges, we have a fall-back.
            pars[:] = init_pars.copy()

            # Keep track of the current number of iterations. When this exceeds
            # *max_iter*, then the iteration is terminated.
            iter_idx = 0

            # Continue iterating until the maximum number of iterations is reached, or
            # if the convergence criterion is reached
            update = np.ones(4, dtype = 'float64')
            while (iter_idx < max_iter) and any(np.abs(update[:2]) > convergence_crit):

                #Calculate the PSF model under the current parameter set
                E_y = 0.5 * (erf((y_field+0.5-pars[0])/sqrt_sig2_2) - \
                    erf((y_field-0.5-pars[0])/sqrt_sig2_2))
                E_x = 0.5 * (erf((x_field+0.5-pars[1])/sqrt_sig2_2) - \
                    erf((x_field-0.5-pars[1])/sqrt_sig2_2))
                model = pars[2] * E_y * E_x + pars[3]

                # Avoid divide-by-zero errors
                nonzero = (model > 0.0)

                # Calculate the derivatives of the model with respect
                # to each of the four parameters
                du_dy = ((pars[2] / (2 * sqrt_sig2_pi_2)) * (
                    np.exp(-(y_field-0.5-pars[0])**2 / sig2_2) - \
                    np.exp(-(y_field+0.5-pars[0])**2 / sig2_2)
                )) * E_x
                du_dx = ((pars[2] / (2 * sqrt_sig2_pi_2)) * (
                    np.exp(-(x_field-0.5-pars[1])**2 / sig2_2) - \
                    np.exp(-(x_field+0.5-pars[1])**2 / sig2_2)
                )) * E_y 
                du_dI0 = E_y * E_x 
                du_dbg = np.ones((window_size, window_size), dtype = 'float64')

                # Determine the gradient of the log-likelihood at the current parameter vector.
                # See the common structure of this term in section (1) of this notebook.
                J_factor = (psf_image[nonzero] - model[nonzero]) / model[nonzero]
                grad[0] = (du_dy[nonzero] * J_factor).sum()
                grad[1] = (du_dx[nonzero] * J_factor).sum()
                grad[2] = (du_dI0[nonzero] * J_factor).sum()
                grad[3] = (du_dbg[nonzero] * J_factor).sum()

                # Determine the Hessian. See the common structure of these terms
                # in section (1) of this notebook.
                H_factor = psf_image[nonzero] / (model[nonzero]**2)
                H[0,0] = (-H_factor * du_dy[nonzero]**2).sum()
                H[0,1] = (-H_factor * du_dy[nonzero]*du_dx[nonzero]).sum()
                H[0,2] = (-H_factor * du_dy[nonzero]*du_dI0[nonzero]).sum()
                H[0,3] = (-H_factor * du_dy[nonzero]).sum()
                H[1,1] = (-H_factor * du_dx[nonzero]**2).sum()
                H[1,2] = (-H_factor * du_dx[nonzero]*du_dI0[nonzero]).sum()
                H[1,3] = (-H_factor * du_dx[nonzero]).sum()
                H[2,2] = (-H_factor * du_dI0[nonzero]**2).sum()
                H[2,3] = (-H_factor * du_dI0[nonzero]).sum()
                H[3,3] = (-H_factor).sum()

                # Use symmetry to complete the Hessian.
                H[1,0] = H[0,1]
                H[2,0] = H[0,2]
                H[3,0] = H[0,3]
                H[2,1] = H[1,2]
                H[3,1] = H[1,3]
                H[3,2] = H[2,3]

                # Invert the Hessian. Here, we may need to stabilize the Hessian by adding
                # a ridge term. We'll increase this ridge as necessary until we can actually
                # invert the matrix.
                Y = np.diag([1.0e-5, 1.0e-5, 1.0e-5, 1.0e-5])
                if enforce_negative_definite:
                    while 1:
                        try:
                            pivots = get_pivots(H - Y)
                            if (pivots > 0.0).any():
                                Y *= 10
                                continue
                            else:
                                H_inv = np.linalg.inv(H - Y)
                                break
                        except (ZeroDivisionError, np.linalg.linalg.LinAlgError):
                            Y *= 10
                            continue
                else:
                    while 1:
                        try:
                            H_inv = np.linalg.inv(H - Y)
                            break
                        except (ZeroDivisionError, np.linalg.linalg.LinAlgError):
                            Y *= 10
                            continue

                # Get the update vector, the change in parameters
                update = -H_inv.dot(grad) * damp 

                # Update the parameters
                pars = pars + update 
                iter_idx += 1

                # Debugging
                if plot:
                    print('iter %d' % iter_idx)
                    fig, ax = plt.subplots(1, 2, figsize = (6, 3))
                    ax[0].imshow(psf_image)
                    ax[1].imshow(model)
                    print(pars[:2])
                    for j in range(2):
                        ax[j].plot(
                            [pars[1]],
                            [pars[0]],
                            color = 'k',
                            markersize = 10,
                            marker = '.',
                        )
                    plt.show(); plt.close()


            # If the estimate is diverging, fall back to the initial guess.
            if any(np.abs(update[:2]) >= divergence_crit):
                pars = init_pars
            if  ~check_pos_inside_window(pars[:2], window_size, edge_tolerance = 2):
                pars = init_pars
            else:
                # Correct for the half-pixel indexing error that results from
                # implicitly assigning intensity values to the corners of pixels
                # in the MLE method
                pars[:2] = pars[:2] + 0.5

            # Give the resulting y, x coordinates in terms of the whole image
            pars[0] = pars[0] + detect_y - half_w
            pars[1] = pars[1] + detect_x - half_w

            # Save the parameter vector for this localization
            locs[c_idx, 2:6] = pars.copy()

            # Save the image variance, which will become useful in the tracking
            # step
            locs[c_idx, 7] = psf_image.var()
                        
            # Update the number of current localizations
            c_idx += 1

            # Make an estimate for the error in the parameter estimation 
            # by inverting the observed information matrix, acquired
            # from the Jacobian
            if calculate_error:

                # Jacobian: 4-by-n_pixels matrix of the derivatives
                # of each model parameter at each pixel
                jac = np.array((du_dy, du_dx, du_dI0, du_dbg))
                J_factor = (psf_image[nonzero] - model[nonzero]) / model[nonzero]
                _grad = (jac[:,nonzero] * J_factor).sum(1)
                jac_div_sqrt_model = jac[:,nonzero] / np.sqrt(model[nonzero])

                # Observed information matrix I
                I = jac_div_sqrt_model.dot(jac_div_sqrt_model.T)
                try:
                    err = np.diagonal(np.linalg.inv(I))
                except np.linalg.LinAlgError as e:
                    err = np.zeros(4)

                locs[c_idx, 10:14] = np.sqrt(err)
        
    # Truncate the result to the actual number of localizations
    locs = locs[:c_idx, :]
        
    # Format the result as a pandas.DataFrame, and enforce some typing
    df_locs = pd.DataFrame(locs, columns = columns)

    df_locs['detect_y_pixels'] = df_locs['detect_y_pixels'].astype('uint16')
    df_locs['detect_x_pixels'] = df_locs['detect_x_pixels'].astype('uint16')
    df_locs['error'] = df_locs['error'].astype('bool')
    df_locs['frame_idx'] = df_locs['frame_idx'].astype('uint16')

    # Save metadata associated with this file
    metadata = {
        'N' : N,
        'M' : M,
        'n_frames' : n_frames,
        'window_size' : window_size,
        'localization_method' : 'mle_gaussian',
        'sigma' : sigma,
        'camera_bg' : camera_bg,
        'camera_gain' : camera_gain,
        'max_iter' : max_iter,
        'initial_guess' : initial_guess,
        'convergence_crit' : convergence_crit,
        'divergence_crit' : divergence_crit,
        'enforce_negative_definite' : enforce_negative_definite,
        'damp' : damp,
        'detect_threshold' : detect_threshold,
        'file_type' : '.nd2',
        'image_file' : file_name,
        'image_file_path' : os.path.abspath(file_name),
    }

    # Close the reader
    reader.close()

    # If desired, save to a file
    if save:
        if type(out_txt) == type(''):
            if '/' in out_txt:
                out_prefix = '/'.join(out_txt.split('/')[:-1])
            else:
                out_prefix = './'
            if not os.path.isdir(out_prefix):
                raise RuntimeError('detect_and_localize_file: could not find out_txt directory %s' % out_prefix)
        else:
            if '.nd2' in file_name:
                _out_f = '%s.locs' % file_name.replace('.nd2', '')
            elif '.tif' in file_name:
                _out_f = '%s.locs' % file_name.replace('.tif', '')
            elif '.tiff' in file_name:
                _out_f = '%s.locs' % file_name.replace('.tiff', '')
            if type(out_txt) == type('') and os.path.isdir(out_txt):
                out_txt = '%s/%s' % (out_txt, _out_f)
            else:
                out_txt = _out_f
        spazio.save_locs(out_txt, df_locs, metadata)

    return df_locs, metadata 

def detect_and_localize_file_parallelized(
    file_name,
    dask_client,
    out_txt = None,
    verbose_times = False,
    **kwargs,
):
    '''
    Run parallelelized detection and localization on a single
    image.

    args
        file_name       :   str, image filename
        dask_client     :   dask.distributed.Client, the parallelizer
        out_txt         :   str, file to save to
        kwargs          :   for detect_and_localize_file()

    returns
        (
            pandas.DataFrame, the localizations;
            dict; the metadata;
        )

    '''
    # Get the number of workers for this dask client
    n_workers = len(dask_client.scheduler_info()['workers'])

    kwargs['verbose'] = False
    kwargs['progress_bar'] = False
    kwargs['start_frame'] = None
    kwargs['stop_frame'] = None 
    kwargs['save'] = False 

    reader = spazio.ImageFileReader(file_name)
    N, M, n_frames = reader.get_shape()
    reader.close()

    # Divide the list of all frames into ranges that will be
    # given to each individual worker
    frames = np.arange(n_frames)
    frame_ranges = [frames[i::n_workers] for i in range(n_workers)]

    # Assign each frame range to a worker
    results = []
    for frame_range_idx, frame_range in enumerate(frame_ranges):
        kwargs['frames_to_do'] = frame_range 
        result_df_metadata = dask.delayed(detect_and_localize_file)(file_name, **kwargs)
        results.append(result_df_metadata)

    t0 = time.time()
    out_tuples = dask_client.compute(results)
    out_tuples = [i.result() for i in out_tuples]
    t1 = time.time()
    if verbose_times: print('Run time: %.2f sec' % (t1 - t0))

    metadata = out_tuples[0][1]
    locs = pd.concat(
        [out_tuples[i][0] for i in range(len(out_tuples))],
        ignore_index = True, sort = False,
    )
    locs = locs.sort_values(by = 'frame_idx')
    locs.index = np.arange(len(locs))

    if out_txt != None:
        spazio.save_locs(out_txt, locs, metadata)

    return locs, metadata

def check_pos_inside_window(pos_vector, window_size, edge_tolerance = 3):
    '''
    Returns True if the vector is inside the window
    and False otherwise.

    '''
    return ~((pos_vector < edge_tolerance).any() or \
        (pos_vector > window_size-edge_tolerance).any())

def detect_and_localize_directory(
    directory_name,
    out_dir = None,
    sigma = 1.0,
    window_size = 9,
    detect_threshold = 20.0,
    damp = 0.2,
    camera_bg = 470,
    camera_gain = 110,
    max_iter = 20,
    initial_guess = 'radial_symmetry',
    convergence_crit = 3.0e-4,
    divergence_crit = 1.0,
    max_locs = 2000000,
    enforce_negative_definite = False,
    verbose = False,
):
    '''
    Detect and localize Gaussian spots in every ND2 file found
    in the directory *directory_name*.
    
    args
        directory_name: str
        
        out_dir: str, location to put output files. If None, these are placed
            in the same directory as the ND2 files

        sigma: float, the expected standard deviation of the Gaussian spot
        
        window_size: int, the width of the window to use for spot detection
            and localization
        
        detect_threshold: float, the threshold in the log-likelihood image to 
            use when calling a spot
        
        damp: float, the factor by which to damp the update vector at each iteration
        
        camera_bg: float, the background level on the camera
        
        camera_gain: float, the grayvalues/photon conversion for the camera
        
        max_iter: int, the maximum number of iterations to execute before escaping
        
        plot: bool, show each step of the result for illustration
        
        initial_guess: str, the method to use for the initial guess. The currently
            implemented options are `radial_symmetry`, `centroid`, and `window_center`
        
        convergence: float, the criterion on the update vector for y and x when
            the algorithm can stop iterating
            
        divergence_crit: float, the criterion on the update vector for y and x
            when the algorithm should abandon MLE and default to a simpler method
            
        max_locs: int, the size of the localizations array to instantiate. This should
            be much greater than the number of expected localizations in the movie

        enforce_negative_definite : bool, whether to force the Hessian to be 
            negative definite by iteratively testing for negative definiteness
            by LU factorization, then subtracting successively larger ridge terms.
            If False, the method will only add ridge terms if numpy throws a
            linalg.linAlgError when trying to the invert the Hessian.

        verbose : bool, show the user the current progress

    '''
    file_list = glob("%s/*.nd2" % directory_name)

    # Construct the output locations
    if out_dir == None:
        out_txt_list = [fname.replace('.nd2', '.locs') for fname in file_list]
    else:
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        out_txt_list = ['%s/%s' % (out_dir, fname.split('/')[-1].replace('.nd2', '.locs')) for fname in file_list]

    for f_idx, fname in enumerate(file_list):
        if verbose: print('Localizing %s...' % fname)
        out_df = detect_and_localize_file(
            fname,
            sigma = sigma,
            out_txt = out_txt_list[f_idx],
            window_size = window_size,
            detect_threshold = detect_threshold,
            damp = damp,
            camera_bg = camera_bg,
            camera_gain = camera_gain,
            max_iter = max_iter,
            initial_guess = initial_guess,
            convergence_crit = convergence_crit,
            divergence_crit = divergence_crit,
            max_locs = max_locs,
            enforce_negative_definite = enforce_negative_definite,
            verbose = verbose,
        )

def detect_and_localize_directory_parallelized(
    directory_name,
    dask_client,
    out_dir = None,
    verbose_times = True,
    suffixes_to_seek = ['.nd2'],
    **kwargs
):
    '''
    Parallelized localization of spots in all image files
    in the passed directory. Out file names are automatically
    generated.

    args
        directory_name          :   str, directory with image files
        dask_client             :   dask.distributed.Client, the 
                                        parallelizer
        out_dir                 :   str, directory in which to save files
        verbose_times           :   bool, show the time to localize each file
        suffixes_to_seek        :   list of str, the suffixes of the image
                                        files (e.g. ['.nd2', '.tif'])
        **kwargs                :   keyword arguments for detect_and_localize_file

    returns
        None

    '''
    # Get the list of input image filenames
    file_list = []
    for suffix in suffixes_to_seek:
        file_list = file_list + glob("%s/*%s" % (directory_name, suffix))

    # Construct the output locations
    if out_dir == None:
        out_txt_list = ['%s.locs' % os.path.splitext(fname)[0] \
            for fname in file_list]
    else:
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        out_txt_list = ['%s/%s' % (out_dir, fname.split('/')[-1].replace('.nd2', '.locs')) \
            for fname in file_list]

    # Run detection/localization on each target file
    for f_idx, f_name in enumerate(file_list):
        df, metadata = detect_and_localize_file_parallelized(
            f_name,
            dask_client,
            out_txt = out_txt_list[f_idx],
            verbose_times = verbose_times,
            **kwargs,
        )
        if verbose_times: print('Finished with %s' % f_name)
# 
# More customized particle detection functions
#

def _detect_with_psf_kernel(
    image,
    psf_kernel_rft,
    Sgc2,
    detect_threshold = 20.0,
    window_size = 9,
):
    '''
    Run detection with a custom PSF kernel. Takes the FT
    of the kernel rather than the kernel itself to avoid
    having to take the FT at each iteration.

    args
        image           :   2D ndarray, real image

        psf_kernel_rft  :   2D ndarray of type complex128,
                            the real Fourier transform of the
                            PSF kernel

        Sgc2            :   float, the sum of squares for the
                            real PSF kernel

        detect_threshold:   float

        window_size     :   int

    returns
        2D ndarray, the y and x coordinates of each detection

    '''
    # Take the Fourier transform of the image
    image = image.astype('float64')
    image_rft = np.fft.rfft2(image)

    # Make sure the shapes of the FTs match
    assert (image_rft.shape == psf_kernel_rft.shape)

    # Take the convolutions necessary for detection
    n_pixels = window_size ** 2
    half_w = window_size // 2
    A = uniform_filter(image, window_size) * n_pixels
    B = uniform_filter(image**2, window_size) * n_pixels
    C = np.fft.ifftshift(np.fft.irfft2(image_rft * psf_kernel_rft))

    # Take likelihood ratio for presence of a spot
    L = 1 - (C**2) / (Sgc2*(B - (A**2)/float(n_pixels)))
    L[:half_w,:] = 1.0
    L[:,:half_w] = 1.0
    L[-half_w:,:] = 1.0
    L[:,-half_w:] = 1.0
    L[L <= 0.0] = 0.001

    # Take log-likelihood
    LL = -(n_pixels / 2) * np.log(L)

    # Find pixels that pass the detection threshold
    detections = LL > detect_threshold

    # For each detection that consists of adjacent pixels,
    # take the local maximum
    peaks = utils.local_max_2d(LL) & detections

    # Find coordinates of detections
    detected_positions = np.asarray(np.nonzero(peaks)).T 

    return detected_positions 

def _detect_gaussian_kernel(
    image,
    sigma = 1.0,
    detect_threshold = 20.0,
    window_size = 9,
    offset_by_half = False,
):
    '''
    Run detection with a Gaussian kernel of custom width.

    args
        image           :   2D ndarray, real image
        sigma           :   float, width of Gaussian in pixels
        detect_threshold:   float
        window_size     :   int
        offset_by_half  :   bool, offset the PSF kernel by 0.5
                            pixels

    returns
        LL, detections, peaks, detected_positions


    '''
    # Take FT of the image
    image = image.astype('float64')
    N, M = image.shape 
    image_rft = np.fft.rfft2(image)

    # Make the PSF kernel
    g = utils.gaussian_model(sigma, window_size, offset_by_half = offset_by_half)
    gc = g - g.mean() 
    gaussian_kernel = utils.expand_window(gc, N, M)
    gc_rft = np.fft.rfft2(gaussian_kernel)

    # Compute some factors
    half_w = window_size // 2
    n_pixels = window_size ** 2
    Sgc2 = (gc ** 2).sum()

    # Run the convolutions for detection
    A = uniform_filter(image, window_size) * n_pixels
    B = uniform_filter(image**2, window_size) * n_pixels
    C = np.fft.ifftshift(np.fft.irfft2(image_rft * gc_rft))

    # Take likelihood ratio for presence of a spot
    L = 1 - (C**2) / (Sgc2*(B - (A**2)/float(n_pixels)))
    L[:half_w,:] = 1.0
    L[:,:half_w] = 1.0
    L[-half_w:,:] = 1.0
    L[:,-half_w:] = 1.0
    L[L <= 0.0] = 0.001

    # Take log-likelihood
    LL = -(n_pixels / 2) * np.log(L)

    # Find pixels that pass the detection threshold
    detections = LL > detect_threshold

    # For each detection that consists of adjacent pixels,
    # take the local maximum
    peaks = utils.local_max_2d(LL) & detections

    # Find coordinates of detections
    detected_positions = np.asarray(np.nonzero(peaks)).T 

    return LL, detections, peaks, detected_positions 

def _detect_dog_filter(
    image,
    bg_kernel_width = 10,
    bg_sub_mag = 1.0,
    spot_kernel_width = 1,
    threshold = 20.0,
):
    image = image.astype('float64')
    image_bg = ndi.gaussian_filter(image, bg_kernel_width)
    image_bg_sub = image - image_bg * bg_sub_mag 
    image_bg_sub[image_bg_sub < 0] = 0
    image_filt = ndi.gaussian_filter(image_bg_sub, spot_kernel_width)
    detections = image_filt > threshold
    peaks = utils.local_max_2d(image) & detections 
    detected_positions = np.asarray(np.nonzero(peaks)).T 

    return image_bg, image_bg_sub, image_filt, detections, peaks, detected_positions 

def _detect_log_filter(
    image,
    bg_kernel_width = 10,
    bg_sub_mag = 1.0,
    spot_kernel_width = 2.0,
    threshold = 20.0,
):
    image = image.astype('float64')
    image_bg = ndi.gaussian_filter(image, bg_kernel_width)
    image_bg_sub = image - image_bg * bg_sub_mag 
    image_bg_sub[image_bg_sub < 0] = 0
    image_filt = -ndi.laplace(ndi.gaussian_filter(image_bg_sub, spot_kernel_width))
    detections = image_filt > threshold
    peaks = utils.local_max_2d(image) & detections 
    detected_positions = np.asarray(np.nonzero(peaks)).T 

    return image_filt, detections, peaks, detected_positions 




