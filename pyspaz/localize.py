'''
localize.py
'''
# Convenient file finder
from glob import glob 

# Get numerical essentials
import numpy as np

# For convolutions in the detection step
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

# The radial symmetry method involves a possible divide-by-zero that
# is subsequently corrected. The divide-by-zero warnings are 
# temporarily disabled for this step.
import warnings

# Opening the result files in the test_detect() and test_localize() programs
import os

# Utility functions for localization
from . import utils 

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

def detect_and_localize_file(
    file_name,
    sigma = 1.0,
    out_txt = None,
    out_dir = None,
    window_size = 9,
    detect_threshold = 20.0,
    damp = 0.2,
    camera_bg = 0,
    camera_gain = 1,
    max_iter = 10,
    plot = False,
    initial_guess = 'radial_symmetry',
    convergence_crit = 3.0e-3,
    divergence_crit = 1.0,
    max_locs = 1000000,
    enforce_negative_definite = False,
    verbose = True,
):
    '''
    Detect and localize Gaussian spots in every frame of a single
    molecule tracking movie in either ND2 or TIF format.
    
    args
        file_name: str, a single ND2 or TIF file
        
        sigma: float, the expected standard deviation of the Gaussian spot

        out_txt: str, the name of a file to save the localizations, if desired

        out_dir: str, location to put output files. If None, these are placed
            in the same directory as the ND2 files
        
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
    uniform_kernel = utils.expand_window(
        np.ones((window_size, window_size)),
        N, M,
    )

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
    locs = np.zeros((max_locs, 10), dtype = 'float64')
    
    # Current localization index
    c_idx = 0
    
    # Iterate through the frames
    for frame_idx in tqdm(range(n_frames)):
        
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
        L[:4,:] = 1.0
        L[:,:4] = 1.0
        L[-4:,:] = 1.0
        L[:,-4:] = 1.0
        L[L <= 0.0] = 0.001

        # Calculate log likelihood of the presence of a spot
        LL = -(n_pixels / 2) * np.log(L)

        # Find pixels that pass the detection threshold
        detections = LL > detect_threshold 

        # For each detection that consists of adjacent pixels,
        # take the local maximum
        peaks = utils.local_max_2d(LL) & detections

        # Find the coordinates of the detections
        detected_positions = np.asarray(np.nonzero(peaks)).T 
        
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

            # If the estimate is diverging, fall back to the initial guess.
            if any(np.abs(update[:2]) >= divergence_crit) or \
                ~check_pos_inside_window(pars[:2], window_size, edge_tolerance = 2) or \
                (pars[2] > 1000):
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
        
    # Truncate the result to the actual number of localizations
    locs = locs[:c_idx, :]
        
    # Format the result as a pandas.DataFrame, and enforce some typing
    df_locs = pd.DataFrame(locs, columns = [
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
    ])
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
    }

    # If desired, save to a file
    if out_txt != None:
        if out_dir != None:
            if not os.path.isdir(out_dir):
                os.mkdir(out_dir)
            out_txt = "%s/%s" % (out_dir, out_txt)
        spazio.save_locs(out_txt, df_locs, metadata)

    return df_locs

def detect_and_localize_file_parallelized(
    file_name,
    n_workers = 8,
    n_threads = 8,
    sigma = 1.0,
    out_txt = None,
    out_dir = None,
    window_size = 9,
    detect_threshold = 20.0,
    damp = 0.2,
    camera_bg = 0,
    camera_gain = 1,
    max_iter = 10,
    plot = False,
    initial_guess = 'radial_symmetry',
    convergence_crit = 3.0e-3,
    divergence_crit = 1.0,
    max_locs = 1000000,
    enforce_negative_definite = False,
    verbose = True,
):
    raise NotImplementedError 

def check_pos_inside_window(pos_vector, window_size, edge_tolerance = 2):
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
    max_iter = 10,
    initial_guess = 'radial_symmetry',
    convergence_crit = 1.0e-3,
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
            out_dir = None,
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



