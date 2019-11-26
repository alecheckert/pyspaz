'''
interpolate_mask.py -- utilities for doing two-dimensional
interpolation between masks

'''
# Numerical stuff
import numpy as np 
from scipy.spatial import distance_matrix 
from scipy import ndimage as ndi 

# DataFrames
import pandas as pd 

# Basic I/O
import os
import sys
import tifffile

# Plotting
import matplotlib.pyplot as plt 

# Path object, for quick tests of mask inclusion
from matplotlib.path import Path 

# pyspaz functions
from pyspaz import spazio
from pyspaz import mask 

# Hungarian algorithm, for tracking masks
from munkres import Munkres 
munkres_solver = Munkres()

# For the spline-based mask interpolation
from scipy.interpolate import splrep, splev

def label_and_interpolate_masks(
    mask_tif,
    frame_interval,
    mask_upsampling_factor = 1.0,
    mask_size_limit = 3000,
    tolerance_radius = 10,
    interpolation_type = 'linear',
):
    '''
    Given a set of binary masks defined in a TIF
    file, construct interpolators for each mask
    that can be used to test whether localizations
    or trajectories are inside the mask.

    args
        mask_tif                :   str, TIF file with the binary masks.
                                    The encoded array should be shape (n_intervals,
                                    N, M), where [m, :, :] returns the mask
                                    definitions at the m^th interval.
                                    
                                    All nonzero pixels are considered to encode
                                    masks.

        frame_interval          :   int, the number of frames in the original
                                    movie between the mask definitions

        mask_upsampling_factor  :   float, the number of mask pixels per
                                    real camera pixel (in case masks are 
                                    defined on a PALM reconstruction, for 
                                    instance)

        mask_size_limit         :   int, the maximum number of edge points
                                    to include for each mask. 3000 is tractable
                                    for most purposes.

        tolerance_radius        :   float, the search radius for edge point
                                    reconnection in mask reconstruction

        interpolation_type      :   str, either 'linear' or 'spline'. Note
                                    that spline interpolation is only meaningful
                                    when masks are defined on at least 3 
                                    separate frames (e.g. labeled_stack.shape[0] >= 3)

    returns
        list of LinearMaskInterpolator or SplineMaskInterpolator objects,
            interpolators for each mask. These have the methods

                LinearMaskInterpolator.interpolate(frame_idx)
                SplineMaskInterpolator.interpolator(frame_idx)

        which return a 2D ndarray of shape (n_points, 2), the mask interpolation
            at the desired frame.

    '''
    # Get the set of masks
    spazio.check_file_exists(mask_tif)
    binary_stack = (tifffile.imread(mask_tif) > 0).astype('uint16')
    assert len(binary_stack.shape) == 3

    # Label the masks: assign each separable mask a unique index
    labeled_stack, n_labels = label_stack(binary_stack)

    # Make the interpolator objects
    interpolators = interpolate_masks(
        labeled_stack,
        frame_interval,
        mask_upsampling_factor = mask_upsampling_factor,
        mask_size_limit = mask_size_limit,
        tolerance_radius = tolerance_radius,
        interpolation_type = interpolation_type,
    )

    # Return a list of LinearMaskInterpolator or SplineMaskInterpolator
    # objects
    return interpolators

def interpolate_masks(
    labeled_stack,
    frame_interval,
    mask_upsampling_factor = 1.0,
    mask_size_limit = 3000,
    tolerance_radius = 30,
    interpolation_type = 'linear',
):
    '''
    Given a set of masks defined in a labeled image stack,
    track the masks between frames and make an interpolator
    for each mask trajectory. This interpolator can be used
    to reconstruct the mask at any intervening frame.

    args
        labeled_stack           :   3D ndarray of shape (n_intervals,
                                    N, M), the labeled masks. "Labeled"
                                    means that each mask should have a 
                                    unique index (e.g. 1, 2, 3, ...) 
                                    corresponding to the pixels with mask
                                    membership. Outside of masks, the image
                                    should be 0.

        frame_interval          :   int, the number of frames in the original
                                    movie between the mask definitions

        mask_upsampling_factor  :   float, the number of mask pixels per
                                    real camera pixel (in case masks are 
                                    defined on a PALM reconstruction, for 
                                    instance)

        mask_size_limit         :   int, the maximum number of edge points
                                    to include for each mask. 3000 is tractable
                                    for most purposes.

        tolerance_radius        :   float, the search radius for edge point
                                    reconnection in mask reconstruction

        interpolation_type      :   str, either 'linear' or 'spline'. Note
                                    that spline interpolation is only meaningful
                                    when masks are defined on at least 3 
                                    separate frames (e.g. labeled_stack.shape[0] >= 3)

    returns
        list of LinearMaskInterpolator or SplineMaskInterpolator objects,
            interpolators for each mask. These have the methods

                LinearMaskInterpolator.interpolate(frame_idx)
                SplineMaskInterpolator.interpolator(frame_idx)

        which return a 2D ndarray of shape (n_points, 2), the mask interpolation
            at the desired frame.

    '''
    # Check user inputs
    assert len(labeled_stack.shape) == 3

    interpolators = []

    # for frame_idx in range(labeled_stack.shape[0]):
    #     plt.imshow(labeled_stack[frame_idx, :, :], cmap = 'gray')
    #     plt.show(); plt.close()

    # If the problem has a single interval, then make a static interpolator
    if labeled_stack.shape[0] == 1:
        labels = [i for i in np.unique(labeled_stack) if i != 0]
        for label in labels:
            label_image = (labeled_stack == label)[0,:,:]
            mask_edge = mask.mask_to_oriented_edge(
                label_image,
                mask_upsampling_factor = mask_upsampling_factor,
                tolerance_radius = tolerance_radius,
            )
            mask_edges = np.zeros((1, mask_edge.shape[0], 2), dtype = mask_edge.dtype)
            mask_edges[0,:,:] = mask_edge 
            interpolator = LinearMaskInterpolator(mask_edges, [0], static = True)
            interpolators.append(interpolator)

    # Otherwise, reconnect the masks in separate frames into trajectories
    else:
        mask_trajs = track_masks(labeled_stack)
        interpolators = []
        for mask_traj in mask_trajs:

            # Binary array with True indicating mask membership for each frame
            mask_stack = (sum([(labeled_stack == i) for i in \
                mask_traj.labels]) > 0)[mask_traj.intervals]

            # The frames at which each 2D mask is defined
            mask_frame_indices = np.array(mask_traj.intervals) * frame_interval 

            # Reconnect the edges of each mask
            mask_edges = match_mask_coords(
                mask_stack,
                tolerance_radius = tolerance_radius,
                mask_upsampling_factor = mask_upsampling_factor,
                mask_size_limit = mask_size_limit,
            )

            # Make an interpolator corresponding to this mask
            if interpolation_type == 'linear':
                interpolator = LinearMaskInterpolator(mask_edges, mask_frame_indices)
                interpolators.append(interpolator)
            elif interpolation_type == 'spline':
                interpolator = SplineMaskInterpolator(mask_edges, mask_frame_indices)
                interpolators.append(interpolator)
            else:
                raise NotImplementedError

    return interpolators

class LinearMaskInterpolator():
    '''
    Construct 2D masks, using linear interpolation between masks
    defined in adjacent frames.

    If the mask is only defined in a single frame, then constructs
    a ``static'' interpolator, which just returns a single mask for
    any frame_idx argument.

    initialization
        mask_edges      :   3D ndarray of shape (n_frames, n_points, 2), the
                            edges of the mask at each frame interval

        mask_frames     :   list of int of shape (n_frames,), the frame
                            indices corresponding to each interval. For instance,
                            mask_frames = [0, 10000, 20000] would mean that 
                            we start with three masks defined at frames 0, 
                            10000, and 20000. 

    methods
        interpolate(frame_idx): interpolate the mask at frame *frame_idx*.
            If this frame is outside the interpolation range, returns a zero
            path.

    '''
    def __init__(
        self,
        mask_edges,
        mask_frames,
        static = False,
    ):
        # Assign user inputs
        self.mask_edges = mask_edges 
        self.mask_frames = mask_frames
        self.static = static 
        self.min_frame = mask_frames[0]
        self.max_frame = mask_frames[-1]
        n_frames, N, M = mask_edges.shape
        self.n_frames = n_frames 
        self.N = N 
        self.M = M 
        self.n_points = mask_edges.shape[1]
        assert self.n_frames == len(mask_frames)

        # If only a single frame, then the interpolator is a simple static 2D shape
        if self.n_frames == 1:
            self.static = True 

        # Otherwise make a real interpolator
        else:
            # slopes[mask_idx, :, :] corresponds to the 2D slopes for the interpolation
            # between mask_idx and mask_idx+1
            self.slopes = np.zeros(
                (self.n_frames-1, self.n_points, 2), 
                dtype = mask_edges.dtype
            )
            for frame_idx in range(self.n_frames-1):
                self.slopes[frame_idx] = (self.mask_edges[frame_idx+1, :, :] - \
                    self.mask_edges[frame_idx, :, :]) / (self.mask_frames[frame_idx+1] - \
                        self.mask_frames[frame_idx])

    def interpolate(self, frame_idx):
        '''
        Interpolate a 2D mask edge.

        args
            frame_idx               :   int

        returns
            2D ndarray of shape (self.n_points, 2), the interpolated
                mask edge at the desired frame

        '''
        # If static, return the same mask edge for any frame
        if self.static:
            return self.mask_edges[0,:,:]

        # If desired frame is outside the interpolation range,
        # return all zeroes
        if (frame_idx < self.min_frame) or (frame_idx > self.max_frame):
            return np.zeros((self.N, self.M))

        # Find the most recent mask
        mask_idx = np.digitize(frame_idx, self.mask_frames) - 1
        if mask_idx == self.n_frames-1:
            return self.mask_edges[mask_idx,:,:]

        # Get the number of frames since the most recent mask
        delta_frames = frame_idx - self.mask_frames[mask_idx]

        return self.mask_edges[mask_idx, :, :] + self.slopes[mask_idx, :, :] * delta_frames

    def make_path(self, frame_idx):
        '''
        Make a matplotlib.Path object corresponding to the 
        edge of this mask at the desired frame. This has
        a *contains_points* method that can be used to 
        quickly determine whether a set of input points
        are inside the 2D mask.

        args
            frame_idx           :   int

        returns
            matplotlib.Path, the path corresponding to the mask
                edge at that frame index

        '''
        return Path(self.interpolate(frame_idx), closed = True)

    def contains_points(self, points, frame_idx):
        '''
        Determine whether each of a set of 2D points
        lie within the interpolated mask.

        args
            points      :   2D ndarray of shape (m, 2), the
                            yx coordinates of each query point

            frame_idx   :   int

        returns
            bool 1D ndarray of shape (m,), the mask membership
                of each point

        '''
        m = points.shape[0]
        mask_edge = self.interpolate(frame_idx)
        if (mask_edge == 0).all():
            return np.zeros(m, dtype = 'bool')
        mask_path = Path(mask_edge, closed = True)
        return mask_path.contains_points(points)

    def assign_locs(self,
        locs,
        mask_col_name,
        frame_col = 'frame_idx',
        y_col = 'y_pixels',
        x_col = 'x_pixels'
    ):
        '''
        Assign a set of localizations to this mask.
        Modifies the input dataframe in place.

        args
            locs            :   pandas.DataFrame

            mask_col_name   :   str, new column to create in *locs* 
                                    with mask membership (bool)

            frame_col       :   str
            y_col           :   str
            x_col           :   str  

        returns
            None

        '''
        inside = np.zeros(len(locs), dtype = 'bool')
        for frame_idx, frame_locs in locs.groupby(frame_col):
            if (frame_idx < self.min_frame) or (frame_idx > self.max_frame):
                continue 
            mask_path = self.make_path(frame_idx)
            inside[frame_locs.index] = mask_path.contains_points(frame_locs[[y_col, x_col]])
        locs[mask_col_name] = inside


class SplineMaskInterpolator():
    '''
    Construct for 2D masks, using spline interpolation between masks
    defined in adjacent frames.

    Note that masks must be defined on at least three intervals
    for spline interpolation to be valid.

    initialization
        mask_edges      :   3D ndarray of shape (n_frames, n_points, 2), the
                            edges of the mask at each frame interval

        mask_frames     :   list of int of shape (n_frames,), the frame
                            indices corresponding to each interval. For instance,
                            mask_frames = [0, 10000, 20000] would mean that 
                            we start with three masks defined at frames 0, 
                            10000, and 20000. 

    methods
        interpolate(frame_idx): interpolate the mask at frame *frame_idx*.
            If this frame is outside the interpolation range, returns a zero
            path.

    '''
    def __init__(
        self,
        mask_edges,
        mask_frames,
        static = False,
    ):
        # Assign user inputs
        self.mask_edges = mask_edges
        self.mask_frames = mask_frames 
        self.static = static

        self.min_frame = mask_frames[0]
        self.max_frame = mask_frames[-1]
        n_frames, N, M = mask_edges.shape 
        self.n_frames = n_frames 
        self.N = N 
        self.M = M 
        self.n_points = mask_edges.shape[1]
        assert self.n_frames == len(mask_frames)

        # If only a single frame, make a static interpolator
        # (returns the same path for any frame argument)
        if self.n_frames == 1:
            self.static = True 

        # Otherwise make a real interpolator
        else:
            k = min([self.n_frames-1, 3])
            self.point_interpolators = []
            for point_idx in range(self.n_points):
                traj = mask_edges[:, point_idx, :]
                tck_y = splrep(self.mask_frames, traj[:,0], k = k)
                tck_x = splrep(self.mask_frames, traj[:,1], k = k)
                self.point_interpolators.append((tck_y, tck_x))

    def interpolate(self, frame_idx):
        '''
        Interpolate a 2D mask edge.

        args
            frame_idx               :   int

        returns
            2D ndarray of shape (self.n_points, 2), the interpolated
                mask edge at the desired frame

        '''
        if (frame_idx < self.min_frame) or (frame_idx > self.max_frame):
            return np.zeros((self.n_points, 2), dtype = 'float64')
        elif frame_idx == self.max_frame:
            return self.mask_edges[-1, :, :]
        else:
            out = np.zeros((self.n_points, 2), dtype = 'float64')
            for point_idx, point_interpolator in enumerate(self.point_interpolators):
                out[point_idx, 0] = splev(frame_idx, point_interpolator[0])
                out[point_idx, 1] = splev(frame_idx, point_interpolator[1])
            return out 

    def make_path(self, frame_idx):
        '''
        Make a matplotlib.Path object corresponding to the 
        edge of this mask at the desired frame. This has
        a *contains_points* method that can be used to 
        quickly determine whether a set of input points
        are inside the 2D mask.

        args
            frame_idx           :   int

        returns
            matplotlib.Path, the path corresponding to the mask
                edge at that frame index

        '''
        return Path(self.interpolate(frame_idx), closed = True)

    def contains_points(self, points, frame_idx):
        '''
        Determine whether each of a set of 2D points
        lie within the interpolated mask.

        args
            points      :   2D ndarray of shape (m, 2), the
                            yx coordinates of each query point

            frame_idx   :   int

        returns
            bool 1D ndarray of shape (m,), the mask membership
                of each point

        '''
        m = points.shape[0]
        mask_edge = self.interpolate(frame_idx)
        if (mask_edge == 0).all():
            return np.zeros(m, dtype = 'bool')
        mask_path = Path(mask_edge, closed = True)
        return mask_path.contains_points(points)

    def assign_locs(self,
        locs,
        mask_col_name,
        frame_col = 'frame_idx',
        y_col = 'y_pixels',
        x_col = 'x_pixels'
    ):
        '''
        Assign a set of localizations to this mask.
        Modifies the input dataframe in place.

        args
            locs            :   pandas.DataFrame

            mask_col_name   :   str, new column to create in *locs* 
                                    with mask membership (bool)

            frame_col       :   str
            y_col           :   str
            x_col           :   str  

        returns
            None

        '''
        inside = np.zeros(len(locs), dtype = 'bool')
        for frame_idx, frame_locs in locs.groupby(frame_col):
            if (frame_idx < self.min_frame) or (frame_idx > self.max_frame):
                continue 
            mask_path = self.make_path(frame_idx)
            inside[frame_locs.index] = mask_path.contains_points(frame_locs[[y_col, x_col]])
        locs[mask_col_name] = inside

def track_masks(labeled_stack):
    '''
    Given a labeled image stack, reconnect masks between frames.

    args
        binary_stack            :   3D ndarray of shape (n_intervals,
                                    N, M), with labels corresponding
                                    to individual masks and 0 in between
                                    masks

    returns
        list of _MaskTraj objects, the mask trajectories. Each _MaskTraj
            has an attribute *labels* that contains all of the masks to 
            which the 


    '''
    assert len(labeled_stack.shape) == 3
    n_intervals = labeled_stack.shape[0]
    n_labels = labeled_stack.max()

    mask_trajs = []
    completed_mask_trajs = []

    # Get the positions of the masks in frame 0
    for label_idx in np.unique(labeled_stack[0,:,:]):
        if label_idx != 0:
            position = np.asarray((labeled_stack[0,:,:] == label_idx).nonzero()).mean(1)
            mask_trajs.append(_MaskTraj(label_idx, 0, position))

    # Iterate through the subsequent frames
    for interval_idx in range(1, n_intervals):

        # Get the positions of every object in the new frame
        labels_1 = [label_idx for label_idx in np.unique(labeled_stack[interval_idx, :, :]) if label_idx != 0]
        positions_1 = np.array([
            np.asarray((labeled_stack[interval_idx,:,:] == label_idx).nonzero()).mean(1)
                for label_idx in labels_1
        ])

        # Make the matrix of inter-mask distances between the last
        # frame and the present frame
        positions_0 = np.asarray([mask_traj.position for mask_traj in mask_trajs])
        n0 = positions_0.shape[0]
        n1 = positions_1.shape[0]
        n_max = max([n0, n1])
        n_min = min([n0, n1])
        distances = np.zeros((n_max, n_max), dtype = 'float64')
        distances[:n0, :n1] = distance_matrix(
            positions_0, 
            positions_1,
        )

        # Solve the assignment problem. assignments[idx_0] is the assignment
        # of idx_0 in positions_0 to idx_1 in positions_1. 
        assignments = np.asarray(munkres_solver.compute(distances))[:,1]

        # Update the mask trajectories
        for idx_0 in range(n0):
            idx_1 = assignments[idx_0]
            # Assigned mask exists
            if idx_1 < n1: 
                mask_trajs[idx_0].labels.append(labels_1[idx_1])
                mask_trajs[idx_0].intervals.append(interval_idx)
                mask_trajs[idx_0].position = positions_1[idx_1, :].copy()
            # Assigned mask is a dead end
            else:
                mask_trajs[idx_0].running = False 

        # For every unassigned mask in the second frame, start a 
        # new mask trajectory
        for idx_1 in range(n1):
            if not any([assignments[idx_0] == idx_1 for idx_0 in range(n0)]):
                mask_trajs.append(_MaskTraj(labels_1[idx_1], interval_idx, positions_1[idx_1, :]))

        # Remove completed mask trajectories
        running = []
        for mask_traj in mask_trajs:
            if mask_traj.running:
                running.append(mask_traj)
            else:
                completed_mask_trajs.append(mask_traj)
        mask_trajs = running 

    # Wrap up
    for mask_traj in mask_trajs:
        completed_mask_trajs.append(mask_traj )

    return completed_mask_trajs 

class _MaskTraj():
    '''
    Convenience object used internally in the *track_masks*
    function.

    '''
    def __init__(self, initial_label, initial_interval, position):
        self.labels = [initial_label]
        self.intervals = [initial_interval]
        self.running = True 
        self.position = position 

def label_stack(binary_stack):
    '''
    Label connected components in a binary stack.

    args
        binary_stack    :   3D ndarray of shape (n_intervals,
                                N, M)

    returns
        (3D ndarray of shape (n_intervals, N, M), n_labels)

    '''
    assert len(binary_stack.shape) == 3
    n_intervals = binary_stack.shape[0]

    # Label connected components in each frame of the mask
    # as separate masks
    labeled_stack = np.zeros(binary_stack.shape, dtype = 'uint16')
    current_labels = 0
    for interval_idx in range(n_intervals):
        labeled_frame, n_labels = ndi.label(
            binary_stack[interval_idx, :, :],
        )
        labeled_frame[labeled_frame > 0] = labeled_frame[labeled_frame > 0] + current_labels
        labeled_stack[interval_idx, :, :] = labeled_frame.copy()
        current_labels += n_labels 

    return labeled_stack, current_labels 

def match_mask_coords(
    binary_masks,
    tolerance_radius = 10,
    mask_upsampling_factor = 1.0,
    mask_size_limit = 3000,
):
    '''
    Given some set of binary masks defined at regular intervals,
    match points on the edges of each mask for subsequent
    interpolation.

    args
        binary_masks                :   3D ndarray of shape (n_intervals, N, M),
                                            the binary masks at each interval

        tolerance_radius            :   float, the search radius for mask point
                                            reconnection

        mask_upsampling_factor      :   float, the number of mask pixels per
                                            real camera pixel

        mask_size_limit             :   int, the maximum number of points to
                                            include in the edge of each mask

    returns
        3D ndarray of shape (n_intervals, n_points, 2), the YX coordinates
            of each mask in each interval.

        The points are matched, so that 
            result[:, point_idx, :]

        corresponds to the trajectory of a 
        single edge point through all of the masks.

    '''
    assert len(binary_masks.shape) == 3
    n_intervals, N, M = binary_masks.shape 

    # Use the first mask to decide how many points in the edge to keep
    binary_mask_0 = binary_masks[0, :, :]
    points_0 = mask.mask_to_oriented_edge(
        binary_mask_0,
        mask_upsampling_factor = mask_upsampling_factor,
        tolerance_radius = tolerance_radius,
    )

    if points_0.shape[0] > mask_size_limit:
        m = (np.arange(mask_size_limit) * points_0.shape[0] / mask_size_limit).round(0).astype('uint16')
        points_0 = points_0[m, :]

    # Save the points corresponding to the mask in each interval
    n_points = points_0.shape[0]
    mask_result = np.zeros((n_intervals, n_points, 2), dtype = 'float64')

    # Save the points corresponding to the first mask
    mask_result[0, :, :] = points_0.copy()

    # Iterate through each mask
    for interval_idx in range(1, n_intervals):

        # Get the list of coordinates corresponding to the edge
        # of the first mask
        points_0 = mask_result[interval_idx - 1, :, :]

        # Get the list of coordinates corresponding to the edge
        # of the second mask
        binary_mask_1 = binary_masks[interval_idx, :, :]
        points_1 = mask.mask_to_oriented_edge(
            binary_mask_1,
            mask_upsampling_factor = mask_upsampling_factor,
            tolerance_radius = tolerance_radius,
        )

        # First, throw away points in the second mask so that
        # the two masks are the same size
        if points_1.shape[0] > points_0.shape[0]:       
            m = (np.arange(points_1.shape[0]) * points_0.shape[0] / points_1.shape[0]).round(0)
            truth_m = [True] + [(m[-i] not in m[:-i]) for i in list(range(1, len(m)))[::-1]]
            points_1 = points_1[np.asarray(truth_m).astype('bool'), :]
        elif points_1.shape[0] < points_0.shape[0]:
            m = (np.arange(points_0.shape[0]) * points_1.shape[0] / points_0.shape[0]).round(0).astype('uint16')
            points_1 = points_1[m, :]
        else:
            pass

        # Align the two masks by centroid
        mean_0 = points_0.mean(axis = 0)
        mean_1 = points_1.mean(axis = 0)
        delta = mean_1 - mean_0 
        points_1_aligned = points_1 - delta 

        # Find the points in the two masks that are closest
        # to each other
        n_points = points_0.shape[0]
        match_idx_0, match_idx_1 = np.unravel_index(
            np.argmin(distance_matrix(points_0, points_1_aligned)),
            (n_points, n_points),
        )
        
        # Match the rest of the points sequentially
        assignments = {match_idx_0 : match_idx_1}
        indicator = shoelace(points_0) * shoelace(points_1)
        if indicator >= 0.0:
            curr_idx_1 = match_idx_1 + 1
            curr_idx_0 = match_idx_0 + 1
            while curr_idx_1 != match_idx_1:
                assignments[curr_idx_0] = curr_idx_1 
                curr_idx_1 += 1
                curr_idx_0 += 1
                if curr_idx_1 == n_points:
                    curr_idx_1 = 0
                if curr_idx_0 == n_points:
                    curr_idx_0 = 0
        elif indicator < 0.0:
            curr_idx_1 = match_idx_1 + 1
            curr_idx_0 = match_idx_0 - 1
            while curr_idx_1 != match_idx_1:
                assignments[curr_idx_0] = curr_idx_1 
                curr_idx_1 += 1
                curr_idx_0 -= 1
                if curr_idx_1 == n_points:
                    curr_idx_1 = 0
                if curr_idx_0 == -1:
                    curr_idx_0 = n_points - 1


        for idx_0 in range(n_points):
            mask_result[interval_idx, idx_0, :] = points_1[assignments[idx_0], :].copy()

    # Return the matched set of mask points
    return mask_result


def shoelace(points):
    '''
    INPUT
        points      :   np.array of shape (N_points, 2)

    RETURNS
        float, the oriented volume of the polygon defined
            by *points*

    '''
    result = 0.0
    for i in range(points.shape[0] - 1):
        result += (points[i + 1, 0] - points[i, 0]) \
            * (points[i + 1, 1] + points[i, 1])
    return result 

