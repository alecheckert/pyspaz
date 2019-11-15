'''
utils.py
'''
import numpy as np 

#
# Localization utilities
#

def expand_window(image, N, M):
    '''
    Pad an image with zeros to force it into the
    shape (N, M), keeping the image centered in 
    the frame.

    args
        image       :   2D np.array with shape (N_in, M_in)
                        where both N_in < N and M_in < M

        N, M        :   floats, the desired image dimensions

    returns
        2D np.array of shape (N, M)

    '''
    N_in, M_in = image.shape
    out = np.zeros((N, M))
    nc = np.floor(N/2 - N_in/2).astype(int)
    mc = np.floor(M/2 - M_in/2).astype(int)
    out[nc:nc+N_in, mc:mc+M_in] = image
    return out

def local_max_2d(image):
    '''
    Determine local maxima in a 2D image.

    Returns 2D np.array of shape (image.shape), 
    a Boolean image of all local maxima
    in the image.

    '''
    N, M = image.shape
    ref = image[1:N-1, 1:M-1]
    pos_max_h = (image[0:N-2, 1:M-1] < ref) & (image[2:N, 1:M-1] < ref)
    pos_max_v = (image[1:N-1, 0:M-2] < ref) & (image[1:N-1, 2:M] < ref)
    pos_max_135 = (image[0:N-2, 0:M-2] < ref) & (image[2:N, 2:M] < ref)
    pos_max_45 = (image[2:N, 0:M-2] < ref) & (image[0:N-2, 2:M] < ref)
    peaks = np.zeros((N, M))
    peaks[1:N-1, 1:M-1] = pos_max_h & pos_max_v & pos_max_135 & pos_max_45
    return peaks.astype('bool')


def gaussian_model(sigma, window_size):
    '''
    Generate a model Gaussian PSF in a square array
    by sampling the value of the Gaussian in the center 
    of each pixel. (**for detection**)

    args
        sigma       :   float, xy sigma
        window_size :   int, pixels per side

    returns
        2D np.array, the PSF model

    '''
    half_w = int(window_size) // 2
    ii, jj = np.mgrid[-half_w:half_w+1, -half_w:half_w+1]
    sig2 = sigma ** 2
    g = np.exp(-((ii**2) + (jj**2)) / (2 * sig2)) / sig2 
    return g 

def get_pivots(matrix):
    '''
    args
        matrix: 2D np.array of shape (n, n)
    
    returns
        np.array of shape (n,), the pivots of the matrix U
            in the factorization matrix = LU
    
    '''
    # Make sure we don't do this in place
    A = matrix.copy()
    
    # Get the shape of the matrix. Should be square.
    n = matrix.shape[0]
    
    # Perform Gaussian elimination
    for col_idx in range(n-1):
        # If we encounter a zero pivot, let the program know]
        if matrix[col_idx, col_idx] == 0.0:
            raise ZeroDivisionError
        
        for row_idx in range(col_idx + 1, n):
            A[row_idx, :] = A[row_idx, :] - A[col_idx, :] * \
                A[row_idx, col_idx] / A[col_idx, col_idx]

    # Return the diagonal, giving the pivots
    return np.diagonal(A)

# 
# Tracking utilities
#
def connected_components(semigraph):
    '''
    Find independent subgraphs in a semigraph by a floodfill procedure.
    
    args
        semigraph : 2D binary np.array (only 0/1 values), representing
            a semigraph
    
    returns
        subgraphs : list of 2D np.array, the independent adjacency subgraphs;
        
        subgraph_y_indices : list of 1D np.array, the y-indices of each 
            independent adjacency subgraph;
            
        subgraph_x_indices : list of 1D np.array, the x-indices of each 
            independent adjacency subgraph;
        
        y_without_x : 1D np.array, the y-indices of y-nodes without any edges
            to x-nodes;
        
        x_without_y : 1D np.array, the x-indices of x-nodes without any edges
            to y-nodes.
            
    '''
    if semigraph.max() > 1:
        raise RuntimeError("connected_components only takes binary arrays")
        
    # The set of all y-nodes (corresponding to y-indices in the semigraph)
    y_indices = np.arange(semigraph.shape[0]).astype('uint16')
    
    # The set of all x-nodes (corresponding to x-indices in the semigraph)
    x_indices = np.arange(semigraph.shape[1]).astype('uint16')

    # Find y-nodes that don't connect to any x-node,
    # and vice versa
    where_y_without_x = (semigraph.sum(axis = 1) == 0)
    where_x_without_y = (semigraph.sum(axis = 0) == 0)
    y_without_x = y_indices[where_y_without_x]
    x_without_y = x_indices[where_x_without_y]
    
    # Consider the remaining nodes, which have at least one edge
    # to a node of the other class 
    semigraph = semigraph[~where_y_without_x, :]
    semigraph = semigraph[:, ~where_x_without_y]
    y_indices = y_indices[~where_y_without_x]
    x_indices = x_indices[~where_x_without_y]
    
    # For the remaining nodes, keep track of (1) the subgraphs
    # encoding connected components, (2) the set of original y-indices
    # corresponding to each subgraph, and (3) the set of original x-
    # indices corresponding to each subgraph
    subgraphs = []
    subgraph_y_indices = []
    subgraph_x_indices = []

    # Work by iteratively removing independent subgraphs from the 
    # graph. The list of nodes still remaining are kept in 
    # *unassigned_y* and *unassigned_x*
    unassigned_y, unassigned_x = (semigraph == 1).nonzero()
    
    # The current index is used to floodfill the graph with that
    # integer. It is incremented as we find more independent subgraphs. 
    current_idx = 2
    
    # While we still have unassigned nodes
    while len(unassigned_y) > 0:
        
        # Start the floodfill somewhere with an unassigned y-node
        semigraph[unassigned_y[0], unassigned_x[0]] = current_idx
    
        # Keep going until subsequent steps of the floodfill don't
        # pick up additional nodes
        prev_nodes = 0
        curr_nodes = 1
        while curr_nodes != prev_nodes:
            # Only floodfill along existing edges in the graph
            where_y, where_x = (semigraph == current_idx).nonzero()
            
            # Assign connected nodes to the same subgraph index
            semigraph[where_y, :] *= current_idx
            semigraph[:, where_x] *= current_idx
            
            # Correct for re-finding the same nodes and multiplying
            # them more than once
            semigraph[semigraph > current_idx] = current_idx
            
            # Update the node counts in this subgraph
            prev_nodes = curr_nodes
            curr_nodes = (semigraph == current_idx).sum()
        current_idx += 1 

        # Get the local indices of the y-nodes and x-nodes (in the context
        # of the remaining graph)
        where_y = np.unique(where_y)
        where_x = np.unique(where_x)

        # Use the local indices to pull this subgraph out of the 
        # bigger graph
        subgraph = semigraph[where_y, :]
        subgraph = subgraph[:, where_x]

        # Save the subgraph
        subgraphs.append(subgraph)
        
        # Get the original y-nodes and x-nodes that were used in this
        # subgraph
        subgraph_y_indices.append(y_indices[where_y])
        subgraph_x_indices.append(x_indices[where_x])

        # Update the list of unassigned y- and x-nodes
        unassigned_y, unassigned_x = (semigraph == 1).nonzero()

    return subgraphs, subgraph_y_indices, subgraph_x_indices, y_without_x, x_without_y

def sq_radial_distance(vector, points):
    return ((vector - points) ** 2).sum(axis = 1)

def sq_radial_distance_array(points_0, points_1):
    '''
    args
        points_0    :   np.array of shape (N, 2), coordinates
        points_1    :   np.array of shape (M, 2), coordinates

    returns
        np.array of shape (N, M), the radial distances between
            each pair of points in the inputs

    '''
    array_points_0 = np.zeros((points_0.shape[0], points_1.shape[0], 2), dtype = 'float')
    array_points_1 = np.zeros((points_0.shape[0], points_1.shape[0], 2), dtype = 'float')
    for idx_0 in range(points_0.shape[0]):
        array_points_0[idx_0, :, :] = points_1 
    for idx_1 in range(points_1.shape[0]):
        array_points_1[:, idx_1, :] = points_0
    result = ((array_points_0 - array_points_1)**2).sum(axis = 2)
    return result 


