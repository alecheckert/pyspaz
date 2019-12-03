'''
simulator.py
'''
import numpy as np 
import matplotlib.pyplot as plt 
from . import traj_analysis

def two_state_brownian(
    f0,
    d0,
    d1,
    n_molecules = 10000,
    n_dt = 4,
    dt = 0.00548,
):
    state_vector = np.random.choice([0, 1], size = n_molecules)
    n0 = (state_vector == 0).sum()
    n1 = (state_vector == 1).sum()

    # Simulate bound
    bound_jumps = np.random.normal(scale = np.sqrt(2*d0*dt), size=(n0, 2, n_dt))
    bound_disps = np.cumsum(bound_jumps, axis = 2)
    bound_r_disps = np.sqrt((bound_disps**2).sum(axis=1))

    # Simulate free
    free_jumps = np.random.normal(scale = np.sqrt(2*d1*dt), size = (n1, 2, n_dt))
    free_disps = np.cumsum(free_jumps, axis = 2)
    free_r_disps = np.sqrt((free_disps ** 2).sum(axis = 1))

    radial_disp_histograms = np.zeros((5000, 4), dtype = 'int64')
    bin_edges = np.linspace(0, 5.0, 5001)
    for dt_idx in range(n_dt):
        radial_disp_histograms[:, dt_idx] = radial_disp_histograms[:, dt_idx] + \
            np.histogram(bound_r_disps[:,dt_idx], bins = bin_edges)[0]
        radial_disp_histograms[:, dt_idx] = radial_disp_histograms[:, dt_idx] + \
            np.histogram(free_r_disps[:,dt_idx], bins = bin_edges)[0]

    return radial_disp_histograms, bin_edges 

def two_state_brownian_zcorr(
    f0,
    d0,
    d1,
    n_molecules = 10000,
    n_dt = 4,
    dt = 0.00548,
    dz = 0.7,
    loc_error = 0.035,
    no_gaps = False,
): 
    '''
    Simulate displacements from a population of Brownian particles 
    with two diffusion coefficients, *d0* and *d1*. Particles 
    at one diffusion coefficient do not convert to the other diffusing state.

    Displacements are observed in the 2D plane, but if a free particle
    diffuses outside the bounds of the detection slice in z, then 
    the corresponding XY displacement is not recorded. The thickness
    of the axial detection slice is determined by the *dz* parameter.

    '''
    half_z = dz / 2

    state_vector = (np.random.random(size = n_molecules) > f0).astype('float64')
    n0 = (state_vector == 0).sum()
    n1 = (state_vector == 1).sum()

    # Simulate bound
    bound_jumps = np.random.normal(scale = np.sqrt(2*d0*dt), size=(n0, 2, n_dt))
    bound_disps = np.cumsum(bound_jumps, axis = 2)
    if loc_error != 0.0:
        bound_disps += np.random.normal(scale=loc_error, size=bound_disps.shape)
    bound_r_disps = np.sqrt((bound_disps**2).sum(axis=1))

    # Simulate free
    free_jumps = np.random.normal(scale = np.sqrt(2*d1*dt), size = (n1, 2, n_dt))
    free_disps = np.cumsum(free_jumps, axis = 2)
    if loc_error != 0.0:
        free_disps += np.random.normal(scale=loc_error, size=free_disps.shape)
    free_r_disps = np.sqrt((free_disps ** 2).sum(axis = 1))

    # Simulate diffusion along the axis
    axial_jumps = np.random.normal(scale = np.sqrt(2*d1*dt), size = (n1, n_dt))
    axial_disps = np.cumsum(axial_jumps, axis = 1)
    initial_pos = np.random.uniform(-half_z, half_z, size = n1)
    for dt_idx in range(n_dt):
        axial_disps[:, dt_idx] = axial_disps[:, dt_idx] + initial_pos 

    outside = np.abs(axial_disps) > half_z 

    if no_gaps:
        outside_copy = outside.copy()
        for dt_idx in range(n_dt):
            outside_copy[:, dt_idx] = outside[:, :dt_idx+1].any(axis = 1)
        outside = outside_copy 

    # for dt_idx in range(n_dt):
    #     print('%d/%d outside, frac %f' % (outside[:,dt_idx].sum(), n1, outside[:,dt_idx].sum()/n1))
    # print('bound_r_disps.shape = ', bound_r_disps.shape)
    # print('free_r_disps.shape = ', free_r_disps.shape)

    radial_disp_histograms = np.zeros((5000, 4), dtype = 'int64')
    bin_edges = np.linspace(0, 5.0, 5001)
    for dt_idx in range(n_dt):
        radial_disp_histograms[:, dt_idx] = radial_disp_histograms[:, dt_idx] + \
            np.histogram(bound_r_disps[:,dt_idx], bins = bin_edges)[0]
        radial_disp_histograms[:, dt_idx] = radial_disp_histograms[:, dt_idx] + \
            np.histogram(free_r_disps[(~outside[:, dt_idx]).nonzero(), dt_idx], bins = bin_edges)[0]

    return radial_disp_histograms, bin_edges 




