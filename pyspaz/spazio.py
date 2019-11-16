'''
spazio.py -- I/O functions for tracking

'''
# Dataframes
import pandas as pd

# File type conversions
import numpy as np 

# sio.loadmat, for reading *Tracked.mat files
from scipy import io as sio

# Basic I/O
import os
import sys

# ND2 reading
from nd2reader import ND2Reader

# TIF reading
import tifffile

def save_locs(locs_file, locs_df, metadata):
    '''
    Save a pandas.DataFrame containing localizations
    and associated metadata to a *.locs file.

    args
        locs_file   :   str, file to save to
        locs_df     :   pandas.DataFrame
        metadata    :   dict of metadata terms. All keys
                        and values are converted to str.

    '''
    with open(locs_file, 'w', newline = '') as o:
        o.write('METADATA_START\n')
        for k_idx, key in enumerate(list(metadata.keys())):
            value = metadata[key]
            o.write('%s\t%s\n' % (str(key), str(value)))
        o.write('METADATA_END\n')
        locs_df.to_csv(o, sep='\t', index=False)

def load_locs(locs_file):
    '''
    Load localization data and metadata from a *.locs
    file.

    args
        locs_file       :   str, path

    '''
    check_file_exists(locs_file)

    line = ''
    metadata = {}
    with open(locs_file, 'r') as f:
        while line != 'METADATA_START':
            line = f.readline().replace('\n', '')
        line = f.readline().replace('\n', '')
        while line != 'METADATA_END':
            key, value = line.split('\t')
            metadata[key] = value
            line = f.readline().replace('\n', '')
        locs = pd.read_csv(f, sep = '\t')
    return locs, metadata 

def save_trajs(tracked_mat_file, trajs, metadata, traj_cols = None):
    '''
    args
        tracked_mat_file    :   str
        trajs               :   2D ndarray of shape (n_trajs, m),
                                trajectories and associated 
                                information
        output_mat          :   str, *Tracked.mat to save to

    '''
    out_dict = {
        'trackedPar' : trajs,
        'metadata' : format_metadata_out(metadata),
    }
    if traj_cols != None:
        if len(traj_cols) != trajs.shape[1]:
            raise RuntimeError('save_tracked: traj_cols does not ' \
                'match shape of passed trajectories')
        out_dict['traj_cols'] = traj_cols
    sio.savemat(tracked_mat_file, out_dict)

def load_trajs(tracked_mat_file, unpack_cols = True):
    '''
    args
        tracked_mat_file    :   str
        unpack_cols         :   bool, may need to be changed to 
                                False for MATLAB trajectories

    returns
        3-tuple (
            2D ndarray, the trajectories;
            dict, metadata (if exists);
            list, name of trajectory columns (if exists)
        )

    '''
    check_file_exists(tracked_mat_file)

    in_dict = sio.loadmat(tracked_mat_file)
    keys = [str(i) for i in list(in_dict.keys())]
    if 'metadata' in keys:
        metadata = format_metadata_in(in_dict['metadata'])
    else:
        metadata = {}
    if 'traj_cols' in keys:
        traj_cols = [trim_end_str(i) for i in in_dict['traj_cols']]
    else:
        traj_cols = []
    trajs = in_dict['trackedPar']

    # Correct for indexing error, if originates from MATLAB
    if trajs.shape[0] == 1:
        trajs = trajs[0,:]

    # Unpack the columns for single-value attributes
    if unpack_cols:
        n_cols = len(trajs[0])
        n_trajs = len(trajs)
        for col_idx in range(1, n_cols):
            for traj_idx in range(n_trajs):
                trajs[traj_idx][col_idx] = trajs[traj_idx][col_idx][0]

    return trajs, metadata, traj_cols

def save_trajectory_obj_to_mat(
    mat_file_name,
    trajectories,
    metadata,
    frame_interval_sec,
    pixel_size_um = 0.16,
    convert_pixel_to_um = True,
):
    '''
    Save a list of pyspaz.track.Trajectory objects to 
    *Tracked.mat file.

    args
        mat_file_name       :   str
        trajectories        :   list of pyspaz.track.Trajectory
        metadata            :   dict
        frame_interval_sec  :   float
        pixel_size_um       :   float
        convert_pixel_to_um :   bool

    returns
        dict, the read *Tracked.mat file

    '''
    result_list = []

    if convert_pixel_to_um:
        for traj_idx in range(len(trajectories)):
            trajectories[traj_idx].positions = trajectories[traj_idx].positions * pixel_size_um 

    for traj_idx, trajectory in enumerate(trajectories):
        result_list.append([
            trajectory.positions,
            trajectory.frames,
            [i * frame_interval_sec for i in trajectory.frames],
            trajectory.mle_I0,
            trajectory.mle_bg,
            trajectory.llr_detect,
        ])
    traj_cols = ['position', 'frame_idx', 'time', 'I0', 'bg', 'llr_detect']
    out_dict = {
        'trackedPar' : result_list,
        'metadata' : format_metadata_out(metadata),
        'traj_cols' : traj_cols,
    }

    sio.savemat(mat_file_name, out_dict)
    return out_dict  

def trajs_to_locs(trajs, traj_cols, units = 'um'):
    '''
    Deconstruct trajectories into individual localizations,
    returning the result as a pandas.DataFrame.

    Importantly, this function ASSUMES that the first column
    of each trajectory corresponds to a 2D position vector.

    args
        trajs   :   2D ndarray of shape (n_trajs, len(traj_cols)-1),
                    where the `-1` accounts for the fact that 
                    `position` is rolled up into a single XY array

        traj_cols   :   list of str

        units       :   str, for column naming

    returns
        pandas.DataFrame

    '''
    if len(traj_cols) != len(trajs[0]):
        raise RuntimeError('trajs_to_locs: number of trajectory column ' \
            'labels does not match number of trajectory attributes')
    n_trajs = trajs.shape[0]
    n_locs = sum([trajs[traj_idx][0].shape[0] for traj_idx in range(n_trajs)])
    out = np.zeros((n_locs, len(traj_cols)+1), dtype = 'float64')
    c_idx = 0
    for traj_idx in range(n_trajs):
        for l_idx in range(trajs[traj_idx][0].shape[0]):
            out[c_idx,:2] = trajs[traj_idx][0][l_idx,:]
            for col in range(1, len(traj_cols)):
                out[c_idx, col+1] = trajs[traj_idx][col][l_idx]
            c_idx += 1
    if units == 'um':
        column_names = ['y_um', 'x_um'] + list(traj_cols)[1:]
    else:
        column_names = ['y_pixels', 'x_pixels'] + list(traj_cols)[1:]
    df = pd.DataFrame(out, columns = column_names)
    return df


def extract_positions_from_trajs(trajs):
    '''
    Return a 2D ndarray with all of the localization
    positions found in the trajectories.

    args
        trajs       :   trajectory object

    returns
        2D ndarray of shape (n_locs, 2), the y and x
            positions of each localization in um

    '''
    n_trajs = trajs.shape[0]
    n_locs = sum([trajs[traj_idx][0].shape[0] for traj_idx in range(N_trajs)])
    positions = np.zeros((n_locs, 2))
    loc_idx = 0
    for traj_idx in range(n_trajs):
        for _idx in range(trajs[traj_idx][0].shape[0]):
            positions[loc_idx, :] = trajs[traj_idx][0][_idx, :]
            loc_idx += 1
    return positions 

class ImageFileReader(object):
    '''
    Interface for grabbing frames from TIF or ND2 files.
    
        file_name: str, the name of a single TIF or ND2 file
        
    '''
    def __init__(
        self,
        file_name,
    ):
        self.file_name = file_name
        if '.nd2' in file_name:
            self.type = 'nd2'
            self.file_reader = ND2Reader(file_name)
            self.is_closed = False
        elif ('.tif' in file_name) or ('.tiff' in file_name):
            self.type = 'tif'
            self.file_reader = tifffile.TiffFile(file_name)
            self.is_closed = False 
        else:
            print('Image format %s not recognized' % \
                 os.path.splitext(file_name)[1])
            self.type = None
            self.is_closed = True 
        
    def get_shape(self):
        '''
        returns
            (int, int, int), the y dimension, x dimension, and
            t dimension of the data
        
        '''
        if self.is_closed:
            raise RuntimeError("Object is closed")
            
        if self.type == 'nd2':
            y_dim = self.file_reader.metadata['height']
            x_dim = self.file_reader.metadata['width']
            t_dim = self.file_reader.metadata['total_images_per_channel']
        elif self.type == 'tif':
            y_dim, x_dim = self.file_reader.pages[0].shape 
            t_dim = len(self.file_reader.pages)
        
        return (y_dim, x_dim, t_dim)
    
    def get_frame(self, frame_idx):
        '''
        args
            frame_idx: int
        
        returns
            2D np.array, the corresponding frame
        
        '''
        if self.is_closed:
            raise RuntimeError("Object is closed")
            
        if self.type == 'nd2':
            return self.file_reader.get_frame_2D(t = frame_idx)
        elif self.type == 'tif':
            return self.file_reader.pages[frame_idx].asarray()

    def close(self):
        self.file_reader.close()
        self.is_closed = True 

def check_file_exists(path):
    if not os.path.isfile(path):
        raise RuntimeError('check_file_exists: %s not found' % path)

def try_numeric_convert(arg):
    try:
        return int(arg)
    except ValueError:
        try:
            return float(arg)
        except ValueError:
            return arg

def trim_end_str(string):
    while string[-1] == ' ':
        string = string[:-1]
    return string

def format_metadata_out(metadata_dict):
    '''
    Format metadata so that it can be stored in 
    a *Tracked.mat file.

    '''
    out = []
    for k, v in zip(metadata_dict.keys(), metadata_dict.values()):
        out.append((str(k), str(v)))
    return out 

def format_metadata_in(metadata_tuple_list):
    '''
    From a list of 2-tuples, assemble a dictionary.
    Convert string values to numeric if possible.

    MAT files use a uniform width for their tuple
    arrays, so we also trim the ends to remove this 
    feature.

    args
        metadata_tuple_list     :   list of 2-tuple,
                                    the (key, value) 
                                    pairs from a 
                                    *Tracked.mat file

    returns
        dict, the assembled dictionary

    '''
    out = {}
    for k, v in metadata_tuple_list:
        out[trim_end_str(k)] = try_numeric_convert(trim_end_str(v))
    return out 




