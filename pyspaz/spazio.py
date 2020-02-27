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

# Get the list of files in a given directory
from glob import glob 

# Progress bar
from tqdm import tqdm 

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
        locs_df.index = np.arange(len(locs_df))
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
            metadata[key] = try_numeric_convert(value)
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

def load_trajs(filename, unpack_cols = True):
    '''
    args
        filename            :   str, either *Tracked.mat or *.csv
        unpack_cols         :   bool, may need to be changed to 
                                False for MATLAB trajectories

    returns
        3-tuple (
            2D ndarray, the trajectories;
            dict, metadata (if exists);
            list, name of trajectory columns (if exists)
        )

    '''
    check_file_exists(filename)

    if 'Tracked.mat' in filename:
        in_dict = sio.loadmat(filename)
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

    elif '.txt' in filename or '.trajs' in filename:
        df, metadata = load_locs(filename)
        return df, metadata, df.columns

    else:
        raise RuntimeError('spazio.load_trajs: input file must be either *Tracked.mat or *.txt/*.trajs')

def trajs_to_tracked_mat_directory(
    directory_with_traj_files,
    out_dir=None,
    matlabexec = '/Applications/MATLAB_R2014b.app/bin/matlab',
    run_conversion=True,
):
    """
    For every *.trajs file in a given directory, convert
    to *Tracked.mat format. This attempts to 

    args
    ----
        directory_with_traj_files :  str, path to directory
                                    containing *.trajs files
        out_dir :  str, file to save the resulting *Tracked.mat
            files to

    returns
    -------
        None

    """

    # Get all *.trajs files in this directory
    traj_files = glob("%s/*.trajs" % directory_with_traj_files)

    # Specify output directory
    if out_dir is None:
        out_dir = directory_with_traj_files

    # Create output directory, if doesn't exist
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    # Run on every individual *.trajs file
    for traj_file_idx, traj_file in tqdm(enumerate(traj_files)):

        # Generate output file name
        out_tracked_mat = '%s/%s' % (
            out_dir, os.path.basename(traj_file).replace('.trajs', '_Tracked.mat')
        )

        # Save to that file name
        trajs_to_tracked_mat(
            traj_file,
            out_tracked_mat,
        )

    # Do hard file conversion from within MATLAB. 
    # this requires a little trickery (aka bullshit)
    if run_conversion:
        import imp 
        curr_dir = os.getcwd()
        os.chdir(out_dir)
        convert_trajs_to_mat_path = '%s/convert_trajs_to_mat.m' % imp.find_module('pyspaz')[1]
        os.system('cp %s convert_trajs_to_mat.m' % convert_trajs_to_mat_path)
        os.system('%s -r -nodesktop -nosplash -nodisplay convert_trajs_to_mat' % matlabexec)
        os.chdir(curr_dir)

def trajs_to_tracked_mat(
    in_trajs_file,
    out_tracked_mat,
):
    """
    Convert a .trajs (.locs format)-formatted file 
    into a *Tracked.mat file.

    args
    ----
        in_trajs_file :  str
        out_tracked_mat :  str

    returns
    -------

    """
    # Load trajectories into file
    trajs, metadata = load_locs(in_trajs_file)

    # Convert to MATLAB indexing
    trajs['frame_idx'] = trajs['frame_idx'] + 1

    # Exclude unassigned localizations
    trajs = trajs[trajs['traj_idx'] > 0]

    # Add timestamp column
    try:
        frame_interval_sec = metadata['frame_interval_sec']
    except KeyError:
        frame_interval_sec = 0.00
    trajs['time'] = trajs['frame_idx'] * frame_interval_sec

    # Convert to np.ndarray for faster indexing
    trajs = np.asarray(trajs[['frame_idx', 'time', 'y_pixels',
        'x_pixels', 'traj_idx']])

    # Convert from pixels to um
    pixel_size_um = metadata['pixel_size_um']
    trajs[:,2:4] = trajs[:,2:4] * pixel_size_um 

    # Convert to the expected MATLAB format
    result_list = []
    unique_traj_indices = np.unique(trajs[:,4]).astype('int64')
    for traj_idx in unique_traj_indices:
        traj = trajs[trajs[:,4] == traj_idx]

        result_list.append([
            traj[:,2:4],  # position
            traj[:,0],  # frame index
            traj[:,1],  # timestamp, seconds
        ])

    traj_cols = ['position', 'frame_idx', 'time']
    out_dict = {
        'trackedPar' : result_list,
        'metadata' : format_metadata_out(metadata),
        'traj_cols' : traj_cols,
    }

    # Save to .mat file
    sio.savemat(out_tracked_mat, out_dict)

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
            [trajectory.subproblem_shapes[i][0] for i in range(len(trajectory.subproblem_shapes))],
            [trajectory.subproblem_shapes[i][1] for i in range(len(trajectory.subproblem_shapes))],
        ])
    traj_cols = ['position', 'frame_idx', 'time', 'I0', 'bg', 
        'llr_detect', 'subproblem_n_traj', 'subproblem_n_loc']
    out_dict = {
        'trackedPar' : result_list,
        'metadata' : format_metadata_out(metadata),
        'traj_cols' : traj_cols,
    }

    sio.savemat(mat_file_name, out_dict)
    return out_dict  

def save_trajectory_obj_to_txt(
    csv_file_name,
    trajectories,
    metadata,
    frame_interval_sec,
    pixel_size_um = 0.16,
):
    n_trajs = len(trajectories)
    n_locs = sum([len(traj.positions) for traj in trajectories])
    result = np.zeros((n_locs, 10), dtype = 'float64')
    c_idx = 0
    for traj_idx, trajectory in enumerate(trajectories):
        traj_len = trajectory.positions.shape[0]
        result[c_idx:c_idx+traj_len, :2] = trajectory.positions.copy()
        result[c_idx:c_idx+traj_len, 2] = trajectory.frames 
        result[c_idx:c_idx+traj_len, 3] = np.array(trajectory.frames) * frame_interval_sec
        result[c_idx:c_idx+traj_len, 4] = trajectory.mle_I0
        result[c_idx:c_idx+traj_len, 5] = trajectory.mle_bg
        result[c_idx:c_idx+traj_len, 6] = trajectory.llr_detect 
        result[c_idx:c_idx+traj_len, 7] = [trajectory.subproblem_shapes[i][0] for i in range(len(trajectory.subproblem_shapes))]
        result[c_idx:c_idx+traj_len, 8] = [trajectory.subproblem_shapes[i][1] for i in range(len(trajectory.subproblem_shapes))]
        result[c_idx:c_idx+traj_len, 9] = traj_idx 
        c_idx += traj_len 
        
    result_df = pd.DataFrame(result, columns = [
        'y_pixels', 'x_pixels', 'frame_idx', 'time', 'I0', 'bg', 
        'llr_detect', 'subproblem_n_traj', 'subproblem_n_loc',
        'traj_idx',
    ])
    result_df['frame_idx'] = result_df['frame_idx'].astype('uint32')
    result_df['traj_idx'] = result_df['traj_idx'].astype('uint32')
    result_df['subproblem_n_traj'] = result_df['subproblem_n_traj'].astype('uint16')
    result_df['subproblem_n_loc'] = result_df['subproblem_n_traj'].astype('uint16')

    save_locs(csv_file_name, result_df, metadata)

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
        pandas.DataFrame of shape (n_locs, len(traj_cols) + 2)

    '''
    if len(traj_cols) != len(trajs[0]):
        raise RuntimeError('trajs_to_locs: number of trajectory column ' \
            'labels does not match number of trajectory attributes')
    n_trajs = trajs.shape[0]
    n_locs = sum([trajs[traj_idx][0].shape[0] for traj_idx in range(n_trajs)])
    out = np.zeros((n_locs, len(traj_cols)+2), dtype = 'float64')
    c_idx = 0
    for traj_idx in range(n_trajs):
        for l_idx in range(trajs[traj_idx][0].shape[0]):
            out[c_idx,:2] = trajs[traj_idx][0][l_idx,:]
            out[c_idx, 2] = traj_idx 
            for col in range(1, len(traj_cols)):
                out[c_idx, col+2] = trajs[traj_idx][col][l_idx]
            c_idx += 1
    if units == 'um':
        column_names = ['y_um', 'x_um', 'traj_idx'] + list(traj_cols)[1:]
    else:
        column_names = ['y_pixels', 'x_pixels', 'traj_idx'] + list(traj_cols)[1:]
    df = pd.DataFrame(out, columns = column_names)
    df['traj_idx'] = df['traj_idx'].astype('int64')
    return df

def locs_to_trajs(locs, traj_col = 'traj_idx'):
    '''
    Convert a set of trajectories in pandas.DataFrame format
    into trajectories in *Tracked.mat format.

    '''
    if traj_col not in locs.columns:
        raise RuntimeError('spazio.locs_to_trajs: input dataframe must contain the traj_col %s' % traj_col)

    n_trajs = locs[traj_col].max() + 1
    for traj_idx, traj in locs.groupby(traj_col):
        raise NotImplementedError

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
    n_locs = sum([trajs[traj_idx][0].shape[0] for traj_idx in range(n_trajs)])
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

    def min_max(self):
        '''
        returns
            (int, int), the minimum and maximum
                pixel intensities in the stack

        '''
        N, M, n_frames = self.get_shape()
        c_max, c_min = 0, 0
        for frame_idx in range(n_frames):
            frame = self.get_frame(frame_idx)
            frame_min = frame.min()
            frame_max = frame.max()
            if frame_min < c_min:
                c_min = frame_min
            if frame_max > c_max:
                c_max = frame_max
        return c_min, c_max 

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



