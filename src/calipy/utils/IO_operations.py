# System packages
import sys
import os
from glob import glob
import h5py
import psutil
import json
import gc as memory_garbage_collector

# Numerical packages
import numpy as np
import pandas as pd

# Local repository
from third_party import tifffile as tiff
from calipy.general_configs import default as GC
from calipy.utils.Array_operations import temporal_smooth, cross_correlate_image


PROJECTIONS_TYPES = ['mean', 'median', 'max', 'standard_deviation', 'correlation']


def prepare_data_for_GUI(tiff_folder, output_folder, overwrite_frame_first=False, overwrite_time_first=False):
    """

    :param tiff_folder:
    :param output_folder:
    :param overwrite_frame_first:
    :param overwrite_time_first:
    :return:
    """
    # Get list of files
    tiff_files = glob(tiff_folder + '/**/*.tif*', recursive=True)
    n_tiff_files = len(tiff_files)

    # Initialize output dictionary
    PARAMETERS = dict()
    PARAMETERS['dataset_ID'] = os.path.split(os.path.commonprefix(tiff_files))[-1].strip()
    # Initialize conversion flags
    do_convert_frame_first = False
    do_convert_time_first = False

    # Iterate through tiff files to find width and height of each field of view (FOV)
    log('Reading info on files')
    FOV_size = np.zeros((n_tiff_files, 2), dtype=int)
    n_frames_per_file = np.zeros((n_tiff_files, ), dtype=int)
    TIFF_dtype = np.zeros((n_tiff_files, ), dtype=object)
    for i_file in range(n_tiff_files):
        # Read file and get its shape
        TIFF = tiff.imread(tiff_files[i_file])
        TIFF_shape = TIFF.shape
        if len(TIFF_shape) == 4:
            TIFF_shape = TIFF_shape[:3]
        # Store shape
        n_frames_per_file[i_file] = TIFF_shape[0]
        FOV_size[i_file, :] = TIFF_shape[1:]
        TIFF_dtype[i_file] = TIFF.dtype

    # Check consistency of FOV size
    if not np.all(FOV_size[:, 0] == FOV_size[0, 0]) or not np.all(FOV_size[:, 1] == FOV_size[0, 1]):
        raise ValueError('Imaged fields of view are not all of the same size')
    # Get largest data type to accommodate all images
    TIFF_data_type = np.find_common_type(TIFF_dtype, [])
    # Store information
    PARAMETERS['n_frames'] = n_frames_per_file.sum()
    PARAMETERS['frame_height'] = FOV_size[0, 0]
    PARAMETERS['frame_width'] = FOV_size[0, 1]
    PARAMETERS['n_pixels'] = PARAMETERS['frame_height'] * PARAMETERS['frame_width']
    PARAMETERS['dtype'] = TIFF_data_type
    # Make indices of beginning and end of each tiff file
    end_condition = n_frames_per_file.cumsum().reshape(-1, 1)
    start_condition = np.vstack(([0], end_condition[:-1] + 1))
    frames_idx = np.hstack((start_condition, end_condition))
    PARAMETERS['frames_idx'] = frames_idx.tolist()

    # Get name of subfolders
    folders = [os.path.split(i)[-2] for i in tiff_files]
    folders = [i.replace(os.path.commonprefix(tiff_files), '') for i in folders]
    PARAMETERS['condition_names'] = [os.path.splitext(os.path.split(i)[-1])[0] for i in tiff_files]
    # Get beginning and end frame of each session (i.e., a subfolder)
    condition_names, idx_condition_names = np.unique(folders, return_inverse=True)
    sessions_last_frame = np.zeros_like(condition_names, dtype=int)
    for cond_idx, name in enumerate(pd.unique(folders)):
        idx_this_condition = np.where(idx_condition_names == np.where(np.in1d(condition_names, name))[0][0])[0]
        sessions_last_frame[cond_idx] = np.sum(n_frames_per_file[idx_this_condition])
    PARAMETERS['sessions_last_frame'] = np.cumsum(sessions_last_frame).tolist()

    # Calculate the number of frames that can be held in memory while reading / writing final file
    temp = np.empty(shape=(1, 1), dtype=TIFF_data_type)
    n_frames_per_chunk = max_chunk_size(n_bytes=temp.itemsize, memory_cap=0.90)

    # Make filenames of data files
    PARAMETERS['filename_frame_first'] = os.path.join(output_folder, 'stack_frame_first.dat')
    PARAMETERS['filename_time_first'] = os.path.join(output_folder, 'stack_time_first.dat')
    PARAMETERS['filename_projections'] = os.path.join(output_folder, 'projections.hdf5')
    # Make filename of output file
    PARAMETERS['filename_ROIs'] = os.path.join(output_folder, 'ROIs_info.mat')

    # Create frame-first file, if it doesn't exist
    if not os.path.exists(PARAMETERS['filename_frame_first']) or not os.path.exists(PARAMETERS['filename_projections']) or overwrite_frame_first:
        do_convert_frame_first = True
        log('Creating frame-first file in \'%s\'' % PARAMETERS['filename_frame_first'])
        # Make file for data
        destination_shape = (PARAMETERS['n_frames'], PARAMETERS['frame_height'], PARAMETERS['frame_width'])
        frame_first_file = np.memmap(PARAMETERS['filename_frame_first'], dtype=PARAMETERS['dtype'], mode="w+", shape=destination_shape)

        for i_file in range(n_tiff_files):
            # Read tiff file
            filename = tiff_files[i_file]
            log('Copying file %i/%i: %s' % (i_file + 1, n_tiff_files, filename))
            TIFF = tiff.imread(filename).astype(TIFF_data_type)
            TIFF_shape = TIFF.shape
            # Average across color channels
            if len(TIFF_shape) == 4:
                TIFF = np.nanmean(TIFF, axis=3)
            # Replace NaNs with 0s
            TIFF[np.isnan(TIFF)] = 0

            # Get edges of chunk
            start_frame = int(np.clip(PARAMETERS['frames_idx'][i_file][0] - 1, a_min=0, a_max=PARAMETERS['n_frames']))
            end_frame = int(np.clip(PARAMETERS['frames_idx'][i_file][1], a_min=start_frame, a_max=PARAMETERS['n_frames']))
            # Make destination slice
            destination_slice = (slice(start_frame, end_frame), slice(None), slice(None))
            # Copy frames
            frame_first_file[destination_slice] = TIFF.transpose((0, 2, 1))
            # Flush data to disk
            frame_first_file.flush()

        # Complete writing to disk
        log('Finishing file to disk')
        # Close files (flush data to disk)
        del frame_first_file

        # Compute projections for each condition
        log('Creating projections file in \'%s\'' % PARAMETERS['filename_projections'])
        # first_frame_file
        frame_first_file = np.memmap(PARAMETERS['filename_frame_first'], dtype=PARAMETERS['dtype'], mode="r", shape=destination_shape)
        # Make file for projections
        with h5py.File(PARAMETERS['filename_projections'], 'w+', libver='latest') as projections_file:
            analyzed_datasets = list(projections_file.keys())
            for i_file in range(n_tiff_files):
                log('Processing file %i/%i: %s' % (i_file + 1, n_tiff_files, PARAMETERS['condition_names'][i_file]))
                # Get frames of interest
                start_frame = int(np.clip(PARAMETERS['frames_idx'][i_file][0] - 1, a_min=0, a_max=PARAMETERS['n_frames']))
                end_frame = int(np.clip(PARAMETERS['frames_idx'][i_file][1], a_min=start_frame, a_max=PARAMETERS['n_frames']))
                frames = frame_first_file[start_frame:end_frame, :, :]
                # Make group and get list of analyzed datasets
                if PARAMETERS['condition_names'][i_file] in analyzed_datasets:
                    hdf5_group = projections_file[PARAMETERS['condition_names'][i_file]]
                else:
                    hdf5_group = projections_file.create_group('%s' % PARAMETERS['condition_names'][i_file])
                computed_projections = list(hdf5_group.keys())

                # Compute projection frame
                for projection_type in PROJECTIONS_TYPES:
                    # Remove spaces in name so it can be used as h5 dataset name
                    projection_type = projection_type.replace(' ', '_')
                    # Compute projection
                    if projection_type == 'mean':
                        values = np.mean(frames, axis=0)

                    elif projection_type == 'median':
                        values = np.median(frames, axis=0)

                    elif projection_type == 'max':
                        values = np.max(frames, axis=0)

                    elif projection_type == 'standard_deviation':
                        values = np.std(frames, axis=0)

                    elif projection_type == 'correlation':
                        # Compute cross-correlation after smoothing video in time(no NaNs allowed)
                        time_window = int(np.ceil(GC['correlation_time_smoothing_window'] * GC['frame_rate']))
                        frames_smoothed = temporal_smooth(frames.transpose((1, 2, 0)), time_window)
                        frames_smoothed[np.isnan(frames_smoothed)] = 0
                        values = cross_correlate_image(frames_smoothed)

                    else:
                        raise ValueError('Unknown projection type: \'%s\'' % projection_type)

                    # Store data
                    if projection_type in computed_projections:
                        del hdf5_group[projection_type]
                    hdf5_group.create_dataset(projection_type, data=values)

        # Make sure memory-mapped file is unlinked
        del projections_file
        memory_garbage_collector.collect()

    if not os.path.exists(PARAMETERS['filename_time_first']) or overwrite_time_first:
        do_convert_time_first = True
        log('Creating time-first file in \'%s\'' % PARAMETERS['filename_time_first'])
        time_first_file = np.memmap(PARAMETERS['filename_time_first'], dtype=PARAMETERS['dtype'], mode="w+", shape=(PARAMETERS['n_pixels'], PARAMETERS['n_frames']))

        # Memory-map frame-first file for reading
        frame_first_file = np.memmap(PARAMETERS['filename_frame_first'], dtype=PARAMETERS['dtype'], mode="r", shape=(PARAMETERS['n_frames'], PARAMETERS['frame_height'], PARAMETERS['frame_width']))

        # Calculate number of chunks
        n_chunks = int(np.ceil(np.float64(PARAMETERS['n_frames']) / n_frames_per_chunk))
        if n_chunks == 1:
            n_frames_per_chunk = PARAMETERS['n_frames']
        # Loop through chunks
        for i_chunk in range(n_chunks):
            if n_chunks > 1:
                log('\tReshaping chunk %i/%i' % (i_chunk + 1, n_chunks))
            # Get edges of chunk
            start_frame = i_chunk * n_frames_per_chunk
            end_frame = np.clip(start_frame + n_frames_per_chunk, a_min=start_frame + 1, a_max=PARAMETERS['n_frames'])
            # Get frames
            frames = frame_first_file[start_frame:end_frame, :, :].copy()
            # Replace NaNs with 0s
            frames[np.isnan(frames)] = 0
            # Copy data collapsing all pixels in a row
            n_frames_read = frames.shape[0]
            time_first_file[:, start_frame:end_frame] = frames.reshape((n_frames_read, -1)).transpose()
            # Flush data to disk
            time_first_file.flush()

        # Complete writing
        log('Finishing file to disk')
        # Close files (flush data to disk)
        time_first_file.flush()
        del time_first_file
        # Make sure all memory-mapped files are unlinked
        memory_garbage_collector.collect()

    if do_convert_frame_first or do_convert_time_first:
        log('Finished file conversion')

    # Make sure all memory-mapped files are unlinked
    memory_garbage_collector.collect()

    # Store parameters to disk
    parameters_filename = os.path.join(output_folder, 'parameters.json.txt')
    # Convert data types of some fields
    fields_from_num_to_str = ['n_frames', 'frame_height', 'frame_width', 'n_pixels']
    for i in fields_from_num_to_str:
        PARAMETERS[i] = int(PARAMETERS[i])
    PARAMETERS['dtype'] = str(PARAMETERS['dtype'])
    # Write to file
    json.dump(PARAMETERS, open(parameters_filename, 'w+'))
    # Log outcome
    log('Parameters file stored in \'%s\'' % parameters_filename)

    return parameters_filename


################################################################################
# Memory
################################################################################
def available_memory_GB(memory_cap=0.90):
    # Get the amount of available memory in GB
    memory = psutil.virtual_memory()
    total_memory_GB = memory[0] / 1024. / 1024. / 1024.

    # Use only up to 90% of the available memory
    memory_cap_GB = total_memory_GB * memory_cap
    memory_used_GB = memory[3] / 1024. / 1024. / 1024.
    memory_usable_GB = memory_cap_GB - memory_used_GB

    return memory_usable_GB

def max_chunk_size(n_bytes=4, memory_cap=0.90):
    memory_usable_GB = available_memory_GB(memory_cap=memory_cap)

    # Re-convert to bits and divide by the bit_size of the array to load in memory
    memory_usable = np.floor(memory_usable_GB * 1024. * 1024. * 1024.).astype(np.int64)
    n_elements = np.floor(memory_usable / float(n_bytes)).astype(np.int64)

    # Make sure the number of elements is positive
    if n_elements < 1:
        n_elements = 1

    return n_elements


################################################################################
# Console
################################################################################
def log(message):
    sys.stdout.write(message + '\n')
    sys.stdout.flush()
