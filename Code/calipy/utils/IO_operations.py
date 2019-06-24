# System packages
import sys
import os
import h5py
import psutil
import json
import gc as memory_garbage_collector

# Numerical packages
import numpy as np

# Local repository
from third_party import tifffile as tiff
from calipy.general_configs import default as GC
from calipy.utils.Array_operations import temporal_smooth, cross_correlate_image, natural_sort

PROJECTIONS_TYPES = ['mean', 'median', 'max', 'standard_deviation', 'correlation']


def prepare_data_for_GUI(tiff_folder, overwrite_frame_first=False, overwrite_time_first=False):
    # Initialize output dictionary
    PARAMETERS = dict()
    PARAMETERS['dataset_ID'] = os.path.split(tiff_folder)[-1]
    # Initialize conversion flags
    do_convert_frame_first = False
    do_convert_time_first = False

    # Iterate through tiff files to find width and height of each field of view (FOV)
    files = os.listdir(tiff_folder)
    tiff_files = natural_sort([i for i in files if i.lower().endswith('.tif') or i.lower().endswith('.tiff')])
    n_tiff_files = len(tiff_files)

    log('Reading info on files')
    FOV_size = np.zeros((n_tiff_files, 2), dtype=int)
    n_frames_per_file = np.zeros((n_tiff_files, ), dtype=int)
    TIFF_dtype = np.zeros((n_tiff_files, ), dtype=object)
    for i_file in range(n_tiff_files):
        # Read file and get its shape
        filename = os.path.join(tiff_folder, tiff_files[i_file])
        TIFF = tiff.imread(filename)
        TIFF_shape = TIFF.shape
        if len(TIFF_shape) == 4:
            TIFF = np.mean(TIFF, axis=3)
            TIFF_shape = TIFF.shape
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
    PARAMETERS['sessions_last_frame'] = n_frames_per_file.tolist()
    PARAMETERS['condition_names'] = [os.path.splitext(i)[0] for i in tiff_files]

    # Calculate the number of frames that can be held in memory while reading / writing final file
    temp = np.empty(shape=(1, 1), dtype=TIFF_data_type)
    n_frames_per_chunk = max_chunk_size(n_bytes=temp.itemsize, memory_cap=0.90)

    # Make filenames of data files
    PARAMETERS['filename_frame_first'] = os.path.join(tiff_folder, 'stack_frame_first.dat')
    PARAMETERS['filename_time_first'] = os.path.join(tiff_folder, 'stack_time_first.dat')
    PARAMETERS['filename_projections'] = os.path.join(tiff_folder, 'projections.hdf5')
    # Make filename of output file
    PARAMETERS['filename_ROIs'] = os.path.join(tiff_folder, 'ROIs_info.mat')

    # Create frame-first file, if it doesn't exist
    if not os.path.exists(PARAMETERS['filename_frame_first']) or not os.path.exists(PARAMETERS['filename_projections']) or overwrite_frame_first:
        do_convert_frame_first = True
        log('Creating frame-first file in \'%s\'' % PARAMETERS['filename_frame_first'])
        # Make file for data
        n_pixels_FOV = PARAMETERS['frame_height'] * PARAMETERS['frame_width']
        n_pixels = n_pixels_FOV * PARAMETERS['n_frames']
        frame_first_file = np.memmap(PARAMETERS['filename_frame_first'], dtype=PARAMETERS['dtype'], mode="w+", shape=(n_pixels, ))
        # Compute beginning and end of each file
        # n_samples_per_file = n_frames_per_file * PARAMETERS['frame_height'] * PARAMETERS['frame_width']
        # end_condition = n_samples_per_file.cumsum().reshape(-1, 1)
        # start_condition = np.vstack(([0], end_condition[:-1] + 1))
        # sample_idx = np.hstack((start_condition, end_condition))

        for i_file in range(n_tiff_files):
            # Read tiff file
            filename = os.path.join(tiff_folder, tiff_files[i_file])
            log('Copying \'%s\'' % tiff_files[i_file])
            TIFF = tiff.imread(filename)
            TIFF_shape = TIFF.shape
            if len(TIFF_shape) == 4:
                TIFF = np.mean(TIFF, axis=3)
            n_frames = TIFF.shape[0]

            # Calculate number of chunks
            n_frames_per_chunk_this_file = n_frames_per_chunk
            n_chunks_this_file = int(np.ceil(np.float64(n_frames) / n_frames_per_chunk_this_file))
            if n_chunks_this_file == 1:
                n_frames_per_chunk_this_file = n_frames

            # Get index of first frame of this file
            if i_file == 0:
                first_frame_this_file = 0
            else:
                first_frame_this_file = n_frames_per_file[i_file - 1]
            first_sample_this_file = first_frame_this_file * n_pixels_FOV

            # Loop through chunks
            for i_chunk in range(n_chunks_this_file):
                if n_chunks_this_file > 1:
                    log('Copying chunk %i/%i' % (i_chunk + 1, n_chunks_this_file))
                # Get edges of chunk
                start_frame = i_chunk * n_frames_per_chunk_this_file
                end_frame = np.clip(start_frame + n_frames_per_chunk_this_file, a_min=start_frame + 1, a_max=PARAMETERS['n_frames'])
                # Get frames
                frames = TIFF[start_frame:end_frame, :, :]
                # Replace NaNs with 0s
                frames[np.isnan(frames)] = 0
                # Copy frames
                sample_start = first_sample_this_file + start_frame * n_pixels_FOV
                sample_end = first_sample_this_file + end_frame * n_pixels_FOV
                frame_first_file[sample_start:sample_end] = frames.transpose((1, 2, 0)).ravel()
                # Flush data to disk
                frame_first_file.flush()

        # Complete writing to disk
        log('Finishing file to disk')
        # Close files (flush data to disk)
        frame_first_file.flush()
        del frame_first_file

        # Compute projections for each condition
        log('Creating projections file in \'%s\'' % PARAMETERS['filename_projections'])
        # first_frame_file
        frame_first_file = np.memmap(PARAMETERS['filename_frame_first'], dtype=PARAMETERS['dtype'], mode="r", shape=(PARAMETERS['frame_height'], PARAMETERS['frame_width'], PARAMETERS['n_frames']))
        # Make file for projections
        projections_file = h5py.File(PARAMETERS['filename_projections'], 'w', libver='latest')
        for i_file in range(n_tiff_files):
            # Get frames of interest
            frame_indices = frames_idx[i_file, :]
            frames = frame_first_file[:, :, frame_indices[0]:frame_indices[1]]

            # Initialize group
            hdf5_group = projections_file.create_group('%s' % PARAMETERS['condition_names'][i_file])

            # Compute projection frame
            for projection_type in PROJECTIONS_TYPES:
                if projection_type == 'mean':
                    values = np.mean(frames, axis=-1)

                elif projection_type == 'median':
                    values = np.median(frames, axis=-1)

                elif projection_type == 'max':
                    values = np.max(frames, axis=-1)

                elif projection_type == 'standard_deviation':
                    values = np.std(frames, axis=-1)

                elif projection_type == 'correlation':
                    # Compute cross-correlation after smoothing video in time(no NaNs allowed)
                    time_window = int(np.ceil(GC['correlation_time_smoothing_window'] * GC['frame_rate']))
                    frames_smoothed = temporal_smooth(frames, time_window)
                    frames_smoothed[np.isnan(frames_smoothed)] = 0
                    values = cross_correlate_image(frames_smoothed)

                else:
                    raise ValueError('Unknown projection type: \'%s\'' % projection_type)

                # Store data
                hdf5_group.create_dataset(projection_type, data=values)

        # Finish writing to disk
        projections_file.flush()
        projections_file.close()
        # Make sure memory-mapped file is unlinked
        memory_garbage_collector.collect()

    if not os.path.exists(PARAMETERS['filename_time_first']) or overwrite_time_first:
        do_convert_time_first = True
        log('Creating time-first file in \'%s\'' % PARAMETERS['filename_time_first'])
        time_first_file = np.memmap(PARAMETERS['filename_time_first'], dtype=PARAMETERS['dtype'], mode="w+", shape=(PARAMETERS['n_pixels'], PARAMETERS['n_frames']))

        # Memory-map frame-first file for reading
        frame_first_file = np.memmap(PARAMETERS['filename_frame_first'], dtype=PARAMETERS['dtype'], mode="r", shape=(PARAMETERS['frame_height'], PARAMETERS['frame_width'], PARAMETERS['n_frames']))

        # Calculate number of chunks
        n_chunks = int(np.ceil(np.float64(PARAMETERS['n_frames']) / n_frames_per_chunk))
        if n_chunks == 1:
            n_frames_per_chunk = PARAMETERS['n_frames']
        # Loop through chunks
        for i_chunk in range(n_chunks):
            if n_chunks > 1:
                log('Reshaping chunk %i/%i' % (i_chunk + 1, n_chunks))
            # Get edges of chunk
            start_frame = i_chunk * n_frames_per_chunk
            end_frame = np.clip(start_frame + n_frames_per_chunk, a_min=start_frame + 1, a_max=PARAMETERS['n_frames'])
            # Get frames and transpose into expected shape for a video file
            frames = frame_first_file[:, :, start_frame:end_frame].transpose((2, 0, 1)).copy()
            n_frames_read = frames.shape[0]
            # Replace NaNs with 0s
            frames[np.isnan(frames)] = 0
            # Copy data collapsing all pixels in a row
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
    parameters_filename = os.path.join(tiff_folder, 'parameters.json.txt')
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
