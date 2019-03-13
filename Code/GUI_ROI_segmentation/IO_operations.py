# System packages
import sys, os, h5py
import gc as memory_garbage_collector
import numpy as np


def log(message):
    sys.stdout.write(message + '\n')
    sys.stdout.flush()


def prepare_data_for_GUI(PARAMETERS):
    n_frames_per_chunk = 1000

    # Check which file should be used as input
    if os.path.exists(PARAMETERS['filename_local']):
        filename = PARAMETERS['filename_local']
    else:
        filename = PARAMETERS['filename_remote']

    # Open .mat file for reading (this context will automatically close the file on errors)
    with h5py.File(filename, 'r', libver='latest') as file_in:
        # Get last modification date of data file
        data_file_modification_time = os.path.getmtime(filename)

        # Convert files
        do_convert_data_frame_file = False  # Initialize switch
        if not os.path.exists(PARAMETERS['filename_data_frame']):
            do_convert_data_frame_file = True
        else:
            # Compare last modification time of this file to the original data
            # file: If data file is later, reconvert this file
            data_frame_modification_time = os.path.getmtime(PARAMETERS['filename_data_frame'])
            if data_file_modification_time > data_frame_modification_time:
                do_convert_data_frame_file = True

        if do_convert_data_frame_file:
            # Copy entire file in a memory-mapped numpy array
            log('Copying whole dataset in \'%s\'' % PARAMETERS['filename_data_frame'])
            file_data_frame = np.memmap(PARAMETERS['filename_data_frame'], dtype=np.float32, mode="w+", shape=(PARAMETERS['n_frames'], PARAMETERS['frame_height'], PARAMETERS['frame_width']))
            # Calculate number of chunks
            n_chunks = int(np.ceil(np.float64(PARAMETERS['n_frames']) / n_frames_per_chunk))
            if n_chunks == 1:
                n_frames_per_chunk = PARAMETERS['n_frames']
            # Loop through chunks
            for ichunk in range(n_chunks):
                log('Copying chunk %i/%i' % (ichunk + 1, n_chunks))
                # Get edges of chunk
                start_sample = ichunk * n_frames_per_chunk
                end_sample = np.clip(start_sample + n_frames_per_chunk, a_min=start_sample + 1, a_max=PARAMETERS['n_frames'])
                # Get frames
                frames = file_in['Y'][start_sample:end_sample, :, :]
                # Replace NaNs with 0s
                frames[np.isnan(frames)] = 0
                # Copy frames
                file_data_frame[start_sample:end_sample, 0:PARAMETERS['frame_height'], 0:PARAMETERS['frame_width']] = frames.copy()
                # Flush data to disk every 10 chunks
                if (ichunk + 1) % 10 == 0:
                    file_data_frame.flush()
            log('Finishing file to disk')
            # Close files (flush data to disk)
            file_data_frame.flush()
            del file_data_frame
            # Make sure memory-mapped file is unlinked
            memory_garbage_collector.collect()

        do_convert_data_time_file = False  # Initialize switch
        if not os.path.exists(PARAMETERS['filename_data_time']):
            do_convert_data_time_file = True
        else:
            # Compare last modification time of this file to the original data
            # file: If data file is later, reconvert this file
            data_time_modification_time = os.path.getmtime(PARAMETERS['filename_data_time'])
            if data_file_modification_time > data_time_modification_time:
                do_convert_data_time_file = True

        if do_convert_data_time_file:
            # Open file for writing
            log('Copying time-data in %s' % PARAMETERS['filename_data_time'])
            file_data_time = np.memmap(PARAMETERS['filename_data_time'], dtype=np.float32, mode="w+", shape=(PARAMETERS['n_pixels'], PARAMETERS['n_frames']))
            # Calculate number of chunks
            n_chunks = int(np.ceil(np.float64(PARAMETERS['n_frames']) / n_frames_per_chunk))
            if n_chunks == 1:
                n_frames_per_chunk = PARAMETERS['n_frames']
            # Loop through chunks
            for ichunk in range(n_chunks):
                log('Reshaping chunk %i/%i' % (ichunk + 1, n_chunks))
                # Get edges of chunk
                start_sample = ichunk * n_frames_per_chunk
                end_sample = np.clip(start_sample + n_frames_per_chunk, a_min=start_sample + 1, a_max=PARAMETERS['n_frames'])
                # Get frames
                frames = file_in['Y'][start_sample:end_sample, :, :]
                n_frames_read = frames.shape[0]
                # Replace NaNs with 0s
                frames[np.isnan(frames)] = 0
                # Copy data collapsing all pixels in a row
                file_data_time[0:PARAMETERS['n_pixels'], start_sample:end_sample] = frames.reshape((n_frames_read, -1)).transpose()
                # Flush data to disk every 10 chunks
                if (ichunk + 1) % 10 == 0:
                    file_data_time.flush()
            log('Finishing file to disk')
            # Close files (flush data to disk)
            file_data_time.flush()
            del file_data_time
            # Make sure all memory-mapped files are unlinked
            memory_garbage_collector.collect()

    if do_convert_data_frame_file or do_convert_data_time_file:
        log('Finished file conversion')

    # Make sure all memory-mapped files are unlinked
    memory_garbage_collector.collect()

    return PARAMETERS
