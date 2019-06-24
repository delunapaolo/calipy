# System
import os
import sys
import json
import numpy as np
from PyQt5.QtWidgets import QFileDialog

# Local repository
from calipy.calipy_GUI import CalIpy
from calipy.utils.Array_operations import natural_sort
from calipy.utils.IO_operations import prepare_data_for_GUI, log
from calipy.utils.Qt import Qt5_QtApp


if __name__ == '__main__':
    # Get user inputs
    if len(sys.argv) <= 1:
        app = Qt5_QtApp()
        # Set default folder to user's home folder
        home = os.path.expanduser('~')
        tiff_folder = QFileDialog.getExistingDirectory(None, 'Choose folder containing TIFF files', home)
        # Return if user canceled
        if tiff_folder == '':
            log('User canceled')
            exit(code=0)

        # Set default options for overwriting transformed files
        overwrite_frame_first = False
        overwrite_time_first = False
        overwrite_params = False

    else:  # User called this module from command line with arguments
        tiff_folder = sys.argv[1]
        overwrite_frame_first = sys.argv[2]
        overwrite_time_first = sys.argv[3]

    # Check that there are TIFF files in the selected folder
    files = os.listdir(tiff_folder)
    tiff_files = [i for i in files if i.lower().endswith('.tif') or i.lower().endswith('.tiff')]
    if len(tiff_files) == 0:
        raise OSError('No TIFF files in \'%s\'' % tiff_folder)
    # Sort files alphabetically
    tiff_files = natural_sort(tiff_files)
    n_tiff_files = len(tiff_files)
    log('%i files are present' % n_tiff_files)

    # Check that files with concatenated data exist
    filename_frame_first = os.path.join(tiff_folder, 'stack_frame_first.dat')
    filename_time_first = os.path.join(tiff_folder, 'stack_time_first.dat')
    if not os.path.exists(filename_frame_first) or not os.path.exists(filename_time_first) or overwrite_frame_first or overwrite_time_first or overwrite_params:
        log('Concatenating files')
        parameters_filename = prepare_data_for_GUI(tiff_folder, overwrite_frame_first=overwrite_frame_first, overwrite_time_first=overwrite_time_first)
    else:
        log('Concatenated files are present')
        parameters_filename = os.path.join(tiff_folder, 'parameters.json.txt')

    # Load parameters from file stored on disk
    PARAMETERS = json.load(open(parameters_filename, 'r'))

    # Make sure these fields are numpy arrays
    PARAMETERS['sessions_last_frame'] = np.array([PARAMETERS['sessions_last_frame']], dtype=np.int64)
    PARAMETERS['frames_idx'] = np.vstack(PARAMETERS['frames_idx']).astype(np.int64)

    # Launch GUI
    GUI = CalIpy(PARAMETERS)
