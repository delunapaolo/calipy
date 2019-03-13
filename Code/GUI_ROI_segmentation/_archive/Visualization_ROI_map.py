# Add Utilities folder to system path
import os, sys
script_path = os.path.realpath(__file__)
root_path = os.path.split(os.path.dirname(script_path))[0]
sys.path.insert(0, root_path)

# System packages
import sys
from random import shuffle

# Graphical packages
from matplotlib import pyplot as plt
import seaborn as sns

# Numerical packages
import numpy as np

# Local repository
from Utilities import matlab_file
from GUI_ROI_segmentation.IO_operations import log




def main(filename):
    # Load file
    ROIs = matlab_file.load(filename, variable_names='ROIs')

    # Get number of ROIs and image size
    image_height, image_width, n_ROIs = ROIs.shape

    # Multiply each mask for a number that makes that ROI have a unique number
    ids = np.arange(1, n_ROIs+1, dtype=ROIs.dtype)
    shuffle(ids)
    ROIs *= np.atleast_3d(ids).reshape(1, 1, n_ROIs)

    # Sum all ROIs
    ROIs_map = np.sum(ROIs, axis=2)

    mask = np.zeros_like(ROIs_map)
    mask[ROIs_map==0] = True

    fig = plt.figure()
    plt.clf(); ax = sns.heatmap(ROIs_map, mask=mask, square=True, cbar=False, xticklabels=False, yticklabels=False, cmap='tab10')
    # Draw borders
    ax.axhline(0, color='k')
    ax.axhline(image_height, color='k')
    ax.axvline(0, color='k')
    ax.axvline(image_width, color='k')
    # Make figure tighter
    fig.tight_layout()

    print('done')



################################################################################
# Direct call
################################################################################
if __name__ == "__main__":
    # Get user inputs
    if len(sys.argv) <= 1:
        ROI_info_filename = r'V:\2p_imaging_pain\3_segmentation\MA_3_ROI_info.mat'
    else:
        ROI_info_filename = sys.argv[1]

    main(ROI_info_filename)
