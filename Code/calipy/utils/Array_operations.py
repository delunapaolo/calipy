# Numerical packages
import re

import cv2
import numpy as np


def temporal_smooth(stack_in, window_size, add_last_frame=True):
    """

    :param stack_in:
    :param window_size:
    :param add_last_frame:
    :return:
    """
    # Get shape of data
    frame_height, frame_width, n_frames = stack_in.shape

    # Check inputs
    if window_size > n_frames:
        window_size = n_frames

    # Pre-allocate output variable
    stack_out = np.zeros((frame_height, frame_width, n_frames), dtype=np.find_common_type((np.float32, stack_in.dtype), []))
    # Loop until the frame before the last
    for i_frame in range(n_frames - 1):
        # Get index of first and last frame
        start_frame = i_frame
        end_frame = min([start_frame + window_size, n_frames]) + 1
        # Compute mean
        stack_out[:, :, i_frame] = np.nanmean(stack_in[:, :, start_frame:end_frame].astype(np.float32), axis=-1)

    # Add last frame
    if add_last_frame:
        stack_out[:, :, -1] = stack_in[:, :, -1].copy()
    else:
        stack_out = stack_out[:, :, :-1]

    return stack_out


def cross_correlate_image(data, window_size=1, n_frames_per_image=None, squeeze=True):
    """

    :param data:
    :param window_size:
    :param n_frames_per_image:
    :param squeeze:
    :return:
    """
    # Get shape of data
    frame_height, frame_width, n_frames = data.shape

    # Adjust user input
    if n_frames_per_image is None:
        n_frames_per_image = n_frames

    # Get number of chunks
    n_chunks = int(np.ceil(n_frames / n_frames_per_image))

    # Allocate output array
    CCimage = np.zeros((frame_height, frame_width, n_chunks), dtype=np.find_common_type((np.float32, data.dtype), []))
    # Loop through chunks
    for i_chunk in range(n_chunks):
        # Get edges of chunk (and account for last chunk)
        start_sample = i_chunk * n_frames_per_image
        end_sample = min([start_sample + n_frames_per_image, n_frames])
        # Slice data
        this_data = data[:, :, start_sample:end_sample].copy()
        n_frames_in_chunk = this_data.shape[-1]

        # Allocate matrix for computation
        ccimage = np.zeros((frame_height, frame_width), dtype=this_data.dtype)
        # Compute cross-correlation
        # converted to python from MATLAB (https://labrigger.com/blog/2013/06/13/local-cross-corr-images/)
        for y in range(window_size, frame_height - window_size):
            for x in range(window_size, frame_width - window_size):
                # Center pixel
                a = this_data[y, x, :]  # Extract center pixel's time course and subtract its mean
                center_pixel = (a - np.mean(a, axis=-1)).reshape(1, 1, n_frames_in_chunk)
                # Autocorrelation, for normalization later
                ac_center_pixel = np.sum(center_pixel ** 2, axis=-1)

                # Neighborhood
                a = this_data[y-window_size:y+window_size+1, x-window_size:x+window_size+1, :]  # Extract the neighborhood
                neighboring_pixels = a - np.expand_dims(np.mean(a, axis=-1), axis=2)
                ac_neighboring_pixels = np.sum(neighboring_pixels ** 2, axis=-1)  # Autocorrelation, for normalization later

                # Cross-correlation
                product_pixels = center_pixel * neighboring_pixels
                ac_product = ac_center_pixel * ac_neighboring_pixels
                # Cross-correlation with normalization
                ccs = divide0(np.sum(product_pixels, axis=-1), np.sqrt(ac_product), replace_with=0)
                # Delete the middle point
                all_idx = list(range(ccs.size))
                all_idx.remove(int((ccs.size + 1) / 2) - 1)
                ccs = ccs.transpose().ravel()[all_idx]

                # Get the mean cross-correlation with the local neighborhood
                ccimage[y, x] = ccs.mean()

        # Store this matrix
        CCimage[:, :, i_chunk] = ccimage

    # Remove extra dimension, if not necessary
    if squeeze:
        CCimage = CCimage.squeeze()

    return CCimage


def divide0(a, b, replace_with):
    """Divide two numbers but replace its result if division is not possible,
    e.g., when dividing a number by 0. No type-checking or agreement between
    dimensions is performed. Be careful!

    :param a: Numerator.
    :param b: Denominator.
    :param replace_with: If a/b is not defined return this number instead.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        if isinstance(c, np.ndarray):
            c[np.logical_not(np.isfinite(c))] = replace_with
        else:
            if not np.isfinite(c):
                c = replace_with

    return c


def idx2range(idx):
    # Convert to numpy array
    if not type(idx) is np.ndarray:
        idx = np.array([idx], dtype=int).ravel()

    # Make sure input is sorted
    idx = np.sort(idx)

    if idx.shape[0] > 1:
        # Find discontinuities in index
        dataIDX = np.atleast_2d(np.unique(np.hstack((0, np.where(np.diff(idx) > 1)[0]+1)))).transpose()
        dataIDX = np.hstack((dataIDX, np.atleast_2d(np.hstack((dataIDX[1:,0]-1, idx.shape[0]-1))).transpose()))
        # Get original values
        dataIDX = idx[dataIDX]

        # Add column for duration
        dataIDX = np.hstack((dataIDX, np.atleast_2d(dataIDX[:,1] - dataIDX[:,0] + 1).transpose()))

    else:
        dataIDX = np.atleast_2d(np.array([idx, idx, 1]))

    return dataIDX


def expand_indices(start, end):
    if not isinstance(start, np.ndarray):
        start = np.array([start], dtype=int)
    if not isinstance(end, np.ndarray):
        end = np.array([end], dtype=int)
    lens = end - start
    np.cumsum(lens, out=lens)
    i = np.ones(lens[-1], dtype=int)
    i[0] = start[0]
    i[lens[:-1]] += start[1:]
    i[lens[:-1]] -= end[:-1]
    np.cumsum(i, out=i)
    return i


def transform_image_from_parameters(image, scale, angle, offset, return_matrices=False):
    # Make sure image is a numpy array
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    # Convert to float32 to work with opencv
    input_needs_conversion = image.dtype != np.float32

    # Get size of image before being transformed
    frame_size_width = image.shape[1]
    frame_size_height = image.shape[0]
    original_canvas_size = [frame_size_height, frame_size_width]
    # Find out whether the canvas size is an odd number
    odd_number_of_pixels = [c/2. > np.floor(c/2.) for c in original_canvas_size]

    # Scaling
    if np.any(scale != 1):
        # Convert image if necessary
        if input_needs_conversion:
            image = image.astype(np.float32)
            input_needs_conversion = False

        # Get more appropriate interpolation method
        if scale.mean() > 1:
            interpolation = cv2.INTER_CUBIC
        else:
            interpolation = cv2.INTER_AREA
        # Apply scaling
        image = cv2.resize(image, None, fx=scale[1], fy=scale[0], interpolation=interpolation)
        # Store parameters in memory
        scaling_matrix = [scale[1], scale[0], interpolation]
    else:
        scaling_matrix = np.array([1, 1], dtype=int)

    # Determine a good canvas size to allow both rotation and translation
    frame_size_width = image.shape[1]
    frame_size_height = image.shape[0]
    new_canvas_size = [frame_size_height, frame_size_width]
    canvas_size = np.ceil(max(new_canvas_size))
    if angle != 0:  # Add space for complete rotation (e.g., 45 degrees)
        canvas_size *= np.sqrt(2)
    if np.any(offset != 0):
        canvas_size += (np.abs(offset).max() * 2)  # Add double the max offset to allow offsets in any direction
    # Make it an integer multiple of 2
    canvas_size = np.ceil(canvas_size)
    canvas_size = int(np.floor(canvas_size / 2.) * 2.)
    # The same size is used for width and height, for simplicity
    canvas_size = [canvas_size, canvas_size]
    # If original canvas size was an odd number make this one an odd number, too,
    # so the original image can be perfectly centered in the canvas
    if odd_number_of_pixels[0]:
        canvas_size[0] += 1
    if odd_number_of_pixels[1]:
        canvas_size[1] += 1
    # Add padding to account for any residual aliasing (4 pixels should suffice)
    canvas_size = [c + 4 for c in canvas_size]
    # Compute margins around the image
    canvas_margin = (int((canvas_size[0] - image.shape[0]) / 2.), int((canvas_size[1] - image.shape[1]) / 2.))
    # Allocate canvas
    canvas = np.nan * np.zeros((canvas_size[0], canvas_size[1]), dtype=np.float32)
    # Place image in the center
    canvas[canvas_margin[0]:canvas_margin[0] + image.shape[0], canvas_margin[1]:canvas_margin[1] + image.shape[1]] = image
    # Re-calculate position of center
    center = np.floor([canvas_size[0] / 2., canvas_size[1] / 2.]).astype(int)

    # Rotation
    rotation_matrix = angle
    if angle != 0:
        # Convert image if necessary
        if input_needs_conversion:
            image = image.astype(np.float32)
        # For simple cases, do not use opencv
        if angle % 90 == 0:
            if angle == 90:
                image = cv2.transpose(canvas)
                image = cv2.flip(image, 0)
            elif angle == -90 or angle == 270:
                image = cv2.transpose(canvas)
                image = cv2.flip(image, 1)
            elif angle == 180 or angle == -180:
                image = cv2.flip(canvas, -1)
        else:
            # Get transformation matrix and apply it to the image
            rotation_matrix = cv2.getRotationMatrix2D(tuple(center), angle, 1)
            image = cv2.warpAffine(canvas, rotation_matrix, tuple(canvas_size), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    else:
        image = canvas.copy()

    # Translation / Cut-out of the visible portion only
    # Get position of new center
    new_center = center - np.round(offset).astype(int)  # Offset is positive because we move the viewport in the opposite direction to obtain the desired offset in the image
    # Cut image from canvas around the new center
    half_height = int(np.floor(original_canvas_size[0] / 2.))
    half_width = int(np.floor(original_canvas_size[1] / 2.))
    top = new_center[0]-half_height
    left = new_center[1]-half_width
    # Make slice object and use it
    section = np.s_[top:top+original_canvas_size[0], left:left+original_canvas_size[1]]
    image = image[section]
    # Store parameters
    translation_matrix = [[top, top+original_canvas_size[0], left, left+original_canvas_size[1]], canvas_size]

    # Return outputs
    if return_matrices:
        return image, scaling_matrix, rotation_matrix, translation_matrix
    else:
        return image


def transform_image_from_matrices(image, scaling_matrix, rotation_matrix, translation_matrix):
    # Make sure image is a numpy array
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    # Convert to float32 to work with opencv
    input_needs_conversion = not image.dtype is np.float32

    # Convert translation matrix to slice object
    canvas_size = tuple(translation_matrix[1])
    translation_matrix = np.s_[translation_matrix[0][0]:translation_matrix[0][1], translation_matrix[0][2]:translation_matrix[0][3]]

    # Scaling
    if len(scaling_matrix) == 3 and np.any(scaling_matrix[:2] != 1):
        # Convert image if necessary
        if input_needs_conversion:
            image = image.astype(np.float32)
            input_needs_conversion = False
        # Apply transformation
        image = cv2.resize(image, None, fx=scaling_matrix[0], fy=scaling_matrix[1], interpolation=int(scaling_matrix[2]))

    # Make new canvas to allow both rotation and translation
    canvas = np.nan * np.zeros((canvas_size[0], canvas_size[1]), dtype=np.float32)
    canvas_margin = (int((canvas_size[0]-image.shape[0])/2), int((canvas_size[1]-image.shape[1])/2))
    canvas[canvas_margin[0]:canvas_margin[0]+image.shape[0], canvas_margin[1]:canvas_margin[1]+image.shape[1]] = image

    # Rotation
    if not isinstance(rotation_matrix, np.ndarray) and rotation_matrix == 0:
        image = canvas.copy()
    else:
        # Convert image if necessary
        if input_needs_conversion:
            image = image.astype(np.float32)
        # Apply transformation
        if not isinstance(rotation_matrix, np.ndarray):
            angle = rotation_matrix
            if angle % 90 == 0:
                if angle == 90:
                    image = cv2.transpose(canvas)
                    image = cv2.flip(image, 0)
                elif angle == -90 or angle == 270:
                    image = cv2.transpose(canvas)
                    image = cv2.flip(image, 1)
                elif angle == 180 or angle == -180:
                    image = cv2.flip(canvas, -1)
        else:
            image = cv2.warpAffine(canvas, rotation_matrix, None, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)

    # Translation
    image = image[translation_matrix].copy()

    return image


###############################################################################
# https://nedbatchelder.com/blog/200712/human_sorting.html
###############################################################################
def natural_sort(s):
    """ Sort the given list in the way that humans expect."""
    def alphanum_key(s):
        return [int(c) if c.isdigit() else c for c in re.split('([0-9]+)', s)]
    s.sort(key=alphanum_key)
    return s
