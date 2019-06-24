# System packages
from copy import copy
import collections

# Numerical packages
import numpy as np


###############################################################################
# Colormap-related
###############################################################################
def default_colors(normalize_01=True):
    colors = [[11, 198, 255],
              [255, 170, 0],
              [0, 170, 0],
              [255, 85, 255],
              [0, 0, 255],
              [255, 0, 4],
              [190, 92, 35],
              [255, 6, 114]]
    colors = np.array(colors, dtype=int)
    if normalize_01:
        colors /= 255.
    colors = list(zip(*colors.transpose()))

    return colors


def cmapToColormap(cmap, nTicks=16):
    """
    Converts a Matplotlib cmap to pyqtgraph colormaps. No dependency on matplotlib.

    Parameters:
    *cmap*: Cmap object. Imported from matplotlib.cm.*
    *nTicks*: Number of ticks to create when dict of functions is used. Otherwise unused.
    """

    # Case #1: a dictionary with 'red'/'green'/'blue' values as list of ranges (e.g. 'jet')
    # The parameter 'cmap' is a 'matplotlib.colors.LinearSegmentedColormap' instance ...
    if hasattr(cmap, '_segmentdata'):
        colordata = getattr(cmap, '_segmentdata')
        if ('red' in colordata) and isinstance(colordata['red'], collections.Sequence):

            # collect the color ranges from all channels into one dict to get unique indices
            posDict = {}
            for idx, channel in enumerate(('red', 'green', 'blue')):
                for colorRange in colordata[channel]:
                    posDict.setdefault(colorRange[0], [-1, -1, -1])[idx] = colorRange[2]

            indexList = list(posDict.keys())
            indexList.sort()
            # interpolate missing values (== -1)
            for channel in range(3):  # R,G,B
                startIdx = indexList[0]
                emptyIdx = []
                for curIdx in indexList:
                    if posDict[curIdx][channel] == -1:
                        emptyIdx.append(curIdx)
                    elif curIdx != indexList[0]:
                        for eIdx in emptyIdx:
                            rPos = (eIdx - startIdx) / (curIdx - startIdx)
                            vStart = posDict[startIdx][channel]
                            vRange = (posDict[curIdx][channel] - posDict[startIdx][channel])
                            posDict[eIdx][channel] = rPos * vRange + vStart
                        startIdx = curIdx
                        del emptyIdx[:]
            for channel in range(3):  # R,G,B
                for curIdx in indexList:
                    posDict[curIdx][channel] *= 255

            rgb_list = [[i, posDict[i]] for i in indexList]

        # Case #2: a dictionary with 'red'/'green'/'blue' values as functions (e.g. 'gnuplot')
        elif ('red' in colordata) and isinstance(colordata['red'], collections.Callable):
            indices = np.linspace(0., 1., nTicks)
            luts = [np.clip(np.array(colordata[rgb](indices), dtype=np.float), 0, 1) * 255 for rgb in ('red', 'green', 'blue')]
            rgb_list = zip(indices, list(zip(*luts)))

    # If the parameter 'cmap' is a 'matplotlib.colors.ListedColormap' instance, with the attributes 'colors' and 'N'
    elif hasattr(cmap, 'colors') and hasattr(cmap, 'N'):
        colordata = getattr(cmap, 'colors')
        # Case #3: a list with RGB values (e.g. 'seismic')
        if len(colordata[0]) == 3:
            indices = np.linspace(0., 1., len(colordata))
            scaledRgbTuples = [(rgbTuple[0] * 255, rgbTuple[1] * 255, rgbTuple[2] * 255) for rgbTuple in colordata]
            rgb_list = zip(indices, scaledRgbTuples)

        # Case #4: a list of tuples with positions and RGB-values (e.g. 'terrain')
        # -> this section is probably not needed anymore!?
        elif len(colordata[0]) == 2:
            rgb_list = [(idx, (vals[0] * 255, vals[1] * 255, vals[2] * 255)) for idx, vals in colordata]

    # Case #X: unknown format or datatype was the wrong object type
    else:
        raise ValueError("[cmapToColormap] Unknown cmap format or not a cmap!")

    # Convert the RGB float values to RGBA integer values
    return list([(pos, (int(r), int(g), int(b), 255)) for pos, (r, g, b) in rgb_list])


################################################################################
# class-related
################################################################################
def initialize_field(obj, field_name, value_type="class", initial_value=None, shape=None):
    """Initialize an empty object with <field_name> type and adds it to an
    object instance <obj>. This container can be accessed from <obj>.<field_name>

    :param obj: [object] The parent object in which to initialize a new field.
    :param field_name: [str] Name of the variable to add.
    :param value_type: [str] Type of object to initialize: "class", "list",
        "array" or "=".
    :param initial_value: Default attribute of the object. For "list" it is a
        scalar containing the length of the list; for "array" it is the shape of
        the array; for "class" it is ignored; for "=" it becomes equal to
        whatever <initial_value> is.
    """

    class subfield(dict):
        def __init__(self, name="subfield"):
            # Initialize the instance as a dictionary and assign a name to it
            dict.__init__(self)
            self.__name__ = name

            # Allow transformation between dot-call and key-call, i.e.,
            # dict.subfield works as dict["subfield"]
            self.__dict__ = self

    # Check whether the user inputted a list of fields to fill in
    if isinstance(field_name, str):
        field_name = [field_name]

    for f in field_name:
        value = None
        if value_type == "=":
            # Use user's input
            value = copy(initial_value)

        elif value_type == "class":
            # Use a dictionary to contain subfields
            value = subfield(name=f)

        elif value_type == "list":
            # Make a list of Nones
            value = np.empty(shape=shape, dtype=object).tolist()
            # If initial_value is provided, place it in each item of the list
            if initial_value is not None:
                # Handle special instructions
                if initial_value == 'list':
                    value = [list() for _ in value]
                elif initial_value == 'array':
                    value = [np.array([]) for _ in value]
                else:
                    value = [copy(initial_value) for _ in value]

        elif value_type == "array":
            # Make a numpy array
            value = np.empty(shape=shape, dtype=object)
            # If initial_value is provided, place it in each item of the array
            if initial_value is not None:
                # Handle special instructions
                rows, cols = np.unravel_index(np.arange(np.prod(shape)), shape)
                if initial_value == 'list':
                    for row, col in zip(rows, cols):
                        value[row, col] = list()
                elif initial_value == 'array':
                    for row, col in zip(rows, cols):
                        value[row, col] = np.array([])
                else:
                    for row, col in zip(rows, cols):
                        value[row, col] = copy(initial_value)

        # Assign value to object or dictionary
        if type(obj).__name__ == "subfield":
            obj[f] = value
        else:
            setattr(obj, f, value)


def make_new_ROI_id(already_existing):
    """Find the lowest number that can be used as cluster id."""
    # Get the set of all existing ids
    already_existing = np.array(already_existing, dtype=int)
    # If list is empty, return 1
    if already_existing.shape[0] == 0:
        return 1

    last_id = int(np.max(already_existing))
    return last_id + 1


def ask(question, answers, default_answer, type_validator=str):
    # Get number of answers
    if not isinstance(answers, list):
        answers = [answers]
    if answers[0] == '':
        n_answers = 0
    else:
        n_answers = len(answers)

    # Append answer hints to question
    if n_answers > 0 or str(default_answer) != '':
        question_to_show = question + ' ('
        if str(default_answer) != '':
            question_to_show += '[%s]' % str(default_answer)
            if n_answers > 0:
                question_to_show += '/'
        if n_answers > 0:
            question_to_show += '%s' % '/'.join([i for i in answers if str(i) != str(default_answer)])
        question_to_show += ')'

    else:
        question_to_show = question

    # Ask user for an answer
    while True:
        user_answer = input(question_to_show)
        # Set default option when user presses Enter
        if user_answer == '':
            if str(default_answer) == '':
                print('Please try again')
            else:
                user_answer = default_answer
        # Validate user answer
        try:
            user_answer = type_validator(user_answer)
        except ValueError:
            print('Answer type not allowed. Reply something that can be converted to \'%s\'' % repr(type_validator))
            continue

        # Stop if got an answer that is allowed, or if there are no good answers
        if user_answer in answers or n_answers == 0:
            break
        else:
            print('Please try again')

    return user_answer


