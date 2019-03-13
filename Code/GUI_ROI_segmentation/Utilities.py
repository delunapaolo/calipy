# System packages
import ctypes
from copy import copy
import collections

# Graphical packages
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import cv2

# Numerical packages
import numpy as np

# Local repository
from GUI_ROI_segmentation.general_configs import default as GUI_default


################################################################################
# Qt-related
################################################################################
class Qt_window(QtWidgets.QMainWindow):
    resized = QtCore.pyqtSignal()
    about_to_close = QtCore.pyqtSignal()

    def __init__(self):
        super(Qt_window, self).__init__()
        self.allowed_to_close = False

    def contextMenuEvent(self, event):
        pass

    def resizeEvent(self, event):
        """Called whenever the window is resized."""
        self.resized.emit()
        return super(Qt_window, self).resizeEvent(event)

    def closeEvent(self, event):
        """Called whenever the window receives the close command. We only proceed
        if we are allowed to do so."""
        self.about_to_close.emit()
        if self.allowed_to_close:
            event.accept()
            super(Qt_window, self).closeEvent(event)
        else:
            event.ignore()

class Qt5_QtApp(object):
    def __init__(self, appID=None, icon=None):
        """Make Qt app and set its icon."""
        self.app = QtWidgets.QApplication.instance()
        # If not already running, instantiate a new app
        if self.app is None:  # Quit existing instance
            self.app = QtWidgets.QApplication([])
        # Set app ID
        if appID is not None:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(appID)
        # Set app icon
        if icon is not None:
            app_icon = QtGui.QIcon(icon)
            self.app.setWindowIcon(app_icon)
        # Set 'fusion' style
        self.app.setStyle(QtWidgets.QStyleFactory.create('Fusion'))

class ImageView(pg.ImageView):
    def evalKeyState(self):
        if len(self.keysPressed) == 1:
            key = list(self.keysPressed.keys())[0]
            if key == QtCore.Qt.Key_Right:
                self.key_pressed.emit(key)
            elif key == QtCore.Qt.Key_Left:
                self.key_pressed.emit(key)
            elif key == QtCore.Qt.Key_PageUp:
                self.play(-1000)
            elif key == QtCore.Qt.Key_PageDown:
                self.play(1000)
        else:
            self.play(0)

def make_toolbar_button(window, text_or_icon_data, kind='text', fixed_size=(32, 32), **kwargs):
    # Create button
    btn = QtWidgets.QToolButton(window)
    # Set checkable state
    checkable = kwargs.get('checkable', False)
    btn.setCheckable(checkable)

    # Set default properties according to type
    if kind == 'text':
        # Add text and set font
        btn.setText(text_or_icon_data)
        btn.setFont(GUI_default['font_buttons'])
        # Make only text visible
        btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)

    elif kind == 'icon':
        # Set only icon to be visible
        btn.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        # If button is checkable, it will contain two icons (for the 2 states: unchecked and checked)
        if checkable:
            btn.setIcon(QtGui.QIcon(QtGui.QPixmap(text_or_icon_data[0])))  # Initialize as unchecked
        else:
            btn.setIcon(QtGui.QIcon(QtGui.QPixmap(text_or_icon_data)))

    # Set size policy
    if fixed_size is not None:
        btn.setFixedSize(QtCore.QSize(fixed_size[0], fixed_size[1]))

    return btn

def make_toolbar_action(window, text=None, icon=None, fixed_size=(48, 48)):
    # Create button
    btn = QtWidgets.QAction(window)

    # Set text
    if text is not None:
        btn.setIconText(' ' + text)
        btn.setFont(GUI_default['font_actions'])
    # Set icon
    if icon is not None:
        btn.setIcon(QtGui.QIcon(QtGui.QPixmap(icon)))

    return btn

def make_QSplitter(orientation):
    """Make a QSplitter object with default properties."""
    if orientation == "hor":
        ori = QtCore.Qt.Horizontal
    else:
        ori = QtCore.Qt.Vertical
    s = QtWidgets.QSplitter()
    s.setOrientation(ori)
    s.setChildrenCollapsible(False)
    s.setHandleWidth(7)
    s.setOpaqueResize(True)
    s.setStyleSheet("QSplitter::handle {border: 1px dashed #76797C;}"
                    "QSplitter::handle:pressed {background-color: #787876; border: 1px solid #76797C;}")
    return s

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
    colors = zip(*colors.transpose())

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

class CustomViewBox(pg.ViewBox):
    def __init__(self, *args, **kwds):
        pg.ViewBox.__init__(self, *args, **kwds)
        self.menu = QtWidgets.QMenu()

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() == QtCore.Qt.RightButton:
            # Swap button identity
            ev._button = QtCore.Qt.LeftButton
            pg.ViewBox.mouseDragEvent(self, ev, axis)
        elif ev.button() == QtCore.Qt.LeftButton:
            ev.ignore()

    def getMenu(self, ev):
        return QtWidgets.QMenu()

    def updateViewLists(self): pass
    @staticmethod
    def updateAllViewLists(): pass

def ProgressDialog(*args, **kwargs):
    # Create progress dialog
    dlg = pg.ProgressDialog(busyCursor=True, wait=0, *args, **kwargs)
    # Adjust labels size
    label = [ch for ch in dlg.children() if isinstance(ch, QtWidgets.QLabel)][0]
    label.setFont(GUI_default['font_buttons'])
    bar = [ch for ch in dlg.children() if isinstance(ch, QtWidgets.QProgressBar)][0]
    bar.setFont(GUI_default['font'])
    # The window is modal to the application and blocks input to all windows
    dlg.setWindowModality(QtCore.Qt.ApplicationModal)
    # Hide cancel button
    dlg.setCancelButton(None)
    dlg.adjustSize()
    # Disable window's close button
    dlg.setWindowFlags(dlg.windowFlags() & ~QtCore.Qt.WindowCloseButtonHint)
    # Make sure to show the dialog and return it
    dlg.show()

    return dlg

def MessageBox(message, title='', parent=None, box_type='question', button_Yes='Yes', button_No='No', default_answer='Yes', additional_buttons=None):
    # Create empty MessageBox
    msgBox = QtWidgets.QMessageBox(parent)

    # Make buttons
    QButton_Yes = msgBox.addButton(button_Yes, QtWidgets.QMessageBox.YesRole)
    QButton_No = msgBox.addButton(button_No, QtWidgets.QMessageBox.NoRole)
    # Add other buttons
    additional_QButtons = list()
    if additional_buttons is not None:
        if not hasattr(additional_buttons, '__iter__'):
            additional_buttons = [additional_buttons]
        for btn in additional_buttons:
            additional_QButtons.append(msgBox.addButton(btn, QtWidgets.QMessageBox.YesRole))

    # Set default action
    if default_answer.lower() == 'yes':
        msgBox.setDefaultButton(QButton_Yes)
    elif default_answer.lower() == 'no':
        msgBox.setDefaultButton(QButton_No)
    else:
        btn_idx = additional_buttons.index(default_answer)
        msgBox.setDefaultButton(additional_QButtons[btn_idx])
    # Set action associated t oEscape button
    msgBox.setEscapeButton(QButton_No)

    # Replace \n with <br>
    message = '<br />'.join(message.split('\n'))
    # Include message in HTML tags to change font size
    message_to_show = "<font size = %i> %s </font>" % (GUI_default['font'].pointSize()/2.5, message)
    # Add title and text
    if title != '':
        msgBox.setWindowTitle(title)
    msgBox.setText(message_to_show)
    # Add icon
    if box_type == 'question':
        msgBox.setIcon(QtWidgets.QMessageBox.Question)
    elif box_type == 'warning':
        msgBox.setIcon(QtWidgets.QMessageBox.Warning)
    # Execute widget
    msgBox.exec_()

    # Collect answer
    answer = msgBox.clickedButton()
    if answer == QButton_Yes:
        return button_Yes
    elif answer == QButton_No:
        return button_No
    else:
        # Get index of clicked button
        answer_idx = [answer == ii for ii in additional_QButtons].index(True)
        return additional_buttons[answer_idx]

class Slider(QtWidgets.QSlider):
    def __init__(self, minimum, maximum):
        super(Slider, self).__init__()
        self.setMinimum(minimum)
        self.setMaximum(maximum)
        self._range = maximum - minimum

    def mousePressEvent (self, event):
        mouse_click_position = event.x()
        new_value = self.minimum() + (self._range * mouse_click_position) / self.width()
        self.setValue(new_value)

class MultiColumn_QTable(QtWidgets.QTableWidget):
    def __init__(self, data, title, minimum_height=500):
        super(MultiColumn_QTable, self).__init__()

        # Store the maximum height allowed for long tables
        self.minimum_height = minimum_height

        # Make a stylesheet that creates consistent colors between selection states
        self.setStyleSheet("""QTableWidget:item:selected:active {background:#3399FF; color:white}
                            QTableWidget:item:selected:!active {background:gray; color:white}
                            QTableWidget:item:selected:disabled {background:gray; color:white}
                            QTableWidget:item:selected:!disabled {background:#3399FF; color:white}""")

        # Initialize field names
        self.allowed_interaction = None
        self._columns = [title]

        # Hide row numbers
        self.verticalHeader().setVisible(False)

        # Assign data to the table
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        self.data = data
        self.n_rows = self.data.shape[0]
        self.setRowCount(self.n_rows)
        self.setColumnCount(1)

        # Make table
        for irow in range(self.n_rows):
            item = self.data[irow]
            new_item = QtWidgets.QTableWidgetItem(item)
            new_item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            new_item.setFont(GUI_default['font'])
            self.setItem(irow, 0, new_item)

        # Set the header labels
        self.setHorizontalHeaderLabels(self._columns)
        # Resize cell size to content
        self.resizeColumnsToContents()
        self.resizeRowsToContents()

        # Only entire rows can be selected
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        # Rows can be selected with Ctrl and Shift like one would expect in Windows
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        # Set vertical scrollbars as always visible, but hide horizontal ones
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        # Disable drag-and-drop
        self.setDragDropMode(QtWidgets.QAbstractItemView.NoDragDrop)

        # Resize table
        self._adjust_table_size()

        # Get default properties
        self._EditTriggers = self.editTriggers()
        self._FocusPolicy = self.focusPolicy()
        self._SelectionMode = self.selectionMode()
        # Start with interaction enabled
        self.enable_interaction()

    def _adjust_table_size(self):
        """Adapt table size to current content"""
        self._getQTableWidgetSize()
        val = np.min((self.minimum_height, self.table_size.height()))
        tableSize = QtCore.QSize(self.table_size.width(), val)
        self.setMinimumSize(tableSize)
        self.setMaximumSize(self.table_size)

    def _getQTableWidgetSize(self):
        """Calculate the right table size for the current content"""
        w = self.verticalHeader().width() + 4 +20 # +4 seems to be needed
        for i in range(self.columnCount()):
            w += self.columnWidth(i) # seems to include gridline
        h = self.horizontalHeader().height() + 4
        for i in range(self.rowCount()):
            h += self.rowHeight(i)
        self.table_size = QtCore.QSize(w, h)

    def disable_interaction(self):
        """Make table not interactive."""
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.setFocusPolicy(QtCore.Qt.NoFocus)
        self.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.setEnabled(False)
        self.allowed_interaction = False

    def enable_interaction(self):
        """Make table interactive."""
        self.setEditTriggers(self._EditTriggers)
        self.setFocusPolicy(self._FocusPolicy)
        self.setSelectionMode(self._SelectionMode)
        self.setEnabled(True)
        self.allowed_interaction = True

    @staticmethod
    def get_row_number(items):
        """Return the row index of each item in <items>."""
        return np.array([i.row() for i in items])


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
    if not hasattr(field_name, "__iter__"):
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
    # # Get a missing number in the range from 1 to the maximum existing id
    # possible_ids =  [i for i in range(1, last_id) if i not in already_existing]
    # # If the list is empty, there is no number missing. Therefore, return the
    # # maximum value + 1
    # if len(possible_ids) == 0:
    #     return int(last_id + 1)
    # # otherwise, return the first missing number
    # else:
    #     return int(possible_ids[0])


################################################################################
# Arrays
################################################################################
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


################################################################################
# Image transformations
################################################################################
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
