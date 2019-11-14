import ctypes
import os

from PyQt5 import QtWidgets, QtCore, QtGui
import numpy as np

from calipy.general_configs import default as GUI_default
from third_party import pyqtgraph as pg


################################################################################
# Qt
################################################################################
class Qt5_QtApp(object):
    def __init__(self, appID=None, icon=None):
        """Make Qt app and set its icon."""
        self.app = QtWidgets.QApplication.instance()
        # If not already running, instantiate a new app
        if self.app is None:  # Quit existing instance
            self.app = QtWidgets.QApplication([])
        # Set app ID
        if appID is not None and os.name == 'nt':
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(appID)
        # Set app icon
        if icon is not None:
            app_icon = QtGui.QIcon(icon)
            self.app.setWindowIcon(app_icon)
        # Set 'fusion' style
        self.app.setStyle(QtWidgets.QStyleFactory.create('Fusion'))


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


def MessageBox(message, title='', parent=None, box_type='question', button_Yes='Yes', button_No='No', default_answer='Yes', additional_buttons=None):
    # Create empty MessageBox
    msgBox = QtWidgets.QMessageBox(parent)

    # Make buttons
    QButton_Yes = msgBox.addButton(button_Yes, QtWidgets.QMessageBox.YesRole)
    QButton_No = msgBox.addButton(button_No, QtWidgets.QMessageBox.NoRole)
    # Add other buttons
    additional_QButtons = list()
    if additional_buttons is not None:
        if isinstance(additional_buttons, str):
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


################################################################################
# pyqtgraph
################################################################################
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


class TimeAxis(pg.AxisItem):
    def generateDrawSpecs(self, p):
        """
        Calls tickValues() and tickStrings() to determine where and how ticks should
        be drawn, then generates from this a set of drawing commands to be
        interpreted by drawPicture().
        """
        # Initialize local variables


        bounds = self.mapRectFromParent(self.geometry())

        linkedView = self.linkedView()
        if linkedView is None or self.grid is False:
            tickBounds = bounds
        else:
            tickBounds = linkedView.mapRectToItem(self, linkedView.boundingRect())

        span = (bounds.topLeft(), bounds.topRight())
        tickStart = tickBounds.top()
        tickStop = bounds.top()
        tickDir = 1
        axis = 1

        ## Determine size of this item in pixels
        points = list(map(self.mapToDevice, span))
        if None in points:
            return
        lengthInPixels = pg.Point(points[1] - points[0]).length()
        if lengthInPixels == 0:
            return

        # Determine major / minor / subminor axis ticks
        if self._tickLevels is None:
            tickLevels = self.tickValues(self.range[0], self.range[1], lengthInPixels)
            tickStrings = None
        else:
            # Parse self.tickLevels into the formats returned by tickLevels() and tickStrings()
            tickLevels = []
            tickStrings = []
            for level in self._tickLevels:
                values = []
                strings = []
                tickLevels.append((None, values))
                tickStrings.append(strings)
                for val, strn in level:
                    values.append(val)
                    strings.append(strn)

        # Determine mapping between tick values and local coordinates
        dif = self.range[1] - self.range[0]
        if dif == 0:
            xScale = 1
            offset = 0
        else:
            if axis == 0:
                xScale = -bounds.height() / dif
                offset = self.range[0] * xScale - bounds.height()
            else:
                xScale = bounds.width() / dif
                offset = self.range[0] * xScale

        xRange = [x * xScale - offset for x in self.range]
        xMin = min(xRange)
        xMax = max(xRange)

        tickPositions = []  # remembers positions of previously drawn ticks

        # Compute coordinates to draw ticks
        # Draw three different intervals, long ticks first
        tickSpecs = []
        for i_level in range(len(tickLevels)):
            tickPositions.append([])
            ticks = tickLevels[i_level][1]

            ## length of tick
            tickLength = self.style['tickLength'] / ((i_level * 0.5) + 1.0)

            lineAlpha = 255 / (i_level + 1)
            if self.grid is not False:
                lineAlpha *= self.grid / 255. * np.clip((0.05 * lengthInPixels / (len(ticks) + 1)), 0., 1.)

            for v in ticks:
                # Determine actual position to draw this tick
                x = (v * xScale) - offset
                if x < xMin or x > xMax:  # Last check to make sure no out-of-bounds ticks are drawn
                    tickPositions[i_level].append(None)
                    continue
                tickPositions[i_level].append(x)

                p1 = [x, x]
                p2 = [x, x]
                p1[axis] = tickStart
                p2[axis] = tickStop
                if self.grid is False:
                    p2[axis] += tickLength * tickDir
                tickPen = self.pen()
                color = tickPen.color()
                color.setAlpha(lineAlpha)
                tickPen.setColor(color)
                tickSpecs.append((tickPen, pg.Point(p1), pg.Point(p2)))

        if self.style['stopAxisAtTick'][0] is True:
            stop = max(span[0].y(), min(map(min, tickPositions)))
            if axis == 0:
                span[0].setY(stop)
            else:
                span[0].setX(stop)
        if self.style['stopAxisAtTick'][1] is True:
            stop = min(span[1].y(), max(map(max, tickPositions)))
            if axis == 0:
                span[1].setY(stop)
            else:
                span[1].setX(stop)
        axisSpec = (self.pen(), span[0], span[1])

        textOffset = self.style['tickTextOffset'][axis]  # Spacing between axis and text
        textSize2 = 0
        textRects = []
        textSpecs = []

        # If values are hidden, return early
        if not self.style['showValues']:
            return axisSpec, tickSpecs, textSpecs

        # Get all strings
        # all_strings = list()
        # for i_level in range(min(len(tickLevels), self.style['maxTextLevel'] + 1)):
        #     ## Get the list of strings to display for this level
        #     if tickStrings is None:
        #         spacing, values = tickLevels[i_level]
        #         strings = self.tickStrings(values, self.autoSIPrefixScale * self.scale, spacing)
        #     else:
        #         strings = tickStrings[i_level]
        #
        #     if len(strings) == 0:
        #         continue
        #
        #     # Ignore strings belonging to ticks that were previously ignored
        #     for j in range(len(strings)):
        #         if tickPositions[i_level][j] is None:
        #             strings[j] = None





        for i_level in range(min(len(tickLevels), self.style['maxTextLevel'] + 1)):
            ## Get the list of strings to display for this level
            if tickStrings is None:
                spacing, values = tickLevels[i_level]
                strings = self.tickStrings(values, self.autoSIPrefixScale * self.scale, spacing)
            else:
                strings = tickStrings[i_level]

            if len(strings) == 0:
                continue

            # Ignore strings belonging to ticks that were previously ignored
            for j in range(len(strings)):
                if tickPositions[i_level][j] is None:
                    strings[j] = None

            # Measure density of text; decide whether to draw this level
            rects = []
            for s in strings:
                if s is None:
                    rects.append(None)
                else:
                    br = p.boundingRect(QtCore.QRectF(0, 0, 100, 100),
                                        QtCore.Qt.AlignCenter, pg.python2_3.asUnicode(s))
                    ## boundingRect is usually just a bit too large
                    ## (but this probably depends on per-font metrics?)
                    br.setHeight(br.height() * 0.8)

                    rects.append(br)
                    textRects.append(rects[-1])

            if len(textRects) > 0:
                ## measure all text, make sure there's enough room
                if axis == 0:
                    textSize = np.sum([r.height() for r in textRects])
                    textSize2 = np.max([r.width() for r in textRects])
                else:
                    textSize = np.sum([r.width() for r in textRects])
                    textSize2 = np.max([r.height() for r in textRects])
            else:
                textSize = 0
                textSize2 = 0

            if i_level > 0:  ## always draw top level
                ## If the strings are too crowded, stop drawing text now.
                ## We use three different crowding limits based on the number
                ## of texts drawn so far.
                textFillRatio = float(textSize) / lengthInPixels
                finished = False
                for nTexts, limit in self.style['textFillLimits']:
                    if len(textSpecs) >= nTexts and textFillRatio >= limit:
                        finished = True
                        break
                if finished:
                    break

            # spacing, values = tickLevels[best]
            # strings = self.tickStrings(values, self.scale, spacing)
            # Determine exactly where tick text should be drawn
            for j in range(len(strings)):
                vstr = strings[j]
                if vstr is None:  ## this tick was ignored because it is out of bounds
                    continue
                vstr = pg.python2_3.asUnicode(vstr)
                x = tickPositions[i_level][j]
                # textRect = p.boundingRect(QtCore.QRectF(0, 0, 100, 100), QtCore.Qt.AlignCenter, vstr)
                textRect = rects[j]
                height = textRect.height()
                width = textRect.width()
                # self.textHeight = height
                offset = max(0, self.style['tickLength']) + textOffset
                if self.orientation == 'left':
                    textFlags = QtCore.Qt.TextDontClip | QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter
                    rect = QtCore.QRectF(tickStop - offset - width,
                                         x - (height / 2), width, height)
                elif self.orientation == 'right':
                    textFlags = QtCore.Qt.TextDontClip | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter
                    rect = QtCore.QRectF(tickStop + offset, x - (height / 2),
                                         width, height)
                elif self.orientation == 'top':
                    textFlags = QtCore.Qt.TextDontClip | QtCore.Qt.AlignCenter | QtCore.Qt.AlignBottom
                    rect = QtCore.QRectF(x - width / 2.,
                                         tickStop - offset - height, width,
                                         height)
                elif self.orientation == 'bottom':
                    textFlags = QtCore.Qt.TextDontClip | QtCore.Qt.AlignCenter | QtCore.Qt.AlignTop
                    rect = QtCore.QRectF(x - width / 2., tickStop + offset,
                                         width, height)

                # p.setPen(self.pen())
                # p.drawText(rect, textFlags, vstr)
                textSpecs.append((rect, textFlags, vstr))

        ## update max text size if needed.
        self._updateMaxTextSize(textSize2)

        return (axisSpec, tickSpecs, textSpecs)



    # def tickStrings(self, values, scale, spacing):
    #     time_values = np.array(values) * scale / GUI_default['frame_rate']
    #     time_values_str = ['']
    #     # Set range for precision
    #     precision = 0
    #     max_precision = 3
    #     while precision <= max_precision:
    #         # Make format for numbers
    #         str_format = '%%.%if' % precision
    #         # Convert numbers to str
    #         time_values_str = [str_format % i for i in time_values.astype(float)]
    #
    #         # Use label if only one
    #         if len(time_values_str) == 1:
    #             break
    #         else:
    #             # Increase precision if some labels have the same numbers
    #             if np.unique(time_values_str).shape[0] != len(time_values_str):
    #                 precision += 1
    #             else:
    #                 break
    #
    #     return time_values_str


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


################################################################################
# Create elements
################################################################################
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

