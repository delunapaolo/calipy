# Add pyqtgraph folder to system path
import os, sys
script_path = os.path.realpath(__file__)
pyqtgraph_path = os.path.join(os.path.dirname(script_path), 'GUI_ROI_segmentation', '3rd_party', 'pyqtgraph')
sys.path.insert(0, pyqtgraph_path)

# Continue with imports --------------------------------------------------------
# System packages
from shutil import copyfile
import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore', FutureWarning)
    import h5py
from functools import partial

# Graphical packages
import cv2
from matplotlib import cm as mpl_colormaps
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
# Set global settings
pg.setConfigOption('imageAxisOrder', 'col-major')
pg.setConfigOption('antialias', False)

# Numerical packages
import numpy as np
import pandas as pd
from scipy.io import savemat
from scipy import ndimage

# Local repository
from Utilities import matlab_file
from GUI_ROI_segmentation.general_configs import default as GUI_default
from GUI_ROI_segmentation.IO_operations import prepare_data_for_GUI, log
from GUI_ROI_segmentation.Utilities import Qt5_QtApp, initialize_field, make_toolbar_button, make_toolbar_action, MultiColumn_QTable, Qt_window, default_colors, make_new_ROI_id, make_QSplitter, idx2range, cmapToColormap, expand_indices, CustomViewBox, transform_image_from_parameters, transform_image_from_matrices, ProgressDialog, MessageBox, Slider, ImageView
from GUI_ROI_segmentation.ROIs import ROI, PolyLineROI


class Calcium_imaging_data_explorer(object):
    def __init__(self, PARAMETERS, app):
        log('Loading GUI')

        # Make numpy ignore RuntimeWarnings
        warnings.simplefilter('ignore', category=RuntimeWarning)

        # Store pointers to parameters
        self.PARAMETERS = PARAMETERS
        self.app = app
        # Check whether signals are evoked by stimuli
        self.is_stimulus_evoked = bool(self.PARAMETERS['stimulus_evoked'])
        self.n_conditions = len(self.PARAMETERS['condition_names'])
        # Open files for reading
        self.data_frame = np.memmap(self.PARAMETERS['filename_data_frame'], dtype=np.float32, mode='r', offset=0).reshape((self.PARAMETERS['n_frames'], self.PARAMETERS['frame_height'], self.PARAMETERS['frame_width']))
        self.data_time = np.memmap(self.PARAMETERS['filename_data_time'], dtype=np.float32, mode='r', offset=0).reshape((self.PARAMETERS['n_pixels'], self.PARAMETERS['n_frames']))

        # Load data
        self.initialize_GUI()

    def initialize_GUI(self):
        self.stimulus_profile = None
        # Get average frames
        self.data_average = dict()
        with h5py.File(self.PARAMETERS['filename_projections'], 'r', libver='latest') as file_in:
            projections = file_in['PROJECTIONS']
            average_types = projections.keys()
            for avg_typ in average_types:
                self.data_average[avg_typ] = projections[avg_typ][:]
        # Convert index of frames from string to numerical
        self.frames_idx = self.PARAMETERS['frames_idx'].astype(np.int64)
        # Create array of indices
        self.frames_idx_array = np.zeros((np.max(self.frames_idx), ), dtype=np.int64)
        for itrial in range(self.frames_idx.shape[0]):
            self.frames_idx_array[self.frames_idx[itrial, 0]-1:self.frames_idx[itrial, 1]] = itrial
        # Get info on sessions
        sessions_last_frame = np.atleast_2d(self.PARAMETERS['sessions_last_frame']).transpose()
        sessions_last_frame = np.hstack((np.vstack(([0], sessions_last_frame[:-1]+1)), sessions_last_frame))
        self.sessions_idx = np.zeros((self.n_conditions, ), dtype=int)
        for idx in range(sessions_last_frame.shape[0]):
            self.sessions_idx[np.unique(self.frames_idx_array[sessions_last_frame[idx, 0]:sessions_last_frame[idx, 1]])] = idx
        self.n_sessions = np.unique(self.sessions_idx).shape[0]

        # Initialize values and flags used by GUI
        self.menu = None
        initialize_field(self, 'menu')
        self.image_views_title = None
        initialize_field(self, 'image_views_title', 'list', shape=2)
        self.region_selection = list()
        self.buttons = None
        initialize_field(self, 'buttons')
        self.keyboard = None
        initialize_field(self, 'keyboard')
        self.selected_ROI_type = GUI_default['ROI_type']
        self.current_action = ''
        self.selected_ROI = None
        self.last_ROI_coordinates = list()
        self.last_ROI_coordinates_handle = None
        self.last_ROI_coordinates_brushes = list()
        self.translation_ROI = None
        self.translation_image = None
        self.translation_image_levels = None
        self.crosshair = list()
        self.FOV_borders = list()
        initialize_field(self, 'FOV_borders', 'list', shape=2)
        self.next_ROI_color = None
        self.colormap = None
        initialize_field(self, 'colormap', 'list', shape=2)
        self.current_colormap = GUI_default['colormap_index']
        self.histograms_linked = False
        self.updating_histograms = False
        self.auto_adjust_histograms = True
        self.updating_timeline_position = False
        self.toggling_ROI_visibility = False
        self.views_linked = False
        self.trace_anchored = False
        self.projection_types = sorted(GUI_default['all_projection_types'])
        self.current_projection_type = GUI_default['projection_type']
        self.average_types = sorted(self.data_average.keys())
        self.operation_types = sorted(['mean', 'max'])
        if GUI_default['average_type'] in self.average_types:
            self.current_average_type = GUI_default['average_type']
        else:
            self.current_average_type = self.average_types[0]
        self.current_operation_type = 'mean'
        self.current_average_frame = [0]
        self.flood_tolerance = GUI_default['flood_tolerance']

        # Initialize the generator of colors
        self.COLORS = default_colors(normalize_01=False)
        # Initialize table to keep all info on ROIs
        self.ROI_TABLE = pd.DataFrame(columns=['id','handle_ROI','handle_ROI_contour','handle_trace','type','reference_pixel','flood_tolerance','region','color','show_trace'])
        # Initialize table to contain all info on how to translate single frames
        self.TRANSFORMATION_MATRICES = pd.DataFrame(columns=['scale','rotation','translation'])
        # Initialize lookup table to find where info of each condition is stored in the table
        self.TRANSFORMATION_IDX = np.array([])
        initialize_field(self, 'TRANSFORMATION_IDX', 'array', initial_value='list', shape=(self.n_conditions, 2))
        self.TRANSFORMATION_IDX[:, 0] = np.arange(self.n_conditions, dtype=int)
        # Initialize lookup table of pixels in which each value is stored
        pixel_list = np.arange(self.PARAMETERS['n_pixels'], dtype=int)
        pixel_list = pixel_list.reshape(self.PARAMETERS['frame_height'], self.PARAMETERS['frame_width']).transpose().ravel()
        self.MAPPING_TABLE = np.tile(np.atleast_2d(pixel_list), (self.n_conditions, 1)).astype(np.float32)  # Float32 conversion to hold NaNs
        self.modified_MAPPING = np.zeros(shape=self.n_conditions, dtype=bool)
        self.temporary_MAPPING_TABLE = np.array([])

        # Create window
        self.window = Qt_window()
        # Set title
        self.window.setWindowTitle('ROI Segmentation - %s' % self.PARAMETERS['animal_ID'])
        self.window.resized.connect(self.callback_fix_colormap)
        self.window.about_to_close.connect(self.callback_close_window)

        # Make menubar
        self.menu.menubar = self.window.menuBar()
        self.menu.file = self.menu.menubar.addMenu('File')
        self.menu.view = self.menu.menubar.addMenu('View')
        self.menu.average = self.menu.menubar.addMenu('Average')
        self.menu.projections = self.menu.menubar.addMenu('Projection')
        self.menu.ROI_operation = self.menu.menubar.addMenu('ROI operation')
        # Make menu actions for the File menu
        self.menu.save = self.menu.file.addAction('Save ROIs to disk')
        self.menu.file.addSeparator()
        self.menu.load = self.menu.file.addAction('Load ROIs from disk')
        self.menu.file.addSeparator()
        self.menu.delete_ROIs = self.menu.file.addAction('Delete ROIs in memory (no change to file on disk)')
        self.menu.file.addSeparator()
        self.menu.reload = self.menu.file.addAction('Reset transformations to field-of-view')
        self.menu.file.addSeparator()
        self.menu.reload_GUI = self.menu.file.addAction('Restart GUI')
        # Make menu actions for the View menu
        self.menu.auto_adjust_histograms = self.menu.view.addAction('Automatically adjust histograms')
        self.menu.auto_adjust_histograms.setCheckable(True)
        self.menu.auto_adjust_histograms.setChecked(self.auto_adjust_histograms)

        # Make menu actions to choose average type
        self.menu.average_actions = [self.menu.average.addAction(i) for i in self.average_types]
        [i.setCheckable(True) for i in self.menu.average_actions]
        self.menu.average_type = QtWidgets.QActionGroup(self.window)
        [self.menu.average_type.addAction(i) for i in self.menu.average_actions]
        # Make menu actions to choose projection type
        self.menu.projection_actions = [self.menu.projections.addAction(i) for i in self.projection_types]
        [i.setCheckable(True) for i in self.menu.projection_actions]
        self.menu.projection_type = QtWidgets.QActionGroup(self.window)
        [self.menu.projection_type.addAction(i) for i in self.menu.projection_actions]
        # Make menu to choose what operation to perform on pixels in an ROI
        self.menu.ROI_operation_actions = [self.menu.ROI_operation.addAction(i) for i in self.operation_types]
        [i.setCheckable(True) for i in self.menu.ROI_operation_actions]
        self.menu.ROI_operation_type = QtWidgets.QActionGroup(self.window)
        [self.menu.ROI_operation_type.addAction(i) for i in self.menu.ROI_operation_actions]

        # Make toolbar buttons
        self.buttons.toggle_colormap = make_toolbar_button(self.window, GUI_default['icons']['toggle_colormap'], kind=GUI_default['icons_kind']['toggle_colormap'])
        self.buttons.link_views = make_toolbar_button(self.window, GUI_default['icons']['link_views'], kind=GUI_default['icons_kind']['link_views'], checkable=True)
        self.buttons.visibility_ROI = make_toolbar_button(self.window, GUI_default['icons']['visibility_ROI'], kind=GUI_default['icons_kind']['visibility_ROI'])
        self.buttons.zoom = make_toolbar_button(self.window, GUI_default['icons']['zoom'], kind=GUI_default['icons_kind']['zoom'])
        self.buttons.translation_ROI = make_toolbar_button(self.window, GUI_default['icons']['translation_ROI'], kind=GUI_default['icons_kind']['translation_ROI'], checkable=True)
        self.buttons.link_histograms = make_toolbar_button(self.window, GUI_default['icons']['link_histograms'], kind=GUI_default['icons_kind']['link_histograms'], checkable=True)
        # The button to make an ROI has a customized context menu to choose the type of ROI to make
        self.buttons.draw_ROI = make_toolbar_button(self.window, GUI_default['icons']['ROI_polygon'], kind=GUI_default['icons_kind']['ROI_polygon'])
        self.buttons.draw_ROI.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.buttons.draw_ROI.customContextMenuRequested.connect(self.callback_choose_ROI_type)
        self.buttons.draw_ROI.setStyleSheet('background-color: None')
        # Make other actions to pass to the contextual menu
        self.buttons.ROI_fill = make_toolbar_action(self.window, icon=GUI_default['icons']['ROI_fill'], text='fill')
        self.buttons.ROI_polygon = make_toolbar_action(self.window, icon=GUI_default['icons']['ROI_polygon'], text='polygon')
        # Make button to plot traces of selected ROIs
        self.buttons.anchor_trace = make_toolbar_button(self.window, GUI_default['icons']['anchor_trace'], kind=GUI_default['icons_kind']['anchor_trace'], checkable=True)

        # Add buttons to toolbar
        self.toolBar = QtWidgets.QToolBar('Tools')
        self.toolBar.setStyleSheet('QToolBar {background: white}')
        self.window.addToolBar(QtCore.Qt.LeftToolBarArea, self.toolBar)
        self.toolBar.setFloatable(False)
        self.toolBar.setMovable(False)
        self.toolBar.addWidget(self.buttons.link_views)
        self.toolBar.addWidget(self.buttons.anchor_trace)
        self.toolBar.addSeparator()
        self.toolBar.addWidget(self.buttons.toggle_colormap)
        self.toolBar.addWidget(self.buttons.visibility_ROI)
        self.toolBar.addWidget(self.buttons.zoom)
        self.toolBar.addWidget(self.buttons.link_histograms)
        self.toolBar.addSeparator()
        self.toolBar.addWidget(self.buttons.draw_ROI)
        self.toolBar.addWidget(self.buttons.translation_ROI)

        # Make list-box to contain list of conditions
        self.table_conditions = MultiColumn_QTable(self.PARAMETERS['condition_names'], title='conditions')
        self.table_conditions.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.table_sessions = MultiColumn_QTable(['session %i' % ii for ii in range(1, self.n_sessions+1)], title='sessions')
        self.table_sessions.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        table_size = QtCore.QSize(self.table_conditions.width(), self.table_sessions.height())
        self.table_sessions.setMinimumSize(table_size)
        self.table_sessions.setMaximumSize(table_size)
        self.table_sessions.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        # Place tables in layout
        table_layout = QtWidgets.QVBoxLayout()
        table_layout.addWidget(self.table_sessions)
        table_layout.addWidget(self.table_conditions)

        # Make layout on the top to see fields of view
        top_layout = QtWidgets.QHBoxLayout()
        top_left_layout = QtWidgets.QVBoxLayout()
        top_left_layout_bottom = QtWidgets.QHBoxLayout()
        # Make ImageItem for average image
        self.average_image_viewbox, self.average_image_view, self.average_image_scene = self.make_image_box(name='projection')
        self.average_image_view.setImage(self.data_average[self.current_average_type][self.current_average_frame[0], :, :].copy(), autoRange=True, axes={'x': 1, 'y': 2}, autoLevels=True)
        self.average_image_viewbox.setMenuEnabled(False)
        # Add label to identify condition shown
        self.image_views_title[0] = pg.TextItem(self.PARAMETERS['condition_names'][0], color=GUI_default['text_color_dark'], anchor=(0, 0))
        self.image_views_title[0].setFont(GUI_default['font_titles'])
        self.average_image_scene.addItem(self.image_views_title[0])
        top_left_layout_bottom.addWidget(self.average_image_scene)
        top_left_layout_bottom.setStretch(0, 10)
        self.average_histogram = pg.HistogramLUTWidget(image=self.average_image_view, rgbHistogram=False, levelMode='mono')
        self.average_histogram.item.gradient.allowAdd = False
        self.average_histogram.item.vb.setMenuEnabled(False)
        # Delete ticks from histogram
        [ii.setStyle(tickLength=0, showValues=False) for ii in self.average_histogram.items() if isinstance(ii, pg.AxisItem)]
        top_left_layout_bottom.addWidget(self.average_histogram)
        top_left_layout_bottom.setStretch(1, 1)
        # Add slider at the top
        self.slider_label = QtWidgets.QLabel('%.1f' % self.flood_tolerance)
        self.slider_label.setFont(GUI_default['font'])
        self.slider = Slider(minimum=0, maximum=1000)
        self.slider.setOrientation(QtCore.Qt.Horizontal)
        self.slider.setValue(self.flood_tolerance * 10)
        self.slider_update_GUI = True
        # Initialize as inactive
        self.toggle_slider_flood_tolerance(False)
        # Add elements to layout
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(self.slider_label)
        hbox.addWidget(self.slider)
        top_left_layout.addLayout(hbox)
        top_left_layout.addLayout(top_left_layout_bottom)
        top_layout.addLayout(top_left_layout)
        top_layout.setStretch(0, 1)
        # Draw crosshair and turn it off
        self.crosshair.append(pg.InfiniteLine(angle=90, movable=False))
        self.crosshair.append(pg.InfiniteLine(angle=0, movable=False))
        [self.average_image_viewbox.addItem(c, ignoreBounds=True) for c in self.crosshair]
        [c.setVisible(False) for c in self.crosshair]
        # Draw border of the field ov view
        pen = pg.mkPen(width=2)
        self.FOV_borders[0] = list()
        self.FOV_borders[0].append(pg.PlotCurveItem([0, 0], [0, self.PARAMETERS['frame_height']], clickable=False, pen=pen))  # bottom border
        self.FOV_borders[0].append(pg.PlotCurveItem([self.PARAMETERS['frame_width']] * 2, [0, self.PARAMETERS['frame_height']], clickable=False, pen=pen))  # top border
        self.FOV_borders[0].append(pg.PlotCurveItem([0, self.PARAMETERS['frame_width']], [0, 0], clickable=False, pen=pen))  # left border
        self.FOV_borders[0].append(pg.PlotCurveItem([0, self.PARAMETERS['frame_width']], [self.PARAMETERS['frame_height']] * 2, clickable=False, pen=pen))  # right border
        [self.average_image_viewbox.addItem(self.FOV_borders[0][ii]) for ii in range(4)]
        [self.FOV_borders[0][ii].setZValue(1000) for ii in range(4)]

        # Make view box for frame selection
        self.frame_image_view = ImageView(name='frame')
        self.frame_image_view.view.setMenuEnabled(False)
        self.frame_image_view.ui.roiBtn.hide()
        self.frame_image_view.ui.menuBtn.hide()
        self.frame_image_view.getHistogramWidget().item.vb.setMenuEnabled(False)
        # Delete ticks from histogram and timeline
        [ii.setStyle(tickLength=0, showValues=False) for ii in self.frame_image_view.getHistogramWidget().items() if isinstance(ii, pg.AxisItem)]
        self.frame_image_view.ui.roiPlot.hideAxis('bottom')
        # Set image
        self.frame_image_view.setImage(self.data_frame, autoRange=True, axes={'t': 0, 'x': 1, 'y': 2}, autoLevels=True)
        # Hide unused elements
        self.frame_image_view.ui.roiPlot.getPlotItem().hideButtons()  # Hide 'autoscale' button
        [ii.setVisible(False) for ii in self.frame_image_view.timeLine.getViewBox().allChildItems() if isinstance(ii, pg.VTickGroup)]  # Vertical ticks
        # Add boundaries of each session
        self.frame_image_view.timeLine.setZValue(1)
        for ii in range(self.PARAMETERS['sessions_last_frame'].shape[0] - 1):  # -1 because we ignore the very last idx
            l = pg.InfiniteLine(pos=self.PARAMETERS['sessions_last_frame'][ii], angle=90, pen=pg.mkPen(color=(77, 77, 77)))
            l.setZValue(1)
            self.frame_image_view.ui.roiPlot.addItem(l)
        # Add label to identify condition shown
        self.image_views_title[1] = pg.TextItem(self.PARAMETERS['condition_names'][0], color=GUI_default['text_color_dark'], anchor=(0, 0))
        self.image_views_title[1].setFont(GUI_default['font_titles'])
        self.frame_image_view.scene.addItem(self.image_views_title[1])
        # Draw border of the field ov view
        pen = pg.mkPen(width=2)
        self.FOV_borders[1] = list()
        self.FOV_borders[1].append(pg.PlotCurveItem([0, 0], [0, self.PARAMETERS['frame_height']], clickable=False, pen=pen))  # bottom border
        self.FOV_borders[1].append(pg.PlotCurveItem([self.PARAMETERS['frame_width']] * 2, [0, self.PARAMETERS['frame_height']], clickable=False, pen=pen))  # top border
        self.FOV_borders[1].append(pg.PlotCurveItem([0, self.PARAMETERS['frame_width']], [0, 0], clickable=False, pen=pen))  # left border
        self.FOV_borders[1].append(pg.PlotCurveItem([0, self.PARAMETERS['frame_width']], [self.PARAMETERS['frame_height']] * 2, clickable=False, pen=pen))  # right border
        [self.frame_image_view.view.addItem(self.FOV_borders[1][ii]) for ii in range(4)]
        [self.FOV_borders[1][ii].setZValue(1000) for ii in range(4)]
        # Add plot to layout
        top_layout.addWidget(self.frame_image_view)
        top_layout.setStretch(1, 1)

        # Make layout on the bottom to inspect ROIs and traces
        bottom_layout = make_QSplitter('hor')
        # Add on the bottom a graph to see latest ROI maximized
        self.ROI_viewbox, self.ROI_view, self.ROI_border = self.make_image_box(name='ROI_inspector')
        self.ROI_viewbox.setMenuEnabled(False)
        self.color_ROI_view_border(color=None)
        bottom_layout.addWidget(self.ROI_border)
        bottom_layout.setStretchFactor(0, 1)
        # Add on the bottom a graph to see Calcium traces
        self.trace_time_mark = pg.InfiniteLine(angle=90, bounds=[0, self.PARAMETERS['n_frames']], pen=pg.mkPen(color='r', width=3), movable=True)
        self.trace_time_mark.setZValue(3)
        v = pg.GraphicsView()
        self.trace_viewbox = pg.ViewBox()
        # self.trace_axis = pg.AxisItem('left', linkView=self.trace_viewbox)
        # self.trace_viewbox.addItem(self.trace_axis)

        self.trace_viewbox.setMenuEnabled(False)
        # Set limits of viewbox and enable auto-fit
        self.trace_viewbox.setLimits(xMin=0, xMax=self.PARAMETERS['n_frames'])
        self.trace_viewbox.setMouseEnabled(x=True, y=False)
        v.setCentralItem(self.trace_viewbox)
        self.trace_viewbox.addItem(self.trace_time_mark)

        # Add boundaries of each session
        for ii in range(self.PARAMETERS['sessions_last_frame'].shape[0] - 1):  # -1 because we ignore the very last idx
            l = pg.InfiniteLine(pos=self.PARAMETERS['sessions_last_frame'][ii], angle=90, pen=pg.mkPen(color=(77, 77, 77)))
            l.setZValue(1)
            self.trace_viewbox.addItem(l)
        self.trace_viewbox.setRange(xRange=[0, self.PARAMETERS['n_frames']], disableAutoRange=False)
        bottom_layout.addWidget(v)
        bottom_layout.setStretchFactor(1, 3)

        # Make final layout
        graph_layout = QtWidgets.QVBoxLayout()
        graph_layout.addLayout(top_layout)
        graph_layout.addWidget(bottom_layout)
        graph_layout.setStretch(0, 4)
        graph_layout.setStretch(1, 1)
        layout = QtWidgets.QHBoxLayout()
        layout.addLayout(table_layout)
        layout.addLayout(graph_layout)
        # Put layout in the center
        qFrame = QtWidgets.QWidget()
        self.window.setCentralWidget(qFrame)
        qFrame.setLayout(layout)
        qFrame.setFocus()

        # Connect callbacks to buttons
        self.buttons.draw_ROI.clicked.connect(self.callback_draft_ROI)
        self.buttons.toggle_colormap.clicked.connect(partial(self.callback_toggle_colormap, force=None))
        self.buttons.zoom.clicked.connect(partial(self.callback_reset_zoom, where='all'))
        self.buttons.translation_ROI.toggled.connect(self.callback_activate_translation_FOV)
        self.buttons.link_views.toggled.connect(self.callback_link_views)
        self.buttons.link_histograms.toggled.connect(self.callback_link_histograms)
        self.buttons.anchor_trace.toggled.connect(self.callback_anchor_trace)
        self.buttons.visibility_ROI.clicked.connect(partial(self.callback_update_visibility_traces, from_button=True))

        # Connect callbacks for mouse interaction
        self.average_image_viewbox.scene().sigMouseMoved.connect(self.callback_mouse_moved)
        self.average_image_viewbox.scene().sigMouseClicked.connect(self.callback_mouse_clicked)
        self.table_conditions.itemSelectionChanged.connect(partial(self.callback_table_selection, what='condition'))
        self.table_sessions.itemSelectionChanged.connect(partial(self.callback_table_selection, what='session'))
        self.slider.valueChanged.connect(self.callback_flood_tolerance_value_changed)
        # Connect callbacks to timeline slider
        self.frame_image_view.timeLine.sigPositionChanged.disconnect()
        self.frame_image_view.timeLine.sigPositionChanged.connect(partial(self.callback_timeline_frames, from_plot='frame'))
        self.trace_time_mark.sigPositionChanged.connect(partial(self.callback_timeline_frames, from_plot='trace'))
        # Connect callbacks to menu actions
        self.menu.save.triggered.connect(partial(self.callback_save, compute_traces=False))
        self.menu.save.setShortcut(QtGui.QKeySequence('Ctrl+S'))
        self.menu.load.triggered.connect(self.callback_load)
        self.menu.load.setShortcut(QtGui.QKeySequence('Ctrl+L'))
        self.menu.reload.triggered.connect(partial(self.callback_reload, what='frames'))
        self.menu.reload.setShortcut(QtGui.QKeySequence('Ctrl+R'))
        self.menu.reload_GUI.triggered.connect(self.callback_reload_GUI)
        self.menu.delete_ROIs.triggered.connect(partial(self.callback_reload, what='ROIs'))
        self.menu.auto_adjust_histograms.triggered.connect(self.callback_auto_adjust_histograms)
        [ii.triggered.connect(partial(self.callback_menu_projection_type, name=jj)) for ii, jj in zip(self.menu.projection_actions,self.projection_types)]
        [ii.triggered.connect(partial(self.callback_menu_average_type, name=jj)) for ii, jj in zip(self.menu.average_actions, self.average_types)]
        [ii.triggered.connect(partial(self.callback_menu_operation_type, name=jj)) for ii, jj in zip(self.menu.ROI_operation_actions, self.operation_types)]

        # Make keyboard shortcuts
        self.keyboard.Key_Enter = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Enter), self.window)
        self.keyboard.Key_Enter.setAutoRepeat(False)
        self.keyboard.Key_Return = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Return), self.window)
        self.keyboard.Key_Return.setAutoRepeat(False)
        self.keyboard.Key_Escape = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Escape), self.window)
        self.keyboard.Key_Escape.setAutoRepeat(False)
        self.keyboard.Key_Up = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Up), self.window)
        self.keyboard.Key_Up.setAutoRepeat(True)
        self.keyboard.Key_Down = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Down), self.window)
        self.keyboard.Key_Down.setAutoRepeat(True)
        self.keyboard.Key_Left = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Left), self.window)
        self.keyboard.Key_Left.setAutoRepeat(True)
        self.keyboard.Key_Right = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Right), self.window)
        self.keyboard.Key_Right.setAutoRepeat(True)
        self.keyboard.Key_Z = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Z), self.window)
        self.keyboard.Key_Z.setAutoRepeat(True)
        self.keyboard.Key_Space = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Space), self.window)
        self.keyboard.Key_Space.setAutoRepeat(False)
        # Connect callbacks for keyboard interaction
        self.keyboard.Key_Enter.activated.connect(self.callback_continue_last_action)
        self.keyboard.Key_Return.activated.connect(self.callback_continue_last_action)
        self.keyboard.Key_Escape.activated.connect(self.callback_cancel_last_action)
        self.keyboard.Key_Up.activated.connect(partial(self.callback_scroll_average_frame, where='up'))
        self.keyboard.Key_Down.activated.connect(partial(self.callback_scroll_average_frame, where='down'))
        self.keyboard.Key_Left.activated.connect(partial(self.callback_transform_current_frame, direction=-1))
        self.keyboard.Key_Right.activated.connect(partial(self.callback_transform_current_frame, direction=+1))
        self.keyboard.Key_Z.activated.connect(partial(self.callback_reset_zoom, where='all'))
        self.keyboard.Key_Space.activated.connect(self.callback_draft_ROI)

        # Make colormaps
        pos, rgba_colors = zip(*cmapToColormap(mpl_colormaps.gray))
        self.colormap[0] = pg.ColorMap(pos, rgba_colors)
        pos, rgba_colors = zip(*cmapToColormap(mpl_colormaps.gray_r))
        self.colormap[1] = pg.ColorMap(pos, rgba_colors)

        # Initialize button state
        self.buttons.link_views.setChecked(self.views_linked)
        self.buttons.anchor_trace.setChecked(self.trace_anchored)
        self.buttons.link_histograms.setChecked(False)
        self.buttons.translation_ROI.setChecked(False)
        [i.setChecked(True) for i in self.menu.average_actions if i.text()==self.current_average_type]
        [i.setChecked(True) for i in self.menu.projection_actions if i.text() == self.current_projection_type]
        [i.setChecked(True) for i in self.menu.ROI_operation_actions if i.text() == self.current_operation_type]

        # If there are ROIs present, enable the tool that allows to translate the field of view
        button_state = self.ROI_TABLE.shape[0] > 0
        self.buttons.translation_ROI.setEnabled(button_state)

        # Activate GUI
        self.window.showMaximized()
        self.window.show()
        # Automatically load last save
        self.callback_load()
        # Update table
        self.update_table_condition_names()
        # Reset range everywhere
        self.callback_reset_zoom(where='all')
        self.callback_link_histograms(False)
        # Bring to front
        self.window.raise_()
        log('GUI is ready')

    def callback_reload_GUI(self):
        answer = MessageBox('Restart GUI?\nAll progress will be lost!', title='GUI ROI segmentation', parent=self.window)
        if answer.lower() == 'no':
            return
        else:
            # Close window
            self.window.about_to_close.disconnect()
            self.window.allowed_to_close = True
            self.window.close()

            # Set fields to keep (those from __init__())
            fields_to_keep = ['PARAMETERS', 'is_stimulus_evoked', 'n_conditions', 'data_frame', 'data_time', 'app']
            # Delete everything apart from these fields
            all_attributes = list(self.__dict__.keys())
            attributes_to_delete = [i for i in all_attributes if i not in fields_to_keep]
            for name in attributes_to_delete:
                self.__dict__.pop(name, None)
            # Re-initialize GUI
            self.initialize_GUI()


    ############################################################################
    # File-related
    ############################################################################
    def callback_close_window(self):
        answer = MessageBox('Close GUI?', title='GUI ROI segmentation', parent=self.window)
        if answer.lower() == 'no':
            # Make sure window won't be closed
            self.window.allowed_to_close = False
        else:
            answer = MessageBox('Save current ROIs to disk?', title='GUI ROI segmentation', parent=self.window, button_No='Discard')
            if answer.lower() == 'yes':
                self.callback_save(compute_traces=True)
            # Allow window to be closed
            self.window.allowed_to_close = True
            # Allow app to be closed when the window will close
            self.app.app.setQuitOnLastWindowClosed(True)
            log('Quitting GUI')

    def callback_save(self, compute_traces=False):
        if not self.menu.menubar.isEnabled():
            return
        # Count ROIs
        n_ROIs = self.ROI_TABLE.shape[0]
        if n_ROIs < 1:
            return
        # Check whether a previous file exists
        file_present = os.path.exists(self.PARAMETERS['filename_output'])
        # Ask user whether to overwrite the previous file
        if file_present:
            answer = MessageBox('Overwrite previous file?', title='GUI ROI segmentation', parent=self.window, button_Yes='Overwrite', button_No='Cancel', additional_buttons='Backup and make new', default_answer='Backup and make new')
            if answer == 'Cancel':
                return
            elif answer == 'Backup and make new':
                # Get name for backup
                backup_filename, backup_ext = os.path.splitext(self.PARAMETERS['filename_output'])
                backup_filename += '_backup'
                backup_filename += backup_ext
                # Do backup copy of file
                log('Backing-up file to \'%s\'' % backup_filename)
                copyfile(self.PARAMETERS['filename_output'], backup_filename)

        # Log action
        log('Storing segmentation to disk')

        # Disable window
        self.window.setEnabled(False)
        # Allocate variables used in following for-loop
        ROI_masks = np.zeros((self.PARAMETERS['frame_height'], self.PARAMETERS['frame_width'], n_ROIs), dtype=np.int8)
        msg = 'Computing ROI info'
        if compute_traces:
            F = np.zeros((self.PARAMETERS['n_frames'], n_ROIs), dtype=np.float32)
            msg += ' and fluorescence traces'
        # Create a progress dialog without Cancel button
        dlg = ProgressDialog(msg, minimum=0, maximum=n_ROIs, parent=self.window)

        # Get frames
        frames = self.data_average['mean'].copy()
        # Apply transformations
        n_frames = len(self.current_average_frame)
        for iframe in xrange(n_frames):
            T = self.TRANSFORMATION_IDX[self.TRANSFORMATION_IDX[:, 0] == self.current_average_frame[iframe], 1][0]
            if len(T) > 0:
                # Apply transformations to current frame one after the other
                for t_idx in T:
                    # Get matrices
                    scale = self.TRANSFORMATION_MATRICES.loc[t_idx, 'scale']
                    rotation = self.TRANSFORMATION_MATRICES.loc[t_idx, 'rotation']
                    translation = self.TRANSFORMATION_MATRICES.loc[t_idx, 'translation']
                    # Apply them
                    frames[iframe, :, :] = transform_image_from_matrices(frames[iframe, :, :], scale, rotation, translation)
        # Get average frame as template
        image_to_analyze = np.nanmean(frames, axis=0)
        # Loop through ROIs
        rows = self.ROI_TABLE.index.values
        for idx, row in enumerate(rows):
            roi = self.ROI_TABLE.loc[row, 'handle_ROI']
            area, coords = roi.getArrayRegion(image_to_analyze, self.average_image_view, axes=(0, 1), returnMappedCoords=True)
            # Get indices of ROI pixels
            area_idx = np.where(area > 0)
            ROI_idx = np.vstack((coords[0, area_idx[0], area_idx[1]], coords[1, area_idx[0], area_idx[1]]))
            ROI_pixels = np.unique(np.ravel_multi_index(np.round(ROI_idx, decimals=0).astype(int), [self.PARAMETERS['frame_height'], self.PARAMETERS['frame_width']], mode='raise', order='C'))
            # Store mask
            mask_indices = np.unravel_index(ROI_pixels, (self.PARAMETERS['frame_height'], self.PARAMETERS['frame_width']))
            ROI_masks[mask_indices[0], mask_indices[1], idx] = 1

            if compute_traces:
                # Compute mean fluorescence in the ROI
                L = np.mean(self.data_time[ROI_pixels, :], axis=0)
                # Check whether some segments have been translated
                if np.any(self.modified_MAPPING):
                    # Get index of frames that are modified
                    modified_conditions_idx = np.where(self.modified_MAPPING == True)[0]
                    # Get segments that have been translated
                    if modified_conditions_idx.size > 0:
                        L_modified, modified_frames_idx = self.get_translated_trace_segments(modified_conditions_idx, ROI_pixels, from_temporary=False)
                        # Replace values in original array
                        L[modified_frames_idx] = L_modified
                # Store in main variable
                F[:, idx] = L

            # Advance progress dialog
            dlg += 1
        # Close progress dialog
        dlg.close()

        # Keep all information on ROIs (don't keep pointers to handles)
        ROI_info = self.ROI_TABLE[[c for c in self.ROI_TABLE.columns if 'handle' not in c]].values
        # Get index of column corresponding to ROI ID
        col_idx = list(self.ROI_TABLE.columns).index('id')
        ROI_info[:, col_idx] = np.arange(0, n_ROIs, dtype=int)

        # Make dictionary for Matlab
        mdict = dict({'ROIs'                   : ROI_masks,
                      'ROI_info'               : ROI_info,
                      'TRANSFORMATION_MATRICES': self.TRANSFORMATION_MATRICES.values,
                      'TRANSFORMATION_IDX'     : self.TRANSFORMATION_IDX,
                      'MAPPING_TABLE'          : self.MAPPING_TABLE,
                      'modified_MAPPING'       : self.modified_MAPPING})
        # Add traces, if requested
        if compute_traces:
            mdict['ROI_fluorescence'] = F
        # Store info on disk
        dlg = ProgressDialog('Saving ROI info to disk', minimum=0, maximum=0, parent=self.window)
        savemat(self.PARAMETERS['filename_output'], mdict, oned_as='column')
        dlg.close()
        # Enable window
        self.window.setEnabled(True)
        # Log end of action
        log('... done')

    def callback_load(self):
        if not self.menu.menubar.isEnabled():
            return
        # Check that file exists
        if not os.path.exists(self.PARAMETERS['filename_output']):
            return
        # Count ROIs
        n_ROIs = self.ROI_TABLE.shape[0]
        if n_ROIs != 0:
            # Ask user whether to overwrite info in memory with info from the file
            answer = MessageBox('CAREFUL: Current ROI segmentation will be lost and replaced with that from disk.\nAre you sure to continue?', box_type='warning', title='GUI ROI segmentation', parent=self.window)
            if answer.lower() == 'no':
                return

        # Log action
        log('Loading previous segmentation')

        # Check whether window is open (e.g., during __init__())
        has_window = hasattr(self, 'window') and self.window.isVisible()

        if has_window:
            # Disable window
            self.window.setEnabled(False)
            # Create progress dialog
            dlg = ProgressDialog('Loading ROIs from disk', minimum=0, maximum=1, parent=self.window)
            # Delete all previous ROIs
            all_ROIs = self.ROI_TABLE['handle_ROI'].values
            for roi in all_ROIs:
                if not isinstance(roi, np.ndarray):
                    self.callback_remove_ROI(roi, log_outcome=False)
        else:
            # Initialize variable as integer
            dlg = 0

        # Reset tables
        self.ROI_TABLE.drop(self.ROI_TABLE.index, inplace=True)
        self.TRANSFORMATION_MATRICES.drop(self.TRANSFORMATION_MATRICES.index, inplace=True)

        # Load file
        file_content = matlab_file.load(self.PARAMETERS['filename_output'])
        # Unpack variables
        ROI_info = np.atleast_2d(file_content['ROI_info'])
        # Last column contains boolean values
        ROI_info[:, -1] = ROI_info[:, -1].astype(bool)
        n_ROIs = ROI_info.shape[0]
        if has_window:
            dlg.setMaximum(n_ROIs + 1)  # +1 is the step when loading the file
        # Allocate empty values
        for row in range(n_ROIs):
            self.ROI_TABLE.loc[row, :] = None
        # Fill values
        self.ROI_TABLE.loc[:, [c for c in self.ROI_TABLE.columns if 'handle' not in c]] = ROI_info
        # Make sure colors are tuples of integers
        for row in range(n_ROIs):
            self.ROI_TABLE.loc[row, 'color'] = tuple(self.ROI_TABLE.loc[row, 'color'].astype(int))
        # Copy transformation matrices
        self.TRANSFORMATION_IDX = file_content['TRANSFORMATION_IDX'].copy()
        for idx, v in enumerate(self.TRANSFORMATION_IDX[:, 1]):
            if isinstance(v, np.ndarray):
                v = list(v)
            else:
                v = list([v])
            self.TRANSFORMATION_IDX[idx, 1] = v
        # Reset index to match table index
        all_transformations, all_transformations_idx = np.unique(self.TRANSFORMATION_IDX[:, 1], return_inverse=True)
        all_transformations = np.hstack(all_transformations).astype(int)
        all_transformations = np.column_stack((all_transformations, np.arange(len(all_transformations))))
        for idx, v in enumerate(self.TRANSFORMATION_IDX[:, 1]):
            if len(v) > 0:
                self.TRANSFORMATION_IDX[idx, 1] = all_transformations[all_transformations[:,0] == v, 1].tolist()
        if file_content['TRANSFORMATION_MATRICES'].size > 0:
            self.TRANSFORMATION_MATRICES = pd.DataFrame(np.atleast_2d(file_content['TRANSFORMATION_MATRICES']), columns=self.TRANSFORMATION_MATRICES.columns)

        # Copy pixel mapping information
        self.MAPPING_TABLE = file_content['MAPPING_TABLE']
        self.modified_MAPPING = file_content['modified_MAPPING'].astype(bool)
        # Increase progress counter
        dlg += 1

        # Re-draw ROIs
        for ii in range(n_ROIs):
            # Get color of this ROI
            pen = pg.mkPen(color=self.ROI_TABLE.loc[ii, 'color'], width=GUI_default['ROI_linewidth_thick'])
            # Get ROI type and its contour / region
            points = self.ROI_TABLE.loc[ii, 'region']
            # Draw ROI
            roi = PolyLineROI(points, pen=pen, closed=True, movable=True, removable=True)

            # Connect callbacks
            self.ROI_add_context_menu(roi)
            roi.sigRegionChanged.connect(partial(self.callback_update_zoom_ROI, update_trace=True, update_ROI_contour=True))
            roi.sigRemoveRequested.connect(partial(self.callback_remove_ROI, log_outcome=False))
            roi.sigClicked.connect(self.callback_click_ROI)
            roi.setAcceptedMouseButtons(QtCore.Qt.LeftButton)

            # Store id of this ROI it in the handle so we can always come back to the table without checking object identity
            roi.ROI_id = self.ROI_TABLE.loc[ii, 'id']
            # Add ROI to plot
            self.average_image_viewbox.addItem(roi)

            # Transform ROI to a static polygon
            points = np.array(roi.getState()['points'])
            points = np.vstack((points, points[0, :]))
            # Make PlotItem
            pi_right = pg.PlotCurveItem(clickable=False)
            pi_right.setData(x=points[:, 0], y=points[:, 1])
            self.frame_image_view.view.addItem(pi_right)
            pi_right.setPen(pen)

            # Store handles in memory
            self.ROI_TABLE.loc[ii, 'handle_ROI'] = roi
            self.ROI_TABLE.loc[ii, 'handle_ROI_contour'] = pi_right

            # Make ROI inactive unless it is the last one
            self.toggle_ROI_visibility(roi, force=ii==n_ROIs-1)

            # Increase progress counter
            dlg += 1

        # If there are ROIs present, enable the tool that allows to translate the field of view
        self.buttons.translation_ROI.setEnabled(n_ROIs > 0)

        # Find next least common color in drawn ROIs
        all_colors = self.COLORS
        ROI_colors = np.array(np.vstack(self.ROI_TABLE['color'].values), dtype=np.int16)
        ROI_colors = [tuple(c) for c in ROI_colors]
        color_frequency = np.zeros((len(all_colors), ), dtype=int)
        for c in ROI_colors:
            color_frequency[all_colors.index(c)] += 1
        self.next_ROI_color = all_colors[color_frequency.argmin()]

        if has_window:
            # Close progress dialog
            dlg.close()
            # Enable window
            self.window.setEnabled(True)
        # Log end of action
        log('... done')

    def callback_reload(self, what):
        if not self.menu.menubar.isEnabled():
            return
        # Check whether window is open (e.g., during __init__())
        has_window = hasattr(self, 'window')

        if what == 'frames':
            if has_window:
                answer = MessageBox('Reset all transformations to the field of view?', title='GUI ROI segmentation', parent=self.window, button_Yes='Yes, of all frames', button_No='Cancel', additional_buttons='Yes, only of selected frames')
                if answer == 'Cancel':
                    return
                # Disable window
                self.window.setEnabled(False)
                # Check which frames to reset
                reset_all = answer == 'Yes, of all frames'
            else:
                reset_all = True

            if reset_all:
                # Initialize table to contain all info on how to translate single frames
                self.TRANSFORMATION_MATRICES = pd.DataFrame(columns=['scale', 'rotation', 'translation'])
                # Initialize lookup table to find where info of each condition is stored in the table
                self.TRANSFORMATION_IDX = np.array([])
                initialize_field(self, 'TRANSFORMATION_IDX', 'array', initial_value='list', shape=(self.n_conditions, 2))
                self.TRANSFORMATION_IDX[:, 0] = np.arange(self.n_conditions, dtype=int)
                # Initialize lookup table of pixels in which each value is stored
                pixel_list = np.arange(self.PARAMETERS['n_pixels'], dtype=int)
                pixel_list = pixel_list.reshape(self.PARAMETERS['frame_height'], self.PARAMETERS['frame_width']).transpose().ravel()
                self.MAPPING_TABLE = np.tile(np.atleast_2d(pixel_list), (self.n_conditions, 1)).astype(np.float32)  # Float32 conversion to hold NaNs
                self.modified_MAPPING = np.zeros(shape=self.n_conditions, dtype=bool)
            else:
                # Get indices of trials to reset
                trials_to_reset = self.current_average_frame
                # Find transformations applied to these frames
                transformations = self.TRANSFORMATION_IDX[np.in1d(self.TRANSFORMATION_IDX[:, 0], trials_to_reset), 1]
                transformation_indices = np.unique(np.hstack(transformations))
                if transformation_indices.size != 0:  # These frames have not been transformed
                    # Find out whether these transformations are also applied to other frames
                    if not np.any(np.in1d(self.TRANSFORMATION_IDX[:, 1], transformation_indices)):
                        # Delete transformations from table
                        self.TRANSFORMATION_MATRICES.drop(transformation_indices, inplace=True)
                        self.TRANSFORMATION_MATRICES.reset_index(drop=True, inplace=True)
                    # Reset transformations for these frames
                    for trial in trials_to_reset:
                        self.TRANSFORMATION_IDX[trial, 1] = list()
                    # Initialize lookup table of pixels in which each value is stored
                    pixel_list = np.arange(self.PARAMETERS['n_pixels'], dtype=int)
                    pixel_list = pixel_list.reshape(self.PARAMETERS['frame_height'], self.PARAMETERS['frame_width']).transpose().ravel()
                    # Restore pixel mapping in these frames
                    self.MAPPING_TABLE[trials_to_reset, :] = pixel_list
                    self.modified_MAPPING[trials_to_reset] = False

            # Reset temporary variable
            self.temporary_MAPPING_TABLE = np.array([])
            # Update table
            self.update_table_condition_names()
            # Update image
            if has_window:
                self.compute_frames_projection()

        elif what == 'ROIs':
            if has_window:
                answer = MessageBox('Are you sure to delete all ROIs from memory?\n(copy on disk will not be affected)', title='GUI ROI segmentation', parent=self.window)
                if answer.lower() == 'no':
                    return
                # Disable window
                self.window.setEnabled(False)
                # Create progress dialog
                n_ROIs = self.ROI_TABLE['handle_ROI'].shape[0]
                dlg = ProgressDialog('Loading ROIs from disk', minimum=0, maximum=n_ROIs, parent=self.window)
            else:
                dlg = 0
            # Delete all previous ROIs
            all_ROIs = self.ROI_TABLE['handle_ROI'].values
            for roi in all_ROIs:
                if not isinstance(roi, np.ndarray):
                    # Delete ROI
                    self.callback_remove_ROI(roi, log_outcome=False)
                    # Increase counter
                    dlg += 1

        if has_window:
            # Enable window
            self.window.setEnabled(True)


    ############################################################################
    # GUI-related
    ############################################################################
    @staticmethod
    def make_image_box(lock_aspect_ratio=True, aspect_ratio=1, invert_y_axis=True, name=''):
        viewbox = CustomViewBox(name=name)
        image = pg.ImageItem()
        graphics = pg.GraphicsView()
        if lock_aspect_ratio:
            viewbox.setAspectLocked(True, ratio=aspect_ratio)
        viewbox.invertY(invert_y_axis)
        graphics.setCentralItem(viewbox)
        viewbox.addItem(image)

        return viewbox, image, graphics

    def toggle_crosshair(self, state):
        # Bring to top
        [c.setVisible(state=='on') for c in self.crosshair]
        [c.setZValue(1000) for c in self.crosshair]

    def toggle_toolbar(self, state, because):
        state_bool = state == 'on'

        if because == 'ROI_drawing':
            self.buttons.draw_ROI.setEnabled(state_bool)
            self.buttons.translation_ROI.setEnabled(state_bool)
            self.buttons.visibility_ROI.setEnabled(state_bool)
            self.buttons.link_views.setEnabled(state_bool)
            self.buttons.link_histograms.setEnabled(state_bool)
            self.buttons.anchor_trace.setEnabled(state_bool)

        elif because == 'translating_field_of_view':
            self.buttons.draw_ROI.setEnabled(state_bool)
            self.buttons.visibility_ROI.setEnabled(state_bool)
            self.buttons.link_views.setEnabled(state_bool)
            self.buttons.link_histograms.setEnabled(state_bool)
            self.buttons.anchor_trace.setEnabled(state_bool)

        elif because == 'changing_visibility_traces':
            self.buttons.draw_ROI.setEnabled(state_bool)
            self.buttons.translation_ROI.setEnabled(state_bool)
            self.buttons.visibility_ROI.setEnabled(state_bool)
            self.buttons.link_views.setEnabled(state_bool)
            self.buttons.link_histograms.setEnabled(state_bool)
            self.buttons.anchor_trace.setEnabled(state_bool)

    def toggle_slider_flood_tolerance(self, new_state):
        self.slider.setEnabled(new_state)
        self.slider_label.setEnabled(new_state)
        if new_state:
            self.slider.setStyleSheet('')
        else:
            self.slider.setStyleSheet('QSlider::sub-page:horizontal:disabled {background: #eee; border-color: #999;}')

    def make_region(self, regions):
        # Make sure input is a 2D array
        if not isinstance(regions, np.ndarray):
            regions = np.array(regions, dtype=int)
        if regions.ndim == 1:
            regions = np.atleast_2d(regions)

        # If other regions are present, delete them first
        if len(self.region_selection) > 0:
            self.delete_regions()

        # Add new regions
        for ireg in range(regions.shape[0]):
            # Crete region
            reg1 = pg.LinearRegionItem(movable=False, bounds=[0, self.PARAMETERS['n_frames']], brush=GUI_default['region_colors_dark'], pen=pg.mkPen(None))
            # Set region
            reg1.setRegion(regions[ireg, :])
            reg1.setZValue(0)
            self.frame_image_view.timeLine.getViewBox().addItem(reg1, ignoreBounds=True)

            # Crete region
            if len(self.current_average_frame) == 1 and self.trace_anchored:
                brush = pg.mkBrush(None)
            else:
                brush = GUI_default['region_colors_light']
            reg2 = pg.LinearRegionItem(movable=False, bounds=[0, self.PARAMETERS['n_frames']], brush=brush, pen=pg.mkPen(None))
            # Set region
            reg2.setRegion(regions[ireg, :])
            reg2.setZValue(0)
            self.trace_viewbox.addItem(reg2, ignoreBounds=True)

            # Keep values in memory
            self.region_selection.append([reg1, reg2])

    def delete_regions(self):
        if len(self.region_selection) > 0:
            [self.frame_image_view.timeLine.getViewBox().removeItem(ii[0]) for ii in self.region_selection]
            [self.trace_viewbox.removeItem(ii[1]) for ii in self.region_selection]
            self.region_selection = list()

    def compute_frames_projection(self, return_image=False, force_update_histogram=False):
        # Get frames
        frames = self.data_average[self.current_average_type][self.current_average_frame, :, :].copy()

        # Apply transformations
        n_frames = len(self.current_average_frame)
        for iframe in xrange(n_frames):
            T = self.TRANSFORMATION_IDX[self.TRANSFORMATION_IDX[:, 0] == self.current_average_frame[iframe], 1][0]
            if len(T) > 0:
                # Apply transformations to current frame one after the other
                for t_idx in T:
                    # Get matrices
                    scale = self.TRANSFORMATION_MATRICES.loc[t_idx, 'scale']
                    rotation = self.TRANSFORMATION_MATRICES.loc[t_idx, 'rotation']
                    translation = self.TRANSFORMATION_MATRICES.loc[t_idx, 'translation']
                    # Apply them
                    frames[iframe, :, :] = transform_image_from_matrices(frames[iframe, :, :], scale, rotation, translation)

        # Initialize output variable
        projection = None
        # Apply function to compute projection
        if self.current_projection_type == 'max':
            projection = np.nanmax(frames, axis=0)
        elif self.current_projection_type == 'mean':
            projection = np.nanmean(frames, axis=0)
        elif self.current_projection_type == 'median':
            projection = np.nanmedian(frames, axis=0)
        elif self.current_projection_type == 'standard deviation':
            projection = np.nanstd(frames, axis=0)

        # TODO: If correlation map is currently visualized, use red-blue colormap
        # if self.current_average_type == 'corr':
        #     pos, rgba_colors = zip(*cmapToColormap(mpl_colormaps.RdBu_r))
        #     colormap = pg.ColorMap(pos, rgba_colors)
        #     # Top left plot
        #     self.average_histogram.item.gradient.setColorMap(colormap)
        #
        # else:
        #     self.callback_toggle_colormap(force=self.current_colormap)

        if return_image:
            # Apply levels
            levels = self.average_histogram.item.getLevels()
            projection = np.clip(projection, a_min=levels[0], a_max=levels[1])

            return projection

        else:
            # Update image
            self.average_image_view.setImage(projection, autoLevels=False)
            # Update histogram
            if self.auto_adjust_histograms or force_update_histogram:
                projection = projection.ravel()
                levels = np.percentile(projection[~np.isnan(projection)], q=[1, 99])
                self.average_histogram.item.setLevels(levels[0], levels[1])
                self.average_histogram.item.autoHistogramRange()

    def callback_toggle_colormap(self, force=None):
        # Force a colormap to be equal to what the user wants
        if force is not None:
            self.current_colormap = force
        else:
            if self.current_colormap == 0:
                self.current_colormap = 1
            else:
                self.current_colormap = 0

        # Set colors
        if self.current_colormap == 1:
            background_color = 'w'
            foreground_color = 'k'
            timeline_pen = (0, 0, 0, 200)
            border_pen = (0, 0, 0)
        else:
            background_color = 'k'
            foreground_color = 'w'
            timeline_pen = (255, 255, 0, 200)
            border_pen = (255, 255, 255)
        # Top left plot
        self.average_histogram.item.gradient.setColorMap(self.colormap[self.current_colormap])
        self.average_image_viewbox.setBackgroundColor(background_color)
        self.image_views_title[0].setColor(foreground_color)
        self.average_histogram.setBackground(background_color)
        [self.FOV_borders[0][ii].setPen(border_pen) for ii in range(4)]

        # Top right plot
        self.frame_image_view.setColorMap(self.colormap[self.current_colormap])
        self.frame_image_view.view.setBackgroundColor(background_color)
        self.image_views_title[1].setColor(foreground_color)
        self.frame_image_view.ui.histogram.setBackground(background_color)
        self.frame_image_view.timeLine.getViewWidget().setBackground(background_color)
        self.frame_image_view.timeLine.setPen(timeline_pen)
        [self.FOV_borders[1][ii].setPen(border_pen) for ii in range(4)]
        # Change tick color in timeline
        items = self.frame_image_view.timeLine.scene().items()
        item = [ii for ii in items if isinstance(ii, pg.PlotItem)][0]
        item.axes['bottom']['item'].setPen(foreground_color)

        # Bottom panels
        self.trace_viewbox.setBackgroundColor(background_color)
        if self.stimulus_profile is not None:
            self.stimulus_profile.setPen(GUI_default['stimulus_profile_pen'][self.current_colormap])
        lut = self.colormap[self.current_colormap].getLookupTable(0.0, 1.0, 256)
        self.ROI_view.setLookupTable(lut)
        self.ROI_viewbox.setBackgroundColor(background_color)
        if self.selected_ROI is None:
            if self.current_colormap == 1:
                color = (255, 255, 255)
            else:
                color = (0, 0, 0)
            self.color_ROI_view_border(color)

    def color_ROI_view_border(self, color):
        if self.current_colormap == 1:
            if color is None:
                color = (255, 255, 255)
            thickness = 5
        else:
            if color is None:
                color = (0, 0, 0)
            thickness = 6
        # Apply color
        self.ROI_border.setStyleSheet('border:%ipx solid rgb(%i, %i, %i);' % (thickness, color[0], color[1], color[2]))

    def callback_menu_projection_type(self, name):
        if not self.menu.menubar.isEnabled():
            return
        # Set new name
        self.current_projection_type = name
        # Update plot
        self.compute_frames_projection(force_update_histogram=True)

    def callback_menu_average_type(self, name):
        if not self.menu.menubar.isEnabled():
            return
        # Set new name
        self.current_average_type = name
        # Update plot
        self.compute_frames_projection(force_update_histogram=True)

    def callback_menu_operation_type(self, name):
        if not self.menu.menubar.isEnabled():
            return
        # Set new name
        self.current_operation_type = name
        # Update visible traces
        table_rows = self.ROI_TABLE['show_trace'].index.values
        for roi in self.ROI_TABLE.loc[table_rows, 'handle_ROI'].tolist():
            self.callback_update_zoom_ROI(roi, update_trace=True, update_ROI_contour=False)
        self.average_histogram.item.autoHistogramRange()

    def callback_update_histogram(self, source_histogram, update_average, update_ROI, update_frame):
        if not self.updating_histograms:
            # Set flag
            self.updating_histograms = True
            # Get levels and update levels on other histogram
            if source_histogram == 'average':
                levels = self.average_histogram.item.getLevels()
                self.frame_image_view.getHistogramWidget().item.setLevels(levels[0], levels[1])
            elif source_histogram == 'frame':
                levels = self.frame_image_view.getHistogramWidget().item.getLevels()
                self.average_histogram.item.setLevels(levels[0], levels[1])

            # Apply them
            if update_average:
                self.average_image_view.setLevels(levels)
            if update_ROI:
                self.ROI_view.setLevels(levels)
            if update_frame:
                self.frame_image_view.imageItem.setLevels(levels)
            # Reset flag
            self.updating_histograms = False

    def callback_link_histograms(self, new_state):
        self.histograms_linked = new_state
        frame_hist = self.frame_image_view.getHistogramWidget().item
        # Disconnect previous callbacks
        try:
            self.average_histogram.item.sigLevelsChanged.disconnect()
            frame_hist.sigLevelsChanged.disconnect()
        except TypeError, e:
            # An error occurs when no callback was previously connected. The error message contains 'disconnect() failed'
            if 'disconnect() failed' not in str(e):
                raise  # Re-raise error

        # Connect new callbacks
        if new_state:
            self.average_histogram.item.sigLevelsChanged.connect(partial(self.callback_update_histogram, source_histogram='average', update_average=False, update_ROI=True, update_frame=True))
            frame_hist.sigLevelsChanged.connect(partial(self.callback_update_histogram, source_histogram='frame', update_average=True, update_ROI=True, update_frame=False))
            # Force histograms to automatically adjust range
            self.callback_auto_adjust_histograms(new_state=True)
        else:  # Restore original signals
            self.average_histogram.item.sigLevelsChanged.connect(self.average_histogram.item.regionChanged)
            frame_hist.sigLevelsChanged.connect(frame_hist.regionChanged)

        # Update icon
        if new_state:
            icon = QtGui.QIcon(QtGui.QPixmap(GUI_default['icons']['link_histograms'][1]))
        else:
            icon = QtGui.QIcon(QtGui.QPixmap(GUI_default['icons']['link_histograms'][0]))
        self.buttons.link_histograms.setIcon(icon)

    def callback_auto_adjust_histograms(self, new_state):
        if not self.menu.menubar.isEnabled():
            return
        self.auto_adjust_histograms = new_state

    def callback_link_views(self, new_state):
        self.views_linked = new_state
        # If other regions are present, delete them first
        if len(self.region_selection) > 0:
            self.delete_regions()
        self.callback_timeline_frames(from_plot='frame')

        # Update icon
        if new_state:
            icon = QtGui.QIcon(QtGui.QPixmap(GUI_default['icons']['link_views'][1]))
        else:
            icon = QtGui.QIcon(QtGui.QPixmap(GUI_default['icons']['link_views'][0]))
        self.buttons.link_views.setIcon(icon)

    def callback_anchor_trace(self, new_state):
        self.trace_anchored = new_state
        # If other regions are present, delete them first
        if len(self.region_selection) > 0:
            self.delete_regions()
        self.callback_timeline_frames(from_plot='frame')

        # Update icon
        if new_state:
            icon = QtGui.QIcon(QtGui.QPixmap(GUI_default['icons']['anchor_trace'][1]))
        else:
            icon = QtGui.QIcon(QtGui.QPixmap(GUI_default['icons']['anchor_trace'][0]))
        self.buttons.anchor_trace.setIcon(icon)

    def callback_continue_last_action(self):
        if self.current_action == 'drawing_ROI':
            self.callback_draw_ROI()
        elif self.current_action == 'translating_FOV':
            self.callback_accept_translation_FOV()

        elif self.current_action == '':
            roi = self.selected_ROI
            table_row = self.ROI_TABLE[self.ROI_TABLE['id'] == roi.ROI_id].index.values[0]
            ROI_type = self.ROI_TABLE.loc[table_row, 'type']
            if ROI_type == 'fill':
                self.callback_update_zoom_ROI(roi, update_trace=True, update_ROI_contour=True)

    def callback_cancel_last_action(self):
        # Make sure no other action is invoked from these operations
        last_action = self.current_action
        self.current_action = ''

        if last_action == 'drawing_ROI':
            if self.last_ROI_coordinates_handle is not None:
                self.average_image_viewbox.removeItem(self.last_ROI_coordinates_handle)
            self.last_ROI_coordinates_handle = None
            self.last_ROI_coordinates = list()
            # Inactivate crosshair
            self.toggle_crosshair('off')
            # Drawing this ROI is finished, so re-enable menus
            self.menu.menubar.setEnabled(True)
            self.toggle_toolbar('on', because='ROI_drawing')
            # Reset button style
            self.buttons.draw_ROI.setStyleSheet('background-color: None')

        elif last_action == 'translating_FOV':
            # Clear image and reset data into it
            self.average_image_viewbox.removeItem(self.average_image_view)
            self.average_image_viewbox.removeItem(self.translation_ROI)
            self.average_image_view = pg.ImageItem(self.translation_image)
            self.average_image_view.setZValue(0)
            self.average_image_viewbox.addItem(self.average_image_view)
            # Reassign image to histogram widget and set levels to what they were before
            self.average_histogram.item.setImageItem(self.average_image_view)
            self.average_image_view.setLevels(self.translation_image_levels)
            # Disconnect callback
            self.translation_ROI.sigRegionChanged.disconnect()
            # Toggle internal flags
            self.translation_ROI = None
            self.translation_image = None
            self.translation_image_levels = None
            self.temporary_MAPPING_TABLE = np.array([])
            # Toggle GUI button
            self.buttons.translation_ROI.setChecked(False)
            # Make sure colormap is correct
            self.callback_toggle_colormap(force=self.current_colormap)
            self.toggle_toolbar('on', because='translating_field_of_view')

        elif self.toggling_ROI_visibility:
            # Inactivate crosshair
            self.toggle_crosshair('off')
            self.toggling_ROI_visibility = False
            self.toggle_toolbar('on', because='changing_visibility_traces')

    def callback_fix_colormap(self):
        self.callback_toggle_colormap(force=self.current_colormap)


    ############################################################################
    # ROI-related
    ############################################################################
    def get_next_ROI_color(self):
        if self.next_ROI_color is None:
            color_idx = 0
        else:
            n_colors = len(self.COLORS)
            last_color_idx = self.COLORS.index(self.next_ROI_color)
            # Wrap index
            color_idx = (last_color_idx + 1) % n_colors
        self.next_ROI_color = self.COLORS[color_idx]

    def callback_choose_ROI_type(self, position):
        # Make menu
        menu = QtWidgets.QMenu(self.window)
        # List all ROI types allowed
        menu.addAction(self.buttons.ROI_fill)
        menu.addAction(self.buttons.ROI_polygon)
        # Show menu and let user choose an action
        chosen_action = menu.exec_(self.buttons.draw_ROI.mapToGlobal(position))

        # Set flag
        if chosen_action == self.buttons.ROI_fill:
            self.selected_ROI_type = 'fill'
        elif chosen_action == self.buttons.ROI_polygon:
            self.selected_ROI_type = 'polygon'
        else:
            return
        # Set icon
        if GUI_default['icons_kind']['ROI_' + self.selected_ROI_type] == 'text':
            self.buttons.draw_ROI.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
            self.buttons.draw_ROI.setText(GUI_default['icons']['ROI_' + self.selected_ROI_type])
        elif GUI_default['icons_kind']['ROI_' + self.selected_ROI_type] == 'icon':
            self.buttons.draw_ROI.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
            self.buttons.draw_ROI.setIcon(QtGui.QIcon(QtGui.QPixmap(GUI_default['icons']['ROI_' + self.selected_ROI_type])))
        # Toggle flood tolerance slider
        self.toggle_slider_flood_tolerance(self.selected_ROI_type == 'fill')

    def callback_draft_ROI(self):
        if self.current_action == '':  # If not doing anything else already
            # Disable toolbars
            self.menu.menubar.setEnabled(False)
            self.toggle_toolbar('off', because='ROI_drawing')
            # Activate crosshair
            self.toggle_crosshair('on')
            # Toggle internal flag
            self.current_action = 'drawing_ROI'
            # Change background color of button
            self.buttons.draw_ROI.setStyleSheet(str('background-color: rgb%s' % str(tuple(np.array(GUI_default['icon_background']['no_go'], dtype=float) * 255))))
            # Set color to assign to next ROI
            self.get_next_ROI_color()
            # Toggle flood tolerance slider
            self.toggle_slider_flood_tolerance(self.selected_ROI_type == 'fill')

    def callback_draw_ROI(self):
        # Check that user is drawing an ROI
        if self.current_action == 'drawing_ROI':
            # Toggle internal flag
            self.current_action = ''

            # Get user inputs to make ROI
            points = self.last_ROI_coordinates

            # Delete markers
            if self.last_ROI_coordinates_handle is not None:
                self.average_image_viewbox.removeItem(self.last_ROI_coordinates_handle)
            self.last_ROI_coordinates_handle = None
            self.last_ROI_coordinates = list()

            # Get color of this ROI
            pen = pg.mkPen(color=self.next_ROI_color, width=GUI_default['ROI_linewidth_thick'])
            # Compute points
            if self.selected_ROI_type == 'fill':
                # Reset value of tolerance
                self.flood_tolerance = GUI_default['flood_tolerance']
                reference_pixel = points[0].copy()
                points = self.make_ROI_from_point(reference_pixel)
            else:
                reference_pixel = 0

            if len(points) < 3:  # Cannot close an area with less than 3 points
                return
            # Make ROI
            roi = PolyLineROI(points, pen=pen, closed=True, movable=True, removable=True)
            # Automatically select this ROI
            self.selected_ROI = roi
            # Add contextual menu to change color of the ROI
            self.ROI_add_context_menu(roi)
            # Add ROI to plot
            self.average_image_viewbox.addItem(roi)

            # Connect callbacks
            roi.sigRegionChanged.connect(partial(self.callback_update_zoom_ROI, update_trace=True, update_ROI_contour=True))
            roi.sigRemoveRequested.connect(partial(self.callback_remove_ROI, log_outcome=True))
            roi.sigClicked.connect(self.callback_click_ROI)
            roi.setAcceptedMouseButtons(QtCore.Qt.LeftButton)

            # Store info on this ROI
            ROI_id = make_new_ROI_id(self.ROI_TABLE['id'])
            roi.ROI_id = ROI_id  # Write it in the handle so we can always come back to the table without checking where the handle was
            # Get number of traces shown before adding the new one
            shown_traces = np.where(self.ROI_TABLE['show_trace'])[0].shape[0]
            last_row = self.ROI_TABLE.shape[0]
            self.ROI_TABLE.at[last_row, 'id'] = ROI_id
            self.ROI_TABLE.at[last_row, 'handle_ROI'] = roi
            self.ROI_TABLE.at[last_row, 'handle_ROI_contour'] = None
            self.ROI_TABLE.at[last_row, 'handle_trace'] = None
            self.ROI_TABLE.at[last_row, 'type'] = self.selected_ROI_type
            self.ROI_TABLE.at[last_row, 'reference_pixel'] = reference_pixel
            self.ROI_TABLE.at[last_row, 'flood_tolerance'] = self.flood_tolerance
            self.ROI_TABLE.at[last_row, 'region'] = points
            self.ROI_TABLE.at[last_row, 'color'] = self.next_ROI_color
            self.ROI_TABLE.at[last_row, 'show_trace'] = True

            # Deactivate crosshair
            self.toggle_crosshair('off')
            # Reset button style
            self.buttons.draw_ROI.setStyleSheet('background-color: None')
            # Toggle flood tolerance slider
            self.toggle_slider_flood_tolerance(self.selected_ROI_type == 'fill')

            # Transform ROI to a static polygon
            points = np.array(roi.getState()['points'])
            points = np.vstack((points, points[0, :]))
            # Make PlotItem
            pi_right = pg.PlotCurveItem(clickable=False)
            pi_right.setData(x=points[:, 0], y=points[:, 1])
            self.frame_image_view.view.addItem(pi_right)
            pi_right.setPen(pen)
            self.ROI_TABLE.at[last_row, 'handle_ROI_contour'] = pi_right
            # Update plot range
            self.frame_image_view.view.autoRange()

            # Change color around ROI viewbox
            self.color_ROI_view_border(color=self.next_ROI_color)

            # Drawing this ROI is finished, so re-enable menus
            self.menu.menubar.setEnabled(True)
            self.toggle_toolbar('on', because='ROI_drawing')
            # If there are ROIs present, enable the tool that allows to translate the field of view
            n_ROIs = self.ROI_TABLE.shape[0]
            self.buttons.translation_ROI.setEnabled(n_ROIs > 0)
            # Trigger update immediately
            if self.selected_ROI_type != 'fill':
                roi.sigRegionChanged.emit(roi)
            # Update trace range
            if shown_traces == 0:
                self.callback_reset_zoom(where='trace')
            # Set which traces can be shown
            self.callback_update_visibility_traces(from_button=False)
            self.update_trace_zvalue()

            # Set value of slider to tolerance value used to compute this ROI
            if self.selected_ROI_type == 'fill':
                self.slider.blockSignals(True)
                self.slider.setValue(self.flood_tolerance * 10)
                self.slider_label.setText('%.1f' % self.flood_tolerance)
                self.slider.blockSignals(False)

            # Log outcome
            log('Added ROI #%i' % roi.ROI_id)

            # Re-start tool for drawing ROIs
            self.callback_draft_ROI()

    def callback_update_zoom_ROI(self, roi, update_trace=True, update_ROI_contour=True):
        # If user clicked on a child ROI, get the handle of the parent
        if not hasattr(roi, 'ROI_id'):
            roi = roi.parent

        # Get contour of this ROI
        points = np.array(roi.getState()['points'])
        # Return immediately if ROI is empty
        if points.size == 0:
            return

        # Automatically select this ROI if any of its parts is moving or no ROI was previously selected
        if self.selected_ROI is None or roi.isMoving or any([h.isMoving for h in roi.getHandles()]):
            self.selected_ROI = roi

        # Get array region from projection selection graph
        image_to_analyze = self.compute_frames_projection(return_image=True)
        area, coords = roi.getArrayRegion(image_to_analyze, self.average_image_view, axes=(0, 1), returnMappedCoords=True)

        if area.size == 0:
            return

        # Show ROI in lower left panel, if this ROI is selected
        if roi is self.selected_ROI:
            self.ROI_view.setImage(area, autoLevels=True)
            self.ROI_viewbox.autoRange()
            # Change color around ROI viewbox
            color = self.ROI_TABLE.loc[self.ROI_TABLE['id'] == roi.ROI_id, 'color'].values[0]
            self.color_ROI_view_border(color=color)

        # Get row with info on this ROI
        table_row = self.ROI_TABLE[self.ROI_TABLE['id'] == roi.ROI_id].index.values[0]
        # Update corners of ROI
        self.ROI_TABLE.at[table_row, 'region'] = points

        # Get ROI type
        ROI_type = self.ROI_TABLE.loc[table_row, 'type']
        if ROI_type == 'fill' and not self.slider_update_GUI:
            return

        # Update trace, if allowed
        if self.ROI_TABLE.loc[table_row, 'show_trace'] and update_trace:
            # Get indices of ROI pixels
            area_idx = np.where(area > 0)
            ROI_idx = np.vstack((coords[0, area_idx[0], area_idx[1]], coords[1, area_idx[0], area_idx[1]]))
            ROI_pixels = np.unique(np.ravel_multi_index(np.round(ROI_idx, decimals=0).astype(int), [self.PARAMETERS['frame_height'], self.PARAMETERS['frame_width']], mode='clip', order='C'))
            # Luminance values
            if self.current_operation_type == 'mean':
                L = np.mean(self.data_time[ROI_pixels, :], axis=0)
            elif self.current_operation_type == 'max':
                L = np.max(self.data_time[ROI_pixels, :], axis=0)

            # Check whether some segments have been translated
            if self.current_action == 'translating_FOV' or np.any(self.modified_MAPPING):
                # Get index of frames that are modified
                modified_conditions_idx = np.where(self.modified_MAPPING == True)[0]
                if self.current_action == 'translating_FOV':
                    # Get index of currently selected frames
                    temp_modified_conditions_idx = np.array(self.current_average_frame, dtype=int)
                    # Remove frames that will be replaced
                    modified_conditions_idx = np.setdiff1d(modified_conditions_idx, temp_modified_conditions_idx).ravel()
                # Get segments that have been translated
                if modified_conditions_idx.size > 0:
                    L_modified, modified_frames_idx = self.get_translated_trace_segments(modified_conditions_idx, ROI_pixels, from_temporary=False)
                    # Replace values in original array
                    L[modified_frames_idx] = L_modified
                # Get segments from frames that are currently being translated
                if self.current_action == 'translating_FOV':
                    try:
                        L_modified, modified_frames_idx = self.get_translated_trace_segments(temp_modified_conditions_idx, ROI_pixels, from_temporary=True)
                        # Replace values in original array
                        L[modified_frames_idx] = L_modified
                    except TypeError:
                        pass

            # Check whether we have to create a new trace
            if self.ROI_TABLE.loc[table_row, 'handle_trace'] is None:
                # Make new PlotCurveItem
                pi = pg.PlotCurveItem(clickable=False)
                self.trace_viewbox.addItem(pi)
                # Store handle
                self.ROI_TABLE.loc[table_row, 'handle_trace'] = pi
                # Make pen
                ROI_color = self.ROI_TABLE.loc[table_row, 'color']
                pen = pg.mkPen(color=ROI_color)
                pi.setPen(pen)
            else:  # Take curve handle
                pi = self.ROI_TABLE.loc[table_row, 'handle_trace']

            # Update data
            pi.setData(L, connect='finite')
            # Update z-values
            self.update_trace_zvalue()

        if update_ROI_contour:
            # Update ROI overlay
            points = np.array(roi.getState()['points'])
            points = np.vstack((points, points[0, :]))
            # Add offset from original position
            points += np.array(roi.state['pos'])
            # Update data
            self.ROI_TABLE.loc[table_row, 'handle_ROI_contour'].setData(x=points[:, 0], y=points[:, 1])

    def update_trace_zvalue(self):
        # Reset z-value of all traces to 2
        for row in self.ROI_TABLE.index.values:
            handle_trace = self.ROI_TABLE.loc[row, 'handle_trace']
            if handle_trace is not None:
                handle_trace.setZValue(2)
        # Set z-value of current ROI's trace to 3
        table_row = self.ROI_TABLE[self.ROI_TABLE['id'] == self.selected_ROI.ROI_id].index.values[0]
        self.ROI_TABLE.loc[table_row, 'show_trace'] = True
        handle_trace = self.ROI_TABLE.loc[table_row, 'handle_trace']
        if handle_trace is None:
            self.callback_update_zoom_ROI(self.selected_ROI, update_trace=True, update_ROI_contour=True)
            handle_trace = self.ROI_TABLE.loc[table_row, 'handle_trace']
        handle_trace.setZValue(3)

    def callback_update_visibility_traces(self, from_button):
        if from_button:
            self.toggling_ROI_visibility = True
            self.toggle_crosshair('on')
            self.toggle_toolbar('off', because='changing_visibility_traces')
        else:
            max_n = GUI_default['max_n_visible_ROI_traces']
            # Get all unselected ROIs
            all_ROIs = self.ROI_TABLE.loc[self.ROI_TABLE['show_trace'], 'handle_ROI'].values
            unselected_ROIs = np.array([roi for roi in all_ROIs if roi is not self.selected_ROI], dtype=object)
            # Discard last max_n - 1 unselected ROIs
            if max_n == 0:
                unselected_ROIs_to_hide = unselected_ROIs
                unselected_ROIs_to_show = list()
            else:
                unselected_ROIs_to_hide = unselected_ROIs[:-max_n]
                # Make sure that selected ROI is visible and also other unselected ROIs
                if len(unselected_ROIs) > 0:
                    unselected_ROIs_to_show = list([unselected_ROIs[-1]])
                else:
                    unselected_ROIs_to_show = list()
            unselected_ROIs_to_show.append(self.selected_ROI)
            # Toggle visibility
            for roi in unselected_ROIs_to_hide:
                self.toggle_ROI_visibility(roi, force=False)
            for roi in unselected_ROIs_to_show:
                self.toggle_ROI_visibility(roi, force=True)

    def toggle_ROI_visibility(self, roi, force):
        if force is None:
            new_state = roi.pen.style() == QtCore.Qt.CustomDashLine
        else:
            new_state = force

        table_row = self.ROI_TABLE[self.ROI_TABLE['id'] == roi.ROI_id].index.values[0]
        if new_state:
            # Show traces
            self.ROI_TABLE.loc[table_row, 'show_trace'] = True
            self.callback_update_zoom_ROI(roi, update_trace=True, update_ROI_contour=True)
            # Change style of ROI in overlays
            pen = roi.pen
            pen.setStyle(QtCore.Qt.SolidLine)
            pen.setColor(pg.mkColor(self.ROI_TABLE.loc[table_row, 'color']))
            roi.setPen(pen)
            self.ROI_TABLE.loc[table_row, 'handle_ROI_contour'].setPen(pen)
            # Disable interaction with ROI
            roi.allowed_interaction = True
            roi.translatable = True

        else:
            # If trace was shown
            if self.ROI_TABLE.loc[table_row, 'show_trace']:
                # Hide trace
                handle_trace = self.ROI_TABLE.loc[table_row, 'handle_trace']
                self.trace_viewbox.removeItem(handle_trace)
                self.ROI_TABLE.loc[table_row, 'handle_trace'] = None
                self.ROI_TABLE.loc[table_row, 'show_trace'] = False
            # Change style of ROI in overlays
            pen = roi.pen
            pen.setStyle(QtCore.Qt.CustomDashLine)
            pen.setDashPattern([1, 3])
            pen.setColor(pg.mkColor(255, 255, 0))
            roi.setPen(pen)
            self.ROI_TABLE.loc[table_row, 'handle_ROI_contour'].setPen(pen)
            # Disable interaction with ROI
            roi.allowed_interaction = False
            roi.translatable = False

    def callback_remove_ROI(self, roi, log_outcome=True):
        # Disconnect callbacks
        roi.sigRegionChanged.disconnect()
        roi.sigRemoveRequested.disconnect()
        roi.sigClicked.disconnect()

        # Get row in table where this ROI is stored
        table_row = list(self.ROI_TABLE[self.ROI_TABLE['id'] == roi.ROI_id].index.values)
        if len(table_row) > 0:
            table_row = table_row[0]
        else:
            return

        # If this one was the selected ROI, remove it from the inspection view
        if roi is self.selected_ROI:
            self.ROI_view.clear()
            self.color_ROI_view_border(color=None)
            self.selected_ROI = None
        # Delete contour from frame plot
        self.frame_image_view.removeItem(self.ROI_TABLE.at[table_row, 'handle_ROI_contour'])
        # Delete trace
        self.trace_viewbox.removeItem(self.ROI_TABLE.loc[table_row, 'handle_trace'])

        # Delete handle from average view
        self.average_image_viewbox.removeItem(roi)
        # Remove row from table
        self.ROI_TABLE.drop(table_row, inplace=True)
        # Toggle the button that allows the user to translate the entire field of view
        n_ROIs = self.ROI_TABLE.shape[0]
        self.buttons.translation_ROI.setEnabled(n_ROIs > 0)
        # Log outcome
        if log_outcome:
            log('Deleted ROI #%i' % roi.ROI_id)

    def callback_click_ROI(self, roi):
        # Toggle ROI visibility
        if self.toggling_ROI_visibility:
            self.toggle_ROI_visibility(roi, force=None)
            return
        # Store this ROI in memory to be used later
        self.selected_ROI = roi
        # If this ROI is of type 'fill', toggle the slider tool
        table_row = self.ROI_TABLE[self.ROI_TABLE['id'] == self.selected_ROI.ROI_id].index.values[0]
        ROI_type = self.ROI_TABLE.loc[table_row, 'type']
        # Toggle flood tolerance slider
        if ROI_type == 'fill':
            self.slider.blockSignals(True)
            # Restore last value of flood tolerance
            flood_tolerance = self.ROI_TABLE.loc[table_row, 'flood_tolerance']
            self.slider.setValue(flood_tolerance * 10)
            self.slider_label.setText('%.1f' % flood_tolerance)
            self.slider.blockSignals(False)
        self.toggle_slider_flood_tolerance(ROI_type == 'fill')

        # Update visibility trace
        self.callback_update_visibility_traces(from_button=False)
        # Update GUI
        self.update_trace_zvalue()

    def ROI_add_context_menu(self, roi):
        # Create submenu
        roi.submenu = QtWidgets.QMenu('Change color')
        roi.menu.addMenu(roi.submenu)
        # Add actions
        for idx, color in enumerate(self.COLORS):
            # Create action and set its color
            action = QtWidgets.QWidgetAction(roi.submenu)
            label = QtWidgets.QLabel(' ' * 20)
            label.setFixedHeight(30)
            # Set background color
            stylesheet = 'background-color: rgb(%i, %i, %i); ' % color
            if idx != 0:
                stylesheet += 'border-top: 2px solid white; '
            if idx != len(self.COLORS)-1:
                stylesheet += 'border-bottom: 2px solid white; '
            label.setStyleSheet(stylesheet)
            action.setDefaultWidget(label)
            action.triggered.connect(partial(self.ROI_change_color, roi, idx))
            roi.submenu.addAction(action)

    def ROI_change_color(self, roi, color_idx):
        # Update ROI color only if it is different
        current_ROI_color = self.ROI_TABLE.loc[self.ROI_TABLE['id'] == roi.ROI_id, 'color'].values[0]
        selected_color = self.COLORS[color_idx]
        if not current_ROI_color == selected_color:
            # Update color in memory
            self.ROI_TABLE.loc[self.ROI_TABLE['id'] == roi.ROI_id, 'color'] = [selected_color]
            # Make pen
            pen = pg.mkPen(selected_color, width=GUI_default['ROI_linewidth_thick'])
            # Update color of ROIs
            roi.setPen(pen)
            self.ROI_TABLE.loc[self.ROI_TABLE['id'] == roi.ROI_id, 'handle_ROI_contour'].values[0].setPen(pen)
            # Update color of trace
            handle_trace = self.ROI_TABLE.loc[self.ROI_TABLE['id'] == roi.ROI_id, 'handle_trace'].values[0]
            if handle_trace is not None:
                pen = pg.mkPen(color=selected_color)
                handle_trace.setPen(pen)
            # Select this ROI
            self.selected_ROI = roi
            # Update ROI inspection view
            self.callback_update_zoom_ROI(roi, update_trace=True, update_ROI_contour=False)

    def make_ROI_from_point(self, xy_point):
        # Round points
        xy_coords = np.round(np.array(xy_point)).astype(int).ravel()
        # Get currently shown image
        img = self.compute_frames_projection(return_image=True)
        img[np.isnan(img)] = 0
        # Normalize image to [0-100] range
        image = img / np.nanmax(img) * 100.
        # image = 255 * image
        # image = image.astype(np.uint8)
        # Make a 4-connected neighbor structure
        structure = np.zeros((3, 3), dtype=int)
        structure[1, :] = 1
        structure[:, 1] = 1

        # Get value of reference pixel
        ref = image[xy_coords[0], xy_coords[1]]
        color_mask = ref - image <= self.flood_tolerance
        objects = ndimage.label(color_mask, structure=structure)[0]
        # Find objects
        [x, y] = np.where(objects > 0)
        v = objects[objects > 0]
        xy = np.vstack((x, y)).transpose()
        # Find nearest object
        object_idx = np.argmin(np.sum((xy - xy_point)**2, axis=1))
        object_id = v[object_idx]
        # Assign label
        mask = np.zeros((self.PARAMETERS['frame_height'], self.PARAMETERS['frame_width']))
        mask[objects == object_id] = 1
        # Fill holes
        mask = ndimage.morphology.binary_fill_holes(mask, structure=structure).astype(np.uint8)

        # Detect contours in the mask
        points = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1 )[1]
        points = np.squeeze(points[0])

        # Approximate curve
        n_points = points.shape[0]
        max_n_points = GUI_default['max_n_points']
        epsilon = 0
        new_points = points.copy()
        while n_points >= max_n_points:
            new_points = cv2.approxPolyDP(points, epsilon=epsilon, closed=True)
            n_points = new_points.shape[0]
            epsilon += .1
        # Make 2D
        points = np.fliplr(np.squeeze(new_points))

        return points

    def callback_flood_tolerance_value_changed(self, new_value):
        if self.current_action == '':
            # Get type of currently selected ROI
            table_row = self.ROI_TABLE[self.ROI_TABLE['id'] == self.selected_ROI.ROI_id].index.values[0]
            ROI_type = self.ROI_TABLE.loc[table_row, 'type']
            if ROI_type == 'fill':
                # Update slider's label
                self.flood_tolerance = new_value / 10.
                self.slider_label.setText('%.1f' % self.flood_tolerance)

                points = self.make_ROI_from_point(self.ROI_TABLE.loc[table_row, 'reference_pixel'])
                if len(points) < 3:  # Cannot close an area with less than 3 points
                    return
                # Temporarily stop updating trace on slider change
                self.slider_update_GUI = False
                # Update points
                self.selected_ROI.setPoints(points, closed=True)
                # Store new points
                self.ROI_TABLE.loc[table_row, 'region'] = points
                # Store current value of flood tolerance used to calculate these points
                self.ROI_TABLE.loc[table_row, 'flood_tolerance'] = self.flood_tolerance
                # Restore update of the GUI
                self.slider_update_GUI = True

    def callback_activate_translation_FOV(self, new_state):
        # Enabling the tool
        if new_state:
            self.toggle_toolbar('off', because='translating_field_of_view')
            # Make ROI that is as big as field of view
            self.translation_ROI = ROI([0, 0], [self.PARAMETERS['frame_height'], self.PARAMETERS['frame_width']], removable=False, invisible=False)
            self.translation_ROI.setZValue(0)
            # Add rotation and scaling handles
            self.translation_ROI.addRotateHandle([1, 1], center=[0.5, 0.5])
            self.translation_ROI.addScaleHandle([0.5, 0], center=[0.5, 1])
            self.translation_ROI.addScaleHandle([0.5, 1], center=[0.5, 0])
            self.translation_ROI.addScaleHandle([0, 0.5], center=[1, 0.5])
            self.translation_ROI.addScaleHandle([1, 0.5], center=[0, 0.5])
            # Get last image and delete ImageItem
            self.translation_image_levels = self.average_image_view.levels
            self.translation_image = self.average_image_view.image.copy()
            self.average_image_viewbox.removeItem(self.average_image_view)
            # Assign current image to ROI
            self.average_image_view = pg.ImageItem(self.translation_image)
            self.average_image_viewbox.addItem(self.translation_ROI)
            self.translation_ROI.setPos(0, 0, update=True, finish=True)
            self.average_image_view.setParentItem(self.translation_ROI)
            # Reassign image to histogram widget and set levels to what they were before
            self.average_histogram.item.setImageItem(self.average_image_view)
            self.average_image_view.setLevels(self.translation_image_levels)
            self.callback_toggle_colormap(force=self.current_colormap)
            # Connect callback
            self.translation_ROI.sigRegionChanged.connect(self.callback_update_image_while_translating_FOV)
            self.translation_ROI.sigRegionChanged.emit(self.translation_ROI)

            # Clear other graphs
            self.ROI_view.clear()
            # Toggle internal flag
            self.current_action = 'translating_FOV'

        # Accepting translation
        else:
            self.callback_accept_translation_FOV()

    def callback_accept_translation_FOV(self):
        if self.current_action == 'translating_FOV' and not self.translation_ROI.isMoving:
            # Toggle internal flag to forbid entering this section again before its end
            self.current_action = ''
            self.toggle_toolbar('on', because='translating_field_of_view')
            # Disconnect callback
            self.translation_ROI.sigRegionChanged.disconnect()
            # Read translation_ROI state
            scale, angle, offset = self.get_FOV_translation_state()

            # Transform shown image
            new_image, scaling_matrix, rotation_matrix, translation_matrix = transform_image_from_parameters(self.translation_image.copy(), scale, angle, offset, return_matrices=True)
            # Get index where to save the transformation matrices
            table_idx = self.TRANSFORMATION_MATRICES.shape[0]
            # Store this index in the lookup table
            [i.append(table_idx) for i in self.TRANSFORMATION_IDX[self.current_average_frame, 1]]
            # Store parameters in memory
            self.TRANSFORMATION_MATRICES.at[table_idx, 'scale'] = scaling_matrix
            self.TRANSFORMATION_MATRICES.at[table_idx, 'rotation'] = rotation_matrix
            self.TRANSFORMATION_MATRICES.at[table_idx, 'translation'] = translation_matrix

            # Clear image and reset data into it
            self.average_image_viewbox.removeItem(self.average_image_view)
            self.average_image_viewbox.removeItem(self.translation_ROI)
            self.average_image_view = pg.ImageItem(new_image)
            self.average_image_view.setZValue(0)
            self.average_image_viewbox.addItem(self.average_image_view)
            # Reassign image to histogram widget and set levels to what they were before
            self.average_histogram.item.setImageItem(self.average_image_view)
            self.average_image_view.setLevels(self.translation_image_levels)

            # Apply transformation to current frame, if it falls in current condition
            ind, _ = self.frame_image_view.timeIndex(self.frame_image_view.timeLine)
            condition_idx = self.frames_idx_array[ind]
            if condition_idx in self.current_average_frame:
                self.callback_transform_current_frame()

            # Initialize position of each pixel and transform this grid
            pixel_list = np.arange(self.PARAMETERS['n_pixels'], dtype=int)
            pixel_list = pixel_list.reshape(self.PARAMETERS['frame_height'], self.PARAMETERS['frame_width']).transpose().ravel()
            pixel_grid = np.reshape(pixel_list, (self.PARAMETERS['frame_width'], self.PARAMETERS['frame_height']))
            pixel_grid = transform_image_from_parameters(pixel_grid, scale, -angle, offset[::-1])
            # Round pixel value to nearest integer
            pixel_grid = np.round(pixel_grid)
            # Replace original pixel positions with these ones
            self.MAPPING_TABLE[self.current_average_frame, :] = pixel_grid.transpose().ravel()
            self.modified_MAPPING[self.current_average_frame] = True
            # Update shown traces
            table_rows = self.ROI_TABLE['show_trace'].index.values
            for roi in self.ROI_TABLE.loc[table_rows, 'handle_ROI'].tolist():
                self.callback_update_zoom_ROI(roi, update_trace=True, update_ROI_contour=False)

            # Toggle internal flags
            self.translation_ROI = None
            self.translation_image = None
            self.temporary_MAPPING_TABLE = np.array([])
            self.translation_image_levels = None
            # Toggle GUI button
            self.buttons.translation_ROI.setChecked(False)
            # Make sure colormap is correct
            self.callback_toggle_colormap(force=self.current_colormap)
            # Update table
            self.update_table_condition_names()

    def callback_transform_current_frame(self, direction=None):
        if direction is not None:
            self.frame_image_view.setCurrentIndex(self.frame_image_view.currentIndex + direction*2)
        # Get index of condition shown
        condition_idx = self.frames_idx_array[self.frame_image_view.currentIndex]
        # Check whether this condition need to be transformed
        T = self.TRANSFORMATION_IDX[self.TRANSFORMATION_IDX[:, 0] == condition_idx, 1][0]
        if len(T) > 0:
            # Get image currently shown
            frame = self.frame_image_view.image[self.frame_image_view.currentIndex, :, :]

            # Apply transformations one after the other
            for t_idx in T:
                # Get matrices
                scale = self.TRANSFORMATION_MATRICES.loc[t_idx, 'scale']
                rotation = self.TRANSFORMATION_MATRICES.loc[t_idx, 'rotation']
                translation = self.TRANSFORMATION_MATRICES.loc[t_idx, 'translation']
                # Apply them
                frame = transform_image_from_matrices(frame, scale, rotation, translation)

            # Update image
            self.frame_image_view.imageItem.updateImage(frame, autoHistogramRange=self.auto_adjust_histograms)

    def get_FOV_translation_state(self):
        roi = self.translation_ROI
        # Read ROI state (scale and rotation)
        state = roi.state
        scale = np.array([state['size'].x() / self.PARAMETERS['frame_width'], state['size'].y() / self.PARAMETERS['frame_height']])
        angle = state['angle']
        # Calculate the offset from relative position of handles
        handles = roi.getHandles()[1:]  # exclude the first one because it is the rotation handle
        x = np.array([h.viewPos().x() for h in handles])
        y = np.array([h.viewPos().y() for h in handles])
        center_ROI = [x.mean(), y.mean()]
        offset = np.array([center_ROI[0] - self.PARAMETERS['frame_width'] / 2., center_ROI[1] - self.PARAMETERS['frame_height'] / 2.])

        return scale, angle, offset

    def get_translated_trace_segments(self, conditions_idx, ROI_pixels, from_temporary):
        # Expand indices of conditions
        condition_duration = self.frames_idx[conditions_idx, 1] - (self.frames_idx[conditions_idx, 0] - 1)
        modified_frames_idx = expand_indices(self.frames_idx[conditions_idx, 0] - 1, self.frames_idx[conditions_idx, 1])
        # Allocate variable to keep new luminescence values
        L_modified = np.nan * np.zeros(shape=(ROI_pixels.shape[0], modified_frames_idx.shape[0]))
        # Get new values for each pixel
        for idx, pixel in enumerate(ROI_pixels):
            if from_temporary:
                new_pixel = self.temporary_MAPPING_TABLE[pixel]
            else:
                new_pixel = self.MAPPING_TABLE[conditions_idx, pixel]
            # Mark which pixels cannot be tracked along all frames
            bad_pixels = np.isnan(new_pixel)
            # If all pixels are bad, skip this pixel
            if np.all(bad_pixels):
                continue
            # Temporarily mark bad pixels with -1
            new_pixel[bad_pixels] = -1
            # Convert to integer
            new_pixel = new_pixel.astype(int)
            # Unfold pixel value in time
            if from_temporary:
                pixel_time = np.ones_like(modified_frames_idx) * new_pixel
            else:
                pixel_time = np.hstack([np.ones(condition_duration[ii], dtype=int) * new_pixel[ii] for ii in range(len(new_pixel))])
            # Get luminescence values
            L = self.data_time[pixel_time, modified_frames_idx]
            if not from_temporary:
                # Restore NaNs in signal
                L = L.copy()
                L[pixel_time == -1] = np.nan
            L_modified[idx, :] = L

        # Compute the mean across pixels
        if self.current_operation_type == 'mean':
            L_modified = np.nanmean(L_modified, axis=0)
        elif self.current_operation_type == 'max':
            L_modified = np.nanmax(L_modified, axis=0)

        return L_modified, modified_frames_idx

    def update_table_condition_names(self):
        # Get indices of modified frames
        for frame_idx in xrange(self.n_conditions):
            # Get name of condition
            cond_name = self.PARAMETERS['condition_names'][frame_idx]
            # Get number of modifications on this frame
            n_changes = len(self.TRANSFORMATION_IDX[frame_idx, 1])
            # Make suffix to condition name
            if n_changes > 0:
                cond_name += ' ' + '*' * n_changes
            # Update name of condition on table
            self.table_conditions.item(frame_idx, 0).setText(cond_name)


    ############################################################################
    # Mouse and keyboard interaction
    ############################################################################
    def callback_mouse_moved(self, event):
        if self.current_action == 'drawing_ROI' or self.toggling_ROI_visibility:
            pos = QtCore.QPoint(event.x(), event.y())
            if self.average_image_viewbox.rect().contains(pos):
                mousePoint = self.average_image_viewbox.mapSceneToView(pos)
                self.crosshair[0].setPos(mousePoint.x())
                self.crosshair[1].setPos(mousePoint.y())

    def callback_mouse_clicked(self, event):
        # Check whether user is drawing an ROI
        if self.current_action == 'drawing_ROI':
            # Get click position and store it in memory
            pos = event.pos()
            mousePoint = self.average_image_viewbox.mapSceneToView(pos)
            mousePoint = np.array([mousePoint.x(), mousePoint.y()])
            # Ignore click if out of bounds
            if np.any(mousePoint<0) or mousePoint[0]>self.PARAMETERS['frame_width'] or mousePoint[1]>self.PARAMETERS['frame_height']:
                event.ignore()
                return
            else:
                event.accept()

            # If user pressed the left mouse button
            if event.buttons() == QtCore.Qt.LeftButton:
                # Store data in memory
                self.last_ROI_coordinates.append(mousePoint)

                # Initialize plot item if not present
                if self.last_ROI_coordinates_handle is None:
                    brush = pg.mkBrush(self.next_ROI_color)
                    self.last_ROI_coordinates_brushes.append(brush)
                    self.last_ROI_coordinates_handle = pg.PlotDataItem(x=[mousePoint[0]], y=[mousePoint[1]], symbol='o', symbolBrush=brush, symbolSize=10, pxMode=True)
                    self.average_image_viewbox.addItem(self.last_ROI_coordinates_handle)

                else:  # Add point
                    # Set whether new point should have a different color
                    if self.selected_ROI_type == 'fill':
                        self.get_next_ROI_color()
                        brush = pg.mkBrush(self.next_ROI_color)
                        self.last_ROI_coordinates_brushes.append(brush)
                        self.last_ROI_coordinates_handle.opts['symbolBrush'] = self.last_ROI_coordinates_brushes

                    # Update data
                    self.last_ROI_coordinates_handle.setData(np.array(self.last_ROI_coordinates))

                    # Set whether line connecting points should be visible
                    if self.selected_ROI_type == 'fill':
                        self.last_ROI_coordinates_handle.curve.hide()
                    else:
                        self.last_ROI_coordinates_handle.curve.show()

                # Check whether it's possible to activate the button to make the ROI
                n_points = len(self.last_ROI_coordinates)
                if (self.selected_ROI_type == 'polygon' and n_points >= 3) or (self.selected_ROI_type == 'fill' and n_points >= 1):
                    self.buttons.draw_ROI.setStyleSheet(str('background-color: rgb%s' % str(tuple(np.array(GUI_default['icon_background']['go'], dtype=float)*255))))
                else:
                    self.buttons.draw_ROI.setStyleSheet(str('background-color: rgb%s' % str(tuple(np.array(GUI_default['icon_background']['no_go'], dtype=float) * 255))))

                # Automatically accept drawing the ROI
                if self.selected_ROI_type == 'fill' and n_points >= 1:
                    self.callback_draw_ROI()

    def callback_timeline_frames(self, from_plot):
        if not self.updating_timeline_position:
            # Set internal flag
            self.updating_timeline_position = True
            # Get position of timeline
            if from_plot == 'frame':
                ind, time = self.frame_image_view.timeIndex(self.frame_image_view.timeLine)
            elif from_plot == 'trace':
                time = self.trace_time_mark.getXPos()
                ind = int(np.round(time))
            # Abort if timeline has been pulled outside range
            if ind < 0 or ind >= self.PARAMETERS['n_frames']:
                self.updating_timeline_position = False
                return

            # Set position of other timeline
            if from_plot == 'frame':
                self.trace_time_mark.setValue(time)
            elif from_plot == 'trace':
                self.frame_image_view.timeLine.setValue(ind)
            # Update image
            self.frame_image_view.currentIndex = ind
            self.frame_image_view.updateImage()
            self.callback_transform_current_frame()
            # Update title
            self.image_views_title[1].setPlainText(self.PARAMETERS['condition_names'][self.frames_idx_array[ind]])
            # Update visible range
            if not self.histograms_linked and self.auto_adjust_histograms:
                frame = self.frame_image_view.image[self.frame_image_view.currentIndex, :, :].ravel()
                levels = np.percentile(frame[~np.isnan(frame)], q=[1, 99])
                self.frame_image_view.getHistogramWidget().item.setLevels(levels[0], levels[1])
                self.frame_image_view.getHistogramWidget().item.autoHistogramRange()

            # Update other view
            if self.views_linked:
                condition = self.frames_idx_array[ind]
                # Check whether we have to update the image
                if len(self.current_average_frame) == 1 and self.current_average_frame[0] == condition:
                    self.updating_timeline_position = False
                    return
                self.current_average_frame = [condition]
                self.image_views_title[0].setPlainText(self.PARAMETERS['condition_names'][self.frames_idx_array[self.frames_idx[self.current_average_frame[0], 0]]])
                # Update image
                self.compute_frames_projection()
                # Update image of ROI
                if self.selected_ROI is not None:
                    self.callback_update_zoom_ROI(self.selected_ROI, update_trace=False, update_ROI_contour=False)
                # Update selection in table
                self.table_conditions.selectRow(condition)
                # Highlight frames in this condition
                self.make_region(self.frames_idx[self.current_average_frame, :])

            # Limit range in trace viewbox
            if self.trace_anchored and from_plot == 'frame':
                # Get selected frames
                selected_frames = self.frames_idx[self.current_average_frame, :].ravel()
                selected_frames_range = [selected_frames.min(), selected_frames.max()]
                self.trace_viewbox.setRange(xRange=selected_frames_range, padding=0, disableAutoRange=False)

            # Reset internal flag
            self.updating_timeline_position = False

    def callback_reset_zoom(self, where):
        if not hasattr(where, '__iter__'):
            where = [where]
        if len(where) == 1 and where[0] == 'all':
            where = ['average','frame','histograms','ROI','trace']

        # Condition selection
        if 'average' in where:
            self.average_image_viewbox.autoRange()
        # Frame selection
        if 'frame' in where:
            self.frame_image_view.view.autoRange()
        # Histograms
        if 'histograms' in where and self.auto_adjust_histograms:
            # Average image
            image = self.average_image_view.image
            image = image[~np.isnan(image)]
            levels = np.percentile(image.ravel(), q=[1, 99])
            self.average_histogram.item.setLevels(levels[0], levels[1])
            self.average_histogram.item.autoHistogramRange()
            # Single-frame
            timeline = self.frame_image_view.timeLine
            ind, _ = self.frame_image_view.timeIndex(timeline)
            image = self.frame_image_view.image[ind, :, :]
            image = image[~np.isnan(image)]
            levels = np.percentile(image.ravel(), q=[1, 99])
            self.frame_image_view.getHistogramWidget().item.setLevels(levels[0], levels[1])
            self.frame_image_view.getHistogramWidget().item.autoHistogramRange()
        # ROI inspection
        if 'ROI' in where:
            self.ROI_viewbox.autoRange()
        # Trace
        if 'trace' in where:
            self.trace_viewbox.autoRange()
            self.trace_viewbox.enableAutoRange(x=False, y=True)
            self.trace_viewbox.setAutoVisible(x=False, y=True)

    def callback_table_selection(self, what):
        if self.table_conditions.allowed_interaction and not self.updating_timeline_position:
            # This function should not get triggered recursively
            self.table_sessions.allowed_interaction = False
            self.table_conditions.allowed_interaction = False

            # Get selected items
            if what == 'session':
                items = self.table_sessions.selectedItems()
            elif what == 'condition':
                items = self.table_conditions.selectedItems()
            # Get index of this elements
            conditions = [ii.row() for ii in items]
            if len(conditions) < 1:
                return

            # If user selected an entire session, find condition frames corresponding to it
            if what == 'session':
                selected_sessions = list(conditions)
                conditions = np.where(np.in1d(self.sessions_idx, conditions))[0]

            # Adapt other table
            if what == 'session':  # Select all conditions in this session
                self.table_conditions.clearSelection()
                previous_selection_mode = self.table_conditions.selectionMode()
                self.table_conditions.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
                [self.table_conditions.selectRow(cond) for cond in conditions]
                self.table_conditions.setSelectionMode(previous_selection_mode)
            elif what == 'condition':  # Deselect all sessions for clarity
                self.table_sessions.clearSelection()

            # Store range in memory
            self.current_average_frame = sorted(conditions)

            # Highlight frames
            if len(conditions) == 1:
                self.make_region(self.frames_idx[self.current_average_frame, :])
            else:
                # Highlight frames in timeline
                frames_idx = np.hstack([np.arange(start=self.frames_idx[cond, 0], stop=self.frames_idx[cond, 1]+1) for cond in conditions])
                regions = idx2range(frames_idx)[:, :2]
                self.make_region(regions)

            # Update title
            title = ''
            if what == 'session':
                if len(selected_sessions) == 1:
                    title = 'session %i' % (selected_sessions[0] + 1)
            elif what == 'condition':
                if len(conditions) == 1:
                    title = self.PARAMETERS['condition_names'][self.frames_idx_array[self.frames_idx[self.current_average_frame[0], 0]]]
            self.image_views_title[0].setPlainText(title)

            # Update image
            self.compute_frames_projection()

            # Update image of ROI
            if self.selected_ROI is not None:
                self.callback_update_zoom_ROI(self.selected_ROI, update_trace=False, update_ROI_contour=False)

            # Get selected frames
            selected_frames = self.frames_idx[self.current_average_frame, :].ravel()
            selected_frames_range = [selected_frames.min(), selected_frames.max()]
            # Limit range in trace viewbox
            if self.trace_anchored:
                self.trace_viewbox.setRange(xRange=selected_frames_range, padding=0, disableAutoRange=False)
                # Show stimulus profile, if any
                if self.is_stimulus_evoked:
                    # Delete previous stimulus profiles
                    if self.stimulus_profile is not None:
                        self.trace_viewbox.removeItem(self.stimulus_profile)
                    # Get number of traces currently shown
                    n_shown_traces = np.where(self.ROI_TABLE['show_trace'] == True)[0].shape[0]
                    # Show stimulus only if one trial selected and at least one trace shown
                    if len(self.current_average_frame) == 1 and n_shown_traces >= 1:
                        # Get index of stimulus presented (and respect python's 0-indexing)
                        stimulus_idx = self.PARAMETERS['condition_has_stimulus'][self.current_average_frame[0]] - 1
                        stimulus_profile = self.PARAMETERS['stimuli'][stimulus_idx, 1]
                        # Make sure visible range starts from 0
                        visible_range = list()
                        visible_traces = self.ROI_TABLE.loc[self.ROI_TABLE['show_trace'] == True, 'handle_trace'].values
                        for t in visible_traces:
                            visible_range.append(t.yData[selected_frames_range[0]:selected_frames_range[1]].max())
                        # Make sure bottom is 0
                        visible_range = [0, max(visible_range)]
                        self.trace_viewbox.setRange(yRange=visible_range, disableAutoRange=False)
                        # Scale stimulus profile to span entire visible range
                        stimulus_profile = stimulus_profile * visible_range[1]
                        # Draw new stimulus profile
                        self.stimulus_profile = pg.PlotCurveItem(clickable=False)
                        self.stimulus_profile.setData(x=np.arange(selected_frames_range[0], selected_frames_range[1]+1), y=stimulus_profile)
                        self.trace_viewbox.addItem(self.stimulus_profile)
                        self.stimulus_profile.setPen(GUI_default['stimulus_profile_pen'][self.current_colormap])
                        self.stimulus_profile.setZValue(0)

            # Update timelines
            if self.views_linked:
                # Toggle internal flag
                self.updating_timeline_position = True
                # Update trace view
                self.trace_time_mark.setValue(selected_frames_range[0])
                self.frame_image_view.timeLine.setValue(selected_frames_range[0])
                # Update image and title
                self.callback_transform_current_frame()
                self.image_views_title[1].setPlainText(self.PARAMETERS['condition_names'][self.frames_idx_array[selected_frames_range[0]]])
                # Update visible range
                if self.auto_adjust_histograms:
                    frame = self.frame_image_view.image[self.frame_image_view.currentIndex, :, :].ravel()
                    levels = np.percentile(frame[~np.isnan(frame)], q=[1, 99])
                    self.frame_image_view.getHistogramWidget().item.setLevels(levels[0], levels[1])
                    self.frame_image_view.getHistogramWidget().item.autoHistogramRange()
                # Toggle internal flag
                self.updating_timeline_position = False

            # Re-enable interaction
            self.table_sessions.allowed_interaction = True
            self.table_conditions.allowed_interaction = True

    def callback_scroll_average_frame(self, where):
        new_frame = 0
        # Get index of last selected frame
        old_frame = self.current_average_frame[-1]
        # Update index
        if where == 'up':
            new_frame = max(0, old_frame - 1)
        elif where == 'down':
            new_frame = min(self.n_conditions, old_frame + 1)
        # Update selection in table
        self.table_conditions.selectRow(new_frame)

    def callback_update_image_while_translating_FOV(self, roi):
        # Get scaling factor
        state = roi.state
        scale = np.array([state['size'].x() / self.PARAMETERS['frame_width'], state['size'].y() / self.PARAMETERS['frame_height']])
        # Get image and scale it
        if np.any(scale != 1):
            rect = roi.boundingRect()
            self.average_image_view.setRect(rect)
        # Read translation_ROI state
        scale, angle, offset = self.get_FOV_translation_state()
        # Initialize position of each pixel and transform this grid
        pixel_list = np.arange(self.PARAMETERS['n_pixels'], dtype=int)
        pixel_list = pixel_list.reshape(self.PARAMETERS['frame_height'], self.PARAMETERS['frame_width']).transpose().ravel()
        pixel_grid = np.reshape(pixel_list, (self.PARAMETERS['frame_width'], self.PARAMETERS['frame_height']))
        pixel_grid = transform_image_from_parameters(pixel_grid, scale, -angle, offset[::-1])
        # Round pixel value to nearest integer
        pixel_grid = np.round(pixel_grid)
        # Replace original pixel positions with these ones
        self.temporary_MAPPING_TABLE = pixel_grid.transpose().ravel()

        # Update GUI data
        table_rows = self.ROI_TABLE['show_trace'].index.values
        for roi in self.ROI_TABLE.loc[table_rows, 'handle_ROI'].tolist():
            self.callback_update_zoom_ROI(roi, update_trace=True, update_ROI_contour=False)


################################################################################
# Direct call
################################################################################
if __name__ == '__main__':
    # Get user inputs
    if len(sys.argv) <= 1:
        # pass
        params_filename = ''
        do_only_conversion = False
    else:
        params_filename = sys.argv[1]
        do_only_conversion = sys.argv[2]
        if do_only_conversion == 'True':
            do_only_conversion = True
        elif do_only_conversion == 'False':
            do_only_conversion = False
        else:
            do_only_conversion = False

    # Load parameters
    PARAMETERS = matlab_file.load(params_filename, variable_names='PARAMETERS')
    # Check that files used by the GUI exist
    if not os.path.exists(PARAMETERS['filename_data_frame']) or not os.path.exists(PARAMETERS['filename_data_time']):
        PARAMETERS = prepare_data_for_GUI(PARAMETERS)

    # Run GUI only if user requested so
    if not do_only_conversion:
        # Make sure some fields are iterables
        if not hasattr(PARAMETERS['sessions_last_frame'], '__iter__'):
            PARAMETERS['sessions_last_frame'] = np.array([PARAMETERS['sessions_last_frame']], dtype=np.int64)

        # Initialize Qt-event loop
        QtApp = Qt5_QtApp(appID=u'calcium_imaging_data_analysis')
        QtApp.app.setQuitOnLastWindowClosed(False)
        # Run GUI
        GUI = Calcium_imaging_data_explorer(PARAMETERS, QtApp)
        # Start Qt-event loop
        sys.exit(QtApp.app.exec_())
