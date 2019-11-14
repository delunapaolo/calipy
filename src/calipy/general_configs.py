# Graphical packages
from PyQt5 import QtGui
from third_party import pyqtgraph as pg

# Local repository
from calipy.utils.icons import histogram_locked, histogram_unlocked, trace_locked, trace_unlocked, colormap, view_unlocked, view_locked, magnifying_glass, bucket, move, polygon


################################################################################
# This dictionary contains the default values for graphical elements.
default = dict()

# Pre-processing options
default['frame_rate'] = 30
default['correlation_time_smoothing_window'] = 1

# GUI options
default['colormap_index'] = 1
default['all_projection_types'] = ['max', 'mean', 'median', 'standard deviation']
default['projection_type'] = 'max'
default['average_type'] = 'correlation'
default['max_n_visible_ROI_traces'] = 0

default['ROI_linewidth_thick'] = 2
default['text_color_dark'] = 'w'

# Font
default['font'] = QtGui.QFont()
default['font'].setFamily('Lucida')
default['font'].setPointSize(12)
default['font'].setBold(False)

default['font_buttons'] = QtGui.QFont()
default['font_buttons'].setFamily('Lucida')
default['font_buttons'].setPointSize(14)
default['font_buttons'].setBold(False)

default['font_editors'] = QtGui.QFont()
default['font_editors'].setFamily('Lucida')
default['font_editors'].setPointSize(14)
default['font_editors'].setBold(False)

default['font_titles'] = QtGui.QFont()
default['font_titles'].setFamily('Lucida')
default['font_titles'].setPointSize(16)
default['font_titles'].setBold(True)

default['font_actions'] = QtGui.QFont()
default['font_actions'].setFamily('Lucida')
default['font_actions'].setPointSize(14)
default['font_actions'].setBold(False)

default['min_button_height'] = 50
default['max_button_width'] = 500


default['ROI_type'] = 'polygon'
default['flood_tolerance'] = 10
default['max_n_points'] = 15
default['max_n_diameter'] = 50
default['max_area_percent'] = 50
default['icons'] = dict()
default['icons']['toggle_colormap'] = colormap
default['icons']['link_views'] = (view_unlocked, view_locked)
default['icons']['visibility_ROI'] = u'\U0001F441'
default['icons']['zoom'] = magnifying_glass
default['icons']['ROI_fill'] = bucket
default['icons']['ROI_polygon'] = polygon
default['icons']['link_histograms'] = (histogram_unlocked, histogram_locked)
default['icons']['translation_ROI'] = (move, move)
default['icons']['anchor_trace'] = (trace_unlocked, trace_locked)

default['icons_kind'] = dict()
default['icons_kind']['toggle_colormap'] = 'icon'
default['icons_kind']['visibility_ROI'] = 'text'
default['icons_kind']['zoom'] = 'icon'
default['icons_kind']['ROI_fill'] = 'icon'
default['icons_kind']['ROI_polygon'] = 'icon'
default['icons_kind']['link_histograms'] = 'icon'
default['icons_kind']['translation_ROI'] = 'icon'
default['icons_kind']['link_views'] = 'icon'
default['icons_kind']['anchor_trace'] = 'icon'

default['icon_background'] = dict()
default['icon_background']['go'] = (.5, 1., 0.)
default['icon_background']['no_go'] = (1., .39, .28)


# Make brush for selection regions
default['region_colors_dark'] = pg.mkBrush(color=(255, 99, 71))
default['region_colors_light'] = pg.mkBrush(color=tuple([int(255 * .9)] * 3))
# Make pen for stimulus profile, if any
default['stimulus_profile_pen'] = [pg.mkPen((255, 255, 255)), pg.mkPen((0, 0, 0))]

default['ROI_handle_pen'] = pg.mkPen(None)
default['ROI_highlighted_segment_pen'] = pg.mkPen((255, 0, 0), width=4)
default['ROI_handle_size'] = 7
