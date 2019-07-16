# calipy

calipy is a GUI to explore 3D stacks of images, as in x $\times$ y $\times$ time of live microscopy, or x $\times$ y $\times$ z in multi-plane imaging. It has been designed for inspecting larger-than-memory files interactively in real-time.


## TL;DR

calipy is a python software which comprises:
- A module to convert images to binary files, for real-time exploration and memory-meapping of larger-than-memory image stacks;
- A GUI with region-of-interest (ROI) selection and average trace inspection.

Jump to [demo](#live_demonstration) below.


# Live microscopy imaging


## The problem

Image stacks microscopy generates large tiff files, which are usually opened and analyzed in [Fiji](https://fiji.sc/) or [ImageJ](https://imagej.nih.gov/ij/), where many plugins exist for these purposes. However, there doesn't seem to exist yet a plugin to visualize the average value of pixels or regions-of-interest (ROIs) along a third dimension. Also, it would be useful to align files (or microscopy sessions) to inspect the same ROIs over several files.


### Enter calipy

The proposed software offers a two-fold solution:

- A converter, which rewrites files in a binary format and concatenates all files together, so they can be analyzed as a whole;
- A GUI with commands to select ROIs manually or via the watershed method. 

It is implemented in python to leverage on two powerful libraries: [numpy](https://www.numpy.org/) and [pyqtgraph](http://pyqtgraph.org/).


## 1. The converter

First, tiff files are concatenated and converted to binary files with the function
    
    calipy.utils.IO_operations.prepare_data_for_GUI()
    
Four files are generated as result: 

- `stack_frame_first.dat` contains all data, written in a standard format.
- `stack_time_first.dat` contains all data, written in a time-first order (see below). It is this file that allows visualization and inspection of the trace in real-time.
- `parameters.json.txt` contains information regarding the current analysis session. An example is the following:

        {
        "dataset_ID": "demo_calipy", 
        "n_frames": 134, 
        "frame_height": 480, 
        "frame_width": 500, 
        "n_pixels": 240000, 
        "dtype": "float64", 
        "frames_idx": [[0, 134]], 
        "sessions_last_frame": [134], 
        "condition_names": ["0104"], 
        "filename_frame_first": "demo_calipy/stack_frame_first.dat", 
        "filename_time_first": "demo_calipy/stack_time_first.dat",
        "filename_projections": "demo_calipy/projections.hdf5",
        "filename_ROIs": "demo_calipy/ROIs_info.mat"
        }

- `projections.hdf5` contains the following "averages" of a small number of images:
    - mean
    - median
    - standard deviation
    - cross-correlation
    - maximum


### Why a converter?

Because reading from a binary file is very fast with numpy. However, all files are written in a "flat" format. 

<div style="text-align:center"><img src="resources/gifs_introduction/1. Reading an image file.gif"/></div>

Usually, pixels that are closer in space are also stored closer on disk. I call this arrangement a "frame-first" stack, because it gives priority to frames, that is, it is easier to read an entire frame ...

<div style="text-align:center"><img src="resources/gifs_introduction/2. Reading a single frame from a frame-first image stack.gif"/></div>

... than reading a single location across frames. 

<div style="text-align:center"><img src="resources/gifs_introduction/3. Reading along time from a frame-first image stack.gif"/></div>


### What kind of conversion?

calipy performs two types of conversion: It formats the data into binary format for faster disk access with numpy, and it stores the data into a different format, what I call a "time-first" image stack, which allows faster reading along the third dimension because it stores pixels along that dimension closer on disk.

<div style="text-align:center"><img src="resources/gifs_introduction/4. Reading along time from a time-first image stack.gif"/></div>


### Pros of file conversion

- Interactive exploration of traces
_ Speed on large-scale imaging datasets, because it overcomes bottleneck of “seek” time, which averages 4–15 ms for HDDs and 0.08–0.16 ms for SSDs.


### Cons of file conversion

+ Double the amount of space (i.e., same data, but different arrangement)
+ Conversion to “time-first” arrangement might be too time-consuming in some cases


### Benchamrks

| Reading and<br>averaging over | “time-first” | “frame-first” | ratio<br>("frame" / "time") |
|-----------------------:|-------------:|--------------:|:---------------------:|
| 1 pixel | < 1 ms | .058 s | $\infty$ |
| Small ROI (40 pixels) | .022 s | .303 s | 14$\times$ |
| 40 random pixels | .037 s | 153.612 s | 4152$\times$ |
| Large ROI (300 pixels) | .206 s | 27.168 s | 132$\times$ |
| 300 random pixels | .253 s | 4780.582 s | 18896$\times$ |


## 2. The GUI


### Concept

calipy offers a GUI to interactively explore the average pixel value of ROIs along the third dimension of the stack, for example, fluorescence traces in the case of live calcium-imaging microscopy. 
The GUI has four panels and a table:

<div style="text-align:center"><img src="resources/concept/GUI look and feel.png"/></div>

+ `Average frame selector` is made of two tables which allow the selection of a whole group of imaging stacks (the "sessions") or of individual imaging files (the "trials").
+ `Average frame viewer` shows one of the 5 averages corresponding to the selected stacks. On the right hand side, a `Histogram` allows real-time adjustment of the image contrast and brightness.
+ `Single-frame viewer` shows individual frames of the current image stack. It comes with its own `Histogram`.
+ `Single-frame selector` shows a scroll marker which can be dragged over the entire duration of the concatenated files, with no delay. A red rectangle highlights all the frames, whose average is displayed in the `Average frame viewer`.
+ `ROI viewer` magnifies the last selected ROI. The borders of this panel will change according to the color of the ROI, as selected in the `Average frame viewer`.
+ `Trace viewer` shows the mean or the maximum pixel value in the selected ROIs along the third dimension of the stack, for example, time. A vertical cursor marks the same position as in the `Single-frame selector`, and a gray region will mark the frames selected in the `Average frame viewer`.

The GUI is based on pyqtgraph, a very powerful plotting library, which offers many advanced capabilities, as shown in the next section.


### Live demonstration

<ol start="1"><li>Real-time scrolling (~55 GB file, ~200k frames)</ol>


The following gif shows how scrolling through a larger-than-memory file happens in real-time, with no delays. Also, notice how:

+ the histogram on the right of the `Single-frame viewer` quicly updates to the range of the current frame. 
+ the cursor in the `Trace viewer` follows the position marked in the `Single-frame selector`.

<div style="text-align:center"><img src="resources/gifs_demo/1. Real-time scrolling.gif"/></div>


<ol start="2"><li>Select many frames at once + adjusting histograms</li></ol>

In the next gif, I show how selecting a session (which is a compilation of imaging stacks) also happens in real-time. Also, I show how the histogram works, by changing contrast in session 1, 2 and 3, and changing brightness in at the end of session 3.

<div style="text-align:center"><img src="resources/gifs_demo/2. Select sessions.gif"/></div>


<ol start="3"><li>Segmentation by manual selection with lasso
</li></ol>

ROIs can be selected by drawing a polygon and pressing Enter to accept the selection. Notice how, after completing an ROI:

+ The trace immediately appears in the `Trace viewer`.
+ The ROI is magnified in the `ROI viewer`. 

<div style="text-align:center"><img src="resources/gifs_demo/3. Polygon selector.gif"/></div>

This tool is really powerful, because each vertex of the polygon area can be removed or dragged, or new vertices can be added, or the entire ROI can be shifted to a different location. Please note that every change to the ROI will cause a change in the shown trace.

Last aspect to notice is that the ROI is also drawn in the `Single-frame selector` with a dashed yellow contour. It is very subtle, and not really visible in the gif above, but it would become more visible if frames area dragged.


<ol start="4"><li>Segmentation by area selection by watershed</li></ol>

Also known as "flood tool" or "magic wand" in graphic software, calipy provides a simple implementation of the watershed algorithm, which selects an area in the proximity of where the user clicked, based on the value of the clicked pixel. The threshold for computing how similar a pixel is to the reference pixel can be adjusted with the slider above the `Average frame viewer`. In the example below, it was set to 2.5.

<div style="text-align:center"><img src="resources/gifs_demo/4. Flood tool.gif"/></div>

Please also notice how the trace appers immediately after a click.


<ol start="5"><li>Field-of-view transformations: translation, rotation, resize</li></ol>

When different imaging sessions are repeated after some time or performed on live tissue, the stacks could be misaligned, which will negatively affect what the ROI will contain along the third dimension of the stack. There are 3 types of distortions that can occur: rigid shifts (translations), rotation around a point, and shrinking of a dimension.

For these reasons, calipy implements a tool to perform realignment of frames. Once activated, the tool will allow the user to translate, rotate, or resize the currently selected frames, and store the new position. At the end of the process, the ROI traces will be updated with the new values at the adjusted coordinates.

<div style="text-align:center"><img src="resources/gifs_demo/5. Move FOV.gif"/></div>

Although the image seems to be cropped at the end of the process, it is only hidden, and can be restored with an appropriate command in the menu bar.


### Advanced commands

calipy's GUI alos offers more advanced commands to:

+ Synchronize the histograms of both viewer panels so that the same range is shown
+ Synchronize the `Trace viewer` with the frame selectors, to display only the trace of the selected frames
+ Toggle the visibility of one or more ROIs, to declutter the interface
+ Switch to "night mode", where higher pixel values appear white and the blackground of the GUI is black.
+ Save and load progress from disk
+ Superimpose known events over the `Trace viewer` to interpret data in real-time.


# Installation

Run the file `/setup/install_anaconda_environments.py` in a terminal, which will install an Anaconda virtual environment named `calipy`, and install all the necessary packages to allow you to use the software. 

After that, activate the environment and point `python` to `/src/calipy/_run.py`.
    
The file `/src/calipy/general_configs.py` contains a list of parameters, which are exposed to the user for easy default setting.

## Dependencies

pyqtgraph is under active development. Here, I included the version `0.11.0.dev0`, on which I tested calipy. This version is also included in the .yml file, so it will be installed via pip. However, for the moment calipy imports the version included in this repo.

`tifffile.py`, version `2019.03.18`, which is part of the library [scikit-image](https://github.com/scikit-image/scikit-image/blob/master/skimage/external/tifffile/tifffile.py) is also part of the repo, to reduce third-party dependencies.

