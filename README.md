# cvgui-linux: Yet Another Image Analysis Platform for linux
v4l2-ctl and OpenCV based camera in Tkinter GUI platform.

## License
Planning to be distributed by Apache 2.0 (**advice wanted**)

Note: This program is dependent to v4l-utils (v4l2-ctl), which is protected by GNU GPL v2.0 license.
However, we only call v4l2-ctl by subprocess command and do not include the program itself as a library nor module.
So this repository should not need to inherit GPL. Nonetheless, if one recognizes this program as a specialized frontend 
of v4l2-ctl, there seems to be room for discussion.

## Licenses of Core module components to be considered

| Name | LICENSE  | Link | Note |
|-------------|----------|------|----|
| tkinter     |　PSF  | https://docs.python.org/3/license.html ||
| Tk (Tcl/Tk)     |　BSD-type?  |   https://www.tcl.tk/software/tcltk/license.html | | |
| v4l-utils    | GNU GPL v2.0　  |   https://github.com/cz172638/v4l-utils/blob/master/COPYING | Only calls the app via subprocess |
| OpenCV     |　Apache 2  | https://opencv.org/license/ | Version dependency. <4.4 is BSD-3-Clause|
|scikit-image|BSD-3-Clause/BSD-2-Clause|https://github.com/scikit-image/scikit-image/blob/main/LICENSE.txt|Depends on which module to use|

## Licenses of Ext module components

Some GPL dependent image analysis modules are detached from the core component to keep the repository GPL free.

| Name | LICENSE  | Link | Note |
|-------------|----------|------|----|
| iSeed Counter 640x640 Yolov5 onnx     |　GPLv3  | https://github.com/ultralytics/yolov5/blob/master/LICENSE | inherits yolov5 license|

## Requirements

t.b.d.

```commandline
sudo apt-get install python3-tk python3-pil.imagetk
pip install numpy opencv-python scipy scikit-image matplotlib
pip install onnxruntime
```

## Installation

Install only the core module components
```
t.b.d
```
OR Install with ext module (GPL)
```
t.b.d. git clone --recursive XXXXX
```
OR Install ext module (GPL) afterwards
```
t.b.d.
cd PATH/TO/THE/REPO
git submodule update --init --recursive
```

## Run

```
python main.py
```

## Image Acquisition

- set_acquire_config
- if not self.config_q.full():
  self.config_q.put(self.acquire_config)

## Commercially Available UVC-camera compatibility
| Name | Link | Note |
|-------------|------|------|
| t.b.d.    | t.b.d. | t.b.d. |

## Known Issues
torch.hub models seems to not work under multiprocess

## To Do Lists

- manual instructions (organize dependencies)
- interval image acquisition mode

