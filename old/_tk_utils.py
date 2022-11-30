import copy
import cv2
import os
import time
import subprocess
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox

from PIL import Image, ImageTk
import numpy as np

from utility.general_utils import FPS
from utility.v4l2_utils import V4L2Control, RetreiveV4L2Info
from utility.focus_utils import focus_stack
import sys
sys.path.append("../")
from image_analysis_modules.yolox import yolox_onnx
from image_analysis_modules.yolox.yolox_onnx import _COLORS as _COLORS


from matplotlib.colors import rgb2hex
import gc

with open('image_analysis_modules/yolox/coco_classes.txt', 'rt') as f:
    coco_classes = f.read().rstrip('\n').split('\n')


class CameraControl(tk.Frame):
    def __init__(self, master,
                 io_name=None,
                 focus_only=True):
        super().__init__(master)

        self.v4l2control: V4L2Control
        self.ratio: float
        self.camera_config: dict

        self.io_name = io_name
        self.focus_only = focus_only

        self.master.title("Camera Control")

        self.width = self.master.winfo_screenwidth()//2
        self.height = self.master.winfo_screenheight()

        # self.master.geometry("{}x{}+0+0".format(self.width, self.height))
        self.master.columnconfigure(0,weight=1, uniform='a')
        self.master.columnconfigure(1,weight=1, uniform='a')
        self.master.protocol("WM_DELETE_WINDOW", self.hide)



        self.camera_params_component = {}
        self.initialize_camera_control_frame()

        #hide
        self.master.withdraw()

    def hide(self):
        self.master.withdraw()

    def initialize_camera_control_frame(self):

        self.camera_config = RetreiveV4L2Info().config
        self.v4l2control = V4L2Control(self.io_name)

        for widget in self.master.winfo_children():
            widget.destroy()

        n = 0

        for i, (key, val) in enumerate(self.camera_config["parameters"][self.io_name].items()):
            if self.focus_only and "focus" not in key.lower():
                continue

            row = n // 2
            if n % 2 == 0:
                column = 0
            else:
                column = 1

            sticky = "nsew"

            if val["type"] == "int":
                self.camera_params_component[key] = tk.Scale(
                    self.master,
                    label=key + " (default:{})".format(str(val["default"])),
                    orient="h",
                    from_=int(val["min"]) // int(val["step"]),
                    to=int(val["max"]) // int(val["step"]),
                )
                self.camera_params_component[key].set(int(val["value"]) // int(val["step"]))
                self.camera_params_component[key]["command"] = self.v4l2control.parse_int(key=key,step=val["step"])

            elif val["type"] == "bool":

                _var_name = key + "var"
                self.camera_params_component[_var_name] = tk.BooleanVar()
                if int(val["value"]):
                    self.camera_params_component[_var_name].set(1)
                else:
                    # self.camera_params_component[key].deselect()
                    self.camera_params_component[_var_name].set(0)

                self.camera_params_component[key] = tk.Checkbutton(
                    self.master,
                    text=key,
                    variable=self.camera_params_component[_var_name]
                )
                self.camera_params_component[key]["command"] = self.v4l2control.parse_bool(key=key, state=self.camera_params_component[_var_name])

            elif val["type"] == "menu":
                self.camera_params_component[key] = ttk.Combobox(
                    self.master,
                    value=val["items"],
                    state="readonly")
                # self.camera_params_component[key].set(val["value"])
                self.camera_params_component[key].set(key + " (current value:{})".format(val["value"]))
                self.camera_params_component[key].bind(
                    '<<ComboboxSelected>>', self.v4l2control.parse_menu(key=key, cb=self.camera_params_component[key])
                )
                sticky = "ew"
            if key in self.camera_params_component:
                self.camera_params_component[key].grid(row=row, column=column, sticky=sticky)
                n += 1


class MainApp(tk.Frame):

    def __init__(self, master=tk.Tk(),
                 image_queue=None,config_queue=None, result_queue=None,
                 camera_config=None, cls_dict=None, module_queue=None,
                 pause_queue=None,):

        super().__init__(master)

        # Queue()
        self.image_q = image_queue
        self.result_q = result_queue
        self.config_q = config_queue
        self.module_q = module_queue
        self.pause_q = pause_queue

        self.camera_config = camera_config
        self.cls_dict = cls_dict

        self.selected_module_name = None
        self.vis = None

        self.orig_image: np.ndarray
        self.canvas_image: np.ndarray
        self.result = None
        self.selected_config = {}
        self.delay = 1  # [ms]

        # image averaging denoise

        self.fps = FPS()

        self.width = self.master.winfo_screenwidth()
        self.height = self.master.winfo_screenheight()

        self.image_frame = tk.Canvas(self.master, bg="black")
        # choosing camera io and resolution etc
        self.settings_frame = tk.Frame(self.master, bg="white")
        self.ia_cls_frame = tk.Frame(self.master, bg="white")

        # camera input
        self.camera_pulldown = ttk.Combobox(self.settings_frame,
                                state="readonly", height=20, width=40)
        # fourcc resolution fps selection
        self.settings_pulldown = ttk.Combobox(self.settings_frame,
                                state="readonly", height=20, width=40)



        # button frame, save etc....
        self.button_frame = tk.Frame(self.master, bg="white")  # basic components like save button
        self.image_store_path = path = os.path.join(os.environ['HOME'], "Desktop", "data")
        self.denoise = tk.BooleanVar(value=False)
        self.pause = False # flag to stop the update canvas process from grabbing the queue. used when saving images

        # image analysis modules
        self.cls_pulldown = ttk.Combobox(self.button_frame,
                                         state="readonly",
                                         height=20, width=55)

        # show/hide camera_control
        self.check_value = tk.BooleanVar(value=False)
        self.camera_control_sh = tk.Checkbutton(self.settings_frame, variable=self.check_value, text="Camera\nControl")


        self.initialize_main_frame()
        self.initialize_settings_frame()
        self.initialize_button_frame()
        #self.initialize_io_frame()


        self.camera_control_window = tk.Toplevel(self.master)
        self.camera_control_app = None
        self.call_camera_param_control_window()

        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

        if self.image_q:
            self.update_multiprocess()

    def on_closing(self):
        self.master.destroy()

    def call_camera_param_control_window(self):
        # self.camera_control_window = tk.Toplevel(self.master) called in __init__
        self.camera_control_app = CameraControl(self.camera_control_window,
                                                self.selected_config["io_name"])
                                                # self.camera_config)  #  camera config will be retreived everytime from v4l2 for newest current value
        self.camera_control_window.withdraw()

    def initialize_main_frame(self):
        self.master.title("main window")
        self.master.geometry("{}x{}+0+0".format(self.width, self.height))

        self.settings_frame.pack(side=tk.TOP, anchor="n", fill="x")
        self.button_frame.pack(side=tk.TOP, anchor="n", fill="x")
        self.ia_cls_frame.pack(side=tk.TOP, anchor="n", fill="x")

        self.image_frame.pack(side=tk.LEFT, expand=1, fill="both")

    def initialize_button_frame(self):
        def save_single():
            # プロセスがはやすぎると枯渇する？mainappをcamの前に立ち上げてimage_q.getしても停止しないのはqueueがfillされてるから？
            self.pause = True

            filename = time.strftime("%Y%m%d_%H%M%S")
            if not os.path.exists(self.image_store_path):
                os.makedirs(self.image_store_path)
            image = self.image_q.get()

            if self.denoise.get():
                for i in range(4):
                    _image = self.image_q.get()
                    image = np.average([image, _image], axis=0)
                    time.sleep(1)

            cv2.imwrite(os.path.join(self.image_store_path, filename+ ".jpg"), image)
            messagebox.showinfo('', "image saved as {}".format(filename + ".jpg"))

            self.pause = False

        def set_value(io_name, key, value):
            s = "/usr/bin/v4l2-ctl -d {} -c {}={}".format(io_name, key, value)
            output = subprocess.run([s], shell=True, capture_output=True)
            assert output.returncode == 0, "{} failed".format(s)

        def save_multifocus():

            self.pause = True

            # disable multifocus
            set_value(self.selected_config["io_name"], "focus_auto", 0)
            self.camera_config = RetreiveV4L2Info().config

            focus = copy.deepcopy(self.camera_config["parameters"][self.selected_config["io_name"]]["focus_absolute"])  # dict
            # key: max, min, step, type, value
            # normalize so that step size will be 1
            f_min = int(focus["min"]) // int(focus["step"])
            f_max = int(focus["max"]) // int(focus["step"])
            f_value = int(focus["value"]) // int(focus["step"])
            #set_value(self.selected_config["io_name"], "focus_absolute", int(focus["value"]))
            # set the minimum range of increment.
            # if step =1, will increase focus by 5
            # step size will be controlled via toml file? in future
            if int(focus["step"]) == 1:
                step = 10
            else:
                step = int(focus["step"])

            filename = time.strftime("%Y%m%d_%H%M%S")

            if not os.path.exists(os.path.join(self.image_store_path, filename)):
                os.makedirs(os.path.join(self.image_store_path, filename))


            # to be researched. the first focus change seems to not work so.

            for i in range(7):
                f_value += step
                if int(f_value) * int(focus["step"]) > f_max:
                    break
                orig_scale_f_value = min(f_max, int(f_value) * int(focus["step"]))
                set_value(self.selected_config["io_name"], "focus_absolute", orig_scale_f_value)
                time.sleep(0.5)

                image = self.image_q.get()

                if self.denoise.get():
                    for _ in range(4):
                        _image = self.image_q.get()
                        image = np.average([image, _image], axis=0)
                        time.sleep(0.1)

                cv2.imwrite(os.path.join(self.image_store_path,
                                         filename, filename + "_" + str(i).zfill(2) + ".jpg"), image)

            set_value(self.selected_config["io_name"], "focus_absolute", int(focus["value"]))
            answer = messagebox.askyesno('', "images saved in folder {}\n Proceed focus stacking?".format(filename))

            if answer:
                #messagebox.showinfo('', "focus stacking will be soon availabe")
                stacked = focus_stack(os.path.join(self.image_store_path, filename))
                cv2.imwrite(os.path.join(self.image_store_path, filename + "_stacked.jpg"), stacked)
                messagebox.showinfo('', "stacked image saved as {}".format(filename + "_stacked.jpg"))
            gc.collect()

            self.pause = False


        self.buttons = {}
        button_and_commands = [["Denoise", self.denoise, "checkbutton"],
                               ["Save Image", save_single, "button"],
                               ["Save MultiFocus Image", save_multifocus, "button"]
                               ]
        for name, command_or_var, _type in button_and_commands:
            if _type == "button":
                self.buttons[name] = tk.Button(self.button_frame,
                                               text=name,
                                               command=command_or_var)
                self.buttons[name].pack(side=tk.LEFT, padx=5, pady=5)
            elif _type == "checkbutton":
                self.buttons[name] = tk.Checkbutton(self.button_frame, text=name, variable=command_or_var)
                self.buttons[name].pack(side=tk.LEFT, padx=5, pady=5)

        self.cls_pulldown["values"] = [v["description"] for k,v in self.cls_dict.items()]
        self.cls_pulldown.pack(side=tk.LEFT, padx="10")
        self.cls_pulldown.set(self.cls_pulldown["values"][0])
        self.cls_pulldown.bind(
            '<<ComboboxSelected>>', self.change_io_cls_combobox
        )

    def initialize_settings_frame(self):
        cameras = []
        for k, v in self.camera_config["devices"].items():  #camera_2: {camera_name, io_name}
            cameras.append(v["io_name"] + " : " + k[:30]+"...")
        assert len(cameras) >= 1, "no camera detected"
        self.camera_pulldown["values"] = cameras
        self.camera_pulldown.pack(side=tk.LEFT, padx=10)
        self.camera_pulldown.current(0)
        self.camera_pulldown.bind(
            '<<ComboboxSelected>>', self.change_camera_combobox
        )

        vals = self.get_settings_from_camera_and_io_name()
        self.settings_pulldown["values"] = vals
        self.settings_pulldown.pack(side=tk.LEFT, padx=10)
        self.settings_pulldown.current(0)
        self.settings_pulldown.bind(
            '<<ComboboxSelected>>', self.change_settings_combobox
        )
        self.set_selected_config()
        self.camera_control_sh["command"] = self.camera_control_click
        self.camera_control_sh.pack(side=tk.LEFT, padx=10)

    # def initialize_io_frame(self):
    #
    #     self.cls_pulldown["values"] = [v["description"] for k,v in self.cls_dict.items()]
    #     self.cls_pulldown.pack(side=tk.LEFT, padx="10")
    #     self.cls_pulldown.set(self.cls_pulldown["values"][0])
    #     self.cls_pulldown.bind(
    #         '<<ComboboxSelected>>', self.change_io_cls_combobox
    #     )

    def change_io_cls_combobox(self,event):
        self.pause = 1
        self.pause_q.put(1)
        # resume flag will be exec in IaProcess.replace_module

        val = self.cls_pulldown.get() # is description
        selected_module = {k: v for k, v in self.cls_dict.items() if v["description"] == val}
        assert len(selected_module) == 1, "description is not unique:{}".selected_module
        key = list(selected_module.keys())[0]
        self.module_q.put(key)
        self.selected_module_name = key
        self.vis = selected_module[key]["vis"]

        #print(self.vis)


    def get_settings_from_camera_and_io_name(self):
        # get settings from camera and io name in pull down
        io_name = self.camera_pulldown.get().split(" : ")[0]  # "io_name : camera_name"
        vals = []
        for fourcc, v in self.camera_config["resolutions"][io_name].items():  # key is fourcc
            for resolution, v2 in v.items():  # dict
                for fps in v2: # list
                    s = "fourcc: {} / width x height: {} / fps: {}".format(fourcc, resolution, fps)
                    vals.append(s)
        return vals

    def set_config_queue(self):
        self.set_selected_config()
        if not self.config_q.full():
            self.config_q.put(self.selected_config)

    def set_selected_config(self):
        def get_fourcc_res_and_fps_from_concatenated_string(s):
            _fourcc, _wh, _fps = s.split("/")
            fourcc = _fourcc.split("fourcc:")[1].strip()
            width, height = _wh.split("width x height:")[1].strip().split("x")
            fps = _fps.split("fps:")[1].strip()
            return fourcc, int(width), int(height), int(fps)
        self.selected_config["io_name"], self.selected_config["camera_name"] = self.camera_pulldown.get().split(" : ")
        self.selected_config["fourcc"], self.selected_config["width"], self.selected_config["height"],self.selected_config["fps"] \
            = get_fourcc_res_and_fps_from_concatenated_string(self.settings_pulldown.get())

    def change_camera_combobox(self, event):
        # refill settings combobox and set config queue
        vals = self.get_settings_from_camera_and_io_name()
        self.settings_pulldown["values"] = vals
        self.settings_pulldown.current(0)
        self.set_config_queue()

        # uncheck the camera config
        self.check_value.set(False)

        # renew the cameraconfigwindow
        self.camera_control_window.withdraw()
        self.camera_control_app.io_name = self.selected_config["io_name"]
        self.camera_control_app.initialize_camera_control_frame()
        print("new io is",self.camera_control_app.io_name)

    def change_settings_combobox(self, event):
        # set config queue only
        self.set_config_queue()


    def camera_control_click(self):
        if self.check_value.get():
            self.camera_control_window.deiconify()
            self.camera_control_window.lift()
            self.camera_control_window.attributes("-topmost", True)
        else:
            self.camera_control_window.attributes("-topmost", False)
            self.camera_control_window.withdraw()

    def update_multiprocess(self):
        # global image, result  # mandatory if not a class variable.
        if self.pause_q.qsize() != 0:
            self.pause = self.pause_q.get()
        if not self.pause:
            # reset canvas
            self.image_frame.delete("bbox")
            self.image_frame.delete("text")

            self.orig_image = self.image_q.get()
            self.canvas_image = cv2.cvtColor(self.orig_image, cv2.COLOR_BGR2RGB)


            if self.vis:
                self.result = self.result_q.get()
                if self.result is not None and self.selected_module_name == self.result["module_name"]:
                    self.canvas_image = self.vis(self.orig_image, self.result)
                    self.canvas_image = cv2.cvtColor(self.canvas_image, cv2.COLOR_BGR2RGB)

            input_size = [self.image_frame.winfo_width(), self.image_frame.winfo_height()]
            self.ratio = [input_size[0]/self.canvas_image.shape[1], input_size[1]/self.canvas_image.shape[0]]  # x,y fix image to canvas size.

            self.canvas_image = cv2.resize(self.canvas_image, dsize=tuple(input_size))
            self.canvas_image = Image.fromarray(self.canvas_image)
            self.canvas_image = ImageTk.PhotoImage(image=self.canvas_image)
            self.image_frame.create_image(0, 0, image=self.canvas_image, anchor="nw")


            # if self.result_q.qsize() != 0:
            #     self.annotate_canvas()

            s = "FPS: " + str(self.fps())
            self.image_frame.create_text(10, 10, anchor="nw",
                                         text=s, fill="red", font=('', '30', ''),tag="text")
        else:
            while self.result_q.qsize() != 0:
                _ = self.result_q.get()
            self.image_frame.delete("bbox")
            self.image_frame.delete("text")
            self.image_frame.create_text(10, 10, anchor="nw",
                                         text="processing, please wait.", fill="red", font=('', '30', ''), tag="text")

        self.master.after(self.delay, self.update_multiprocess)