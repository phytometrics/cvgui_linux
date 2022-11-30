import copy
import cv2
import os
import time
import subprocess

import numpy as np
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox
from PIL import Image, ImageTk

from .general_utils import FPS
from .v4l2_utils import V4L2Control, RetreiveV4L2Info
from .focus_utils import focus_stack
from .camera_control_window import CameraControl

import sys
sys.path.append("../")


class MainApp(tk.Frame):

    def __init__(self, master=tk.Tk(),
                 image_queue=None, config_queue=None, result_queue=None,
                 camera_config=None, ia_module_dict=None, module_queue=None,
                 pause_queue=None,):

        super().__init__(master)

        # Queue()
        self.image_q = image_queue
        self.result_q = result_queue
        self.config_q = config_queue
        self.module_q = module_queue
        self.pause_q = pause_queue

        self.camera_config = camera_config
        self.ia_module_dict = ia_module_dict

        self.selected_module_name = None
        self.vis = None

        self.orig_image: np.ndarray
        self.canvas_image: np.ndarray
        self.result = None
        self.preview_config = {}
        self.acquire_config = {}
        self.delay = 1  # [ms]

        self.fps = FPS()
        self.v4l2control: V4L2Control

        self.width = self.master.winfo_screenwidth()
        self.height = self.master.winfo_screenheight()

        self.frames = {
            "control_frame_row1": tk.Frame(self.master, bg="white"),
            "control_frame_row2": tk.Frame(self.master, bg="white"),
            "control_frame_row3": tk.Frame(self.master, bg="white"),
            "image_frame": tk.Canvas(self.master, bg="black"),
            "focus_control_frame": tk.Frame(self.master, bg="white")
        }
        self.flags = {
            "sync_flag": tk.BooleanVar(value=False),
            "denoise": tk.BooleanVar(value=False),
            "pause": False,
            "camera_config_flag": tk.BooleanVar(value=False)
        }
        self.widgets = {
            "camera_pulldown":ttk.Combobox(self.frames["control_frame_row1"],
                                           state="readonly", height=20, width=40),
            "preview_fourcc_res_fps_pulldown": ttk.Combobox(self.frames["control_frame_row2"],
                                state="readonly", height=20, width=40),
            "sync_check": tk.Checkbutton(self.frames["control_frame_row2"],
                                         variable=self.flags["sync_flag"],
                                         text="sync",
                                         command=self.sync_click),
            "acquire_fourcc_res_fps_pulldown" : ttk.Combobox(self.frames["control_frame_row2"],
                                                           state="readonly", height=20, width=40),
            "ia_module_pulldown": ttk.Combobox(self.frames["control_frame_row3"],
                                         state="readonly",
                                         height=20, width=55),
            "camera_control_sh": tk.Checkbutton(self.frames["control_frame_row1"],
                                                variable=self.flags["camera_config_flag"],
                                                text="Camera Control")
        }

        self.image_store_path = os.path.join(os.environ['HOME'], "Desktop", "data")

        self.initialize_main_frame()
        self.initialize_control_frame_row1()
        self.initialize_control_frame_row2()
        self.initialize_control_frame_row3()

        self.camera_control_window = tk.Toplevel(self.master)
        self.camera_control_app = None
        self.call_camera_param_control_window()

        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

        if self.image_q:
            self.update_multiprocess()

    def on_closing(self):
        self.master.destroy()

    def call_camera_param_control_window(self):
        self.camera_control_app = CameraControl(self.camera_control_window,
                                                self.preview_config["io_name"])
        self.camera_control_window.withdraw()

    def initialize_main_frame(self):
        self.master.title("main window")
        self.master.geometry("{}x{}+0+0".format(self.width, self.height))
        self.master.configure(bg="white")

        self.frames["control_frame_row1"].pack(side=tk.TOP, anchor="n", fill="x")
        self.frames["control_frame_row2"].pack(side=tk.TOP, anchor="n", fill="x")
        self.frames["control_frame_row3"].pack(side=tk.TOP, anchor="n", fill="x")
        self.frames["image_frame"].pack(side=tk.LEFT, expand=1, fill="both")

    def initialize_focus_control(self):

        # must run after selected config is chosen
        self.focus_component = {}
        self.v4l2control = V4L2Control(self.preview_config["io_name"])

        for widget in self.frames["focus_control_frame"].winfo_children():
            widget.destroy()

        for key, val in self.camera_config["parameters"][self.preview_config["io_name"]].items():
            if "focus" in key.lower():

                if val["type"] == "int":
                    self.focus_component[key] = tk.Scale(
                        self.frames["focus_control_frame"],
                        orient="v",
                        from_=int(val["min"]) // int(val["step"]),
                        to=int(val["max"]) // int(val["step"]),
                        length=500
                    )
                    self.focus_component[key].set(int(val["value"]) // int(val["step"]))
                    self.focus_component[key]["command"] = self.v4l2control.parse_int(key=key,step=val["step"])

                elif val["type"] == "bool":

                    _var_name = key + "var"
                    self.focus_component[_var_name] = tk.BooleanVar()
                    if int(val["value"]):
                        self.focus_component[_var_name].set(1)
                    else:
                        # self.camera_params_component[key].deselect()
                        self.focus_component[_var_name].set(0)

                    self.focus_component[key] = tk.Checkbutton(
                        self.frames["focus_control_frame"],
                        text=key[:10],
                        variable=self.focus_component[_var_name]
                    )
                    self.focus_component[key]["command"] = self.v4l2control.parse_bool(key=key, state=self.focus_component[_var_name])

                if key in self.focus_component:
                    self.focus_component[key].pack(side=tk.TOP, padx=10)

            else:
                continue

        if bool(self.focus_component):
            # if components are present, pack
            self.frames["focus_control_frame"].pack(side=tk.TOP, padx=10, fill="y")


    def initialize_control_frame_row1(self):
        cameras = []
        for k, v in self.camera_config["devices"].items():  #camera_2: {camera_name, io_name}
            cameras.append(v["io_name"] + " : " + k[:30]+"...")
        assert len(cameras) >= 1, "no camera detected"
        self.widgets["camera_pulldown"]["values"] = cameras
        self.widgets["camera_pulldown"].pack(side=tk.LEFT, padx=10)
        self.widgets["camera_pulldown"].current(0)
        self.widgets["camera_pulldown"].bind(
            '<<ComboboxSelected>>', self.change_camera_combobox
        )

        self.widgets["camera_control_sh"]["command"] = self.camera_control_click
        self.widgets["camera_control_sh"].pack(side=tk.LEFT, padx=10)


    def initialize_control_frame_row2(self):
        # resolution etc combobox
        preview_txt = tk.Label(self.frames["control_frame_row2"], text="Settings:",bg="white",pady="5")
        preview_txt.pack(side=tk.LEFT)
        vals = self.get_settings_from_camera_and_io_name()
        self.widgets["preview_fourcc_res_fps_pulldown"]["values"] = vals
        self.widgets["preview_fourcc_res_fps_pulldown"].pack(side=tk.LEFT, padx=5)
        self.widgets["preview_fourcc_res_fps_pulldown"].current(0) #set to the first index
        self.widgets["preview_fourcc_res_fps_pulldown"].bind(
            '<<ComboboxSelected>>', self.change_preview_settings_combobox
        )
        self.set_preview_config()

        # self.widgets["sync_check"].pack(side=tk.LEFT, padx=5)
        #
        # acquire_txt = tk.Label(self.frames["control_frame_row2"], text="Acquire:",bg="white")
        # acquire_txt.pack(side=tk.LEFT)
        #
        # self.widgets["acquire_fourcc_res_fps_pulldown"]["values"] = vals
        # self.widgets["acquire_fourcc_res_fps_pulldown"].pack(side=tk.LEFT, padx=5)
        # self.widgets["acquire_fourcc_res_fps_pulldown"].set(
        #     self.widgets["preview_fourcc_res_fps_pulldown"].get()
        # )
        # self.widgets["acquire_fourcc_res_fps_pulldown"].bind(
        #     '<<ComboboxSelected>>', self.change_acquire_settings_combobox
        # )
        # self.set_acquire_config()


        # for row1 but selected config must be enumerated so placing here
        self.initialize_focus_control()

    def initialize_control_frame_row3(self):
        def save_single():
            self.flags["pause"] = True

            # # previewとacquireの設定が違う場合に切り替える
            # if not self.flags["sync_flag"].get() and self.preview_config != self.acquire_config:
            #     time.sleep(0.2)  # 別プロセスのimage_qが枯渇するのを待つ
            #     pass


            filename = time.strftime("%Y%m%d_%H%M%S")
            if not os.path.exists(self.image_store_path):
                os.makedirs(self.image_store_path)
            image = self.image_q.get()

            if self.flags["denoise"].get():
                for i in range(4):
                    _image = self.image_q.get()
                    image = np.average([image, _image], axis=0)
                    time.sleep(0.2)

            cv2.imwrite(os.path.join(self.image_store_path, filename+ ".jpg"), image)
            messagebox.showinfo('', "image saved as {}".format(filename + ".jpg"))
            # time.sleep(0.2)
            #
            # self.config_q.put(self.preview_config)
            # print("wait")
            # time.sleep(0.2)  # 別プロセスのimage_qが枯渇するのを待つ
            
            self.flags["pause"] = False

        def set_value(io_name, key, value):
            s = "/usr/bin/v4l2-ctl -d {} -c {}={}".format(io_name, key, value)
            output = subprocess.run([s], shell=True, capture_output=True)
            assert output.returncode == 0, "{} failed".format(s)

        def save_multifocus():

            # camera_configの現在focus値が更新されていないので最新のものにする必要あり
            # focusだけをピンポイントに更新するのが理想だけれど、今は全部読み込みなおす。
            self.camera_config = RetreiveV4L2Info().config

            self.flags["pause"] = True

            # # previewとacquireの設定が違う場合に切り替える
            # if not self.flags["sync_flag"].get():
            #     self.config_q.put(self.acquire_config)
            #     time.sleep(0.2)  # 別プロセスのimage_qが枯渇するのを待つ
            #     pass

            # disable multifocus
            # focus absolute and focus name differs by camera, so will search it everytime
            auto_focus_name = None
            focus_absolute_name = None
            for key, val in self.camera_config["parameters"][self.preview_config["io_name"]].items():
                if "focus" in key and "auto" in key:
                    auto_focus_name = key
                elif "focus" in key and "absolute" in key:
                    focus_absolute_name = key


            set_value(self.preview_config["io_name"], auto_focus_name, 0)
            self.camera_config = RetreiveV4L2Info().config

            focus = copy.deepcopy(self.camera_config["parameters"][self.preview_config["io_name"]][focus_absolute_name])  # dict
            # key: max, min, step, type, value
            # normalize so that step size will be 1
            f_min = int(focus["min"]) // int(focus["step"])
            f_max = int(focus["max"]) // int(focus["step"])
            f_value = int(focus["value"]) // int(focus["step"])
            #set_value(self.preview_config["io_name"], "focus_absolute", int(focus["value"]))
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

            for i in range(5):
                f_value += step
                if int(f_value) * int(focus["step"]) > f_max:
                    break
                orig_scale_f_value = min(f_max, int(f_value) * int(focus["step"]))
                set_value(self.preview_config["io_name"], focus_absolute_name, orig_scale_f_value)
                time.sleep(0.5)

                image = self.image_q.get()

                if self.flags["denoise"].get():
                    for _ in range(4):
                        _image = self.image_q.get()
                        image = np.average([image, _image], axis=0)
                        time.sleep(0.1)

                cv2.imwrite(os.path.join(self.image_store_path,
                                         filename, filename + "_" + str(i).zfill(2) + ".jpg"), image)

            set_value(self.preview_config["io_name"], focus_absolute_name, int(focus["value"]))
            answer = messagebox.askyesno('', "images saved in folder {}\n Proceed to focus stack?".format(filename))

            if answer:
                #messagebox.showinfo('', "focus stacking will be soon availabe")
                stacked = focus_stack(os.path.join(self.image_store_path, filename))
                cv2.imwrite(os.path.join(self.image_store_path, filename + "_stacked.jpg"), stacked)
                messagebox.showinfo('', "stacked image saved as {}".format(filename + "_stacked.jpg"))
            #gc.collect()

            # self.config_q.put(self.preview_config)
            self.flags["pause"] = False


        self.buttons = {}
        button_and_commands = [["Denoise", self.flags["denoise"], "checkbutton"],
                               ["Save Image", save_single, "button"],
                               ["Save MultiFocus Image", save_multifocus, "button"]
                               ]
        for name, command_or_var, _type in button_and_commands:
            if _type == "button":
                self.buttons[name] = tk.Button(self.frames["control_frame_row3"],
                                               text=name,
                                               command=command_or_var)
                self.buttons[name].pack(side=tk.LEFT, padx=5, pady=5)
            elif _type == "checkbutton":
                self.buttons[name] = tk.Checkbutton(self.frames["control_frame_row3"], text=name, variable=command_or_var)
                self.buttons[name].pack(side=tk.LEFT, padx=5, pady=5)

        self.widgets["ia_module_pulldown"]["values"] = [v["description"] for k,v in self.ia_module_dict.items()]
        self.widgets["ia_module_pulldown"].pack(side=tk.LEFT, padx="10")
        self.widgets["ia_module_pulldown"].set("Image Analysis Modules")
        self.widgets["ia_module_pulldown"].bind(
            '<<ComboboxSelected>>', self.change_ia_module_combobox
        )

    def change_ia_module_combobox(self,event):
        self.flags["pause"] = 1
        self.pause_q.put(1)
        # resume flag will be exec in IaProcess.replace_module

        val = self.widgets["ia_module_pulldown"].get() # is description
        selected_module = {k: v for k, v in self.ia_module_dict.items() if v["description"] == val}
        assert len(selected_module) == 1, "description is not unique:{}".selected_module
        key = list(selected_module.keys())[0]
        self.module_q.put(key)
        self.selected_module_name = key
        self.vis = selected_module[key]["vis"]

        #print(self.vis)

    def get_settings_from_camera_and_io_name(self):
        # get settings from camera and io name in pull down
        io_name = self.widgets["camera_pulldown"].get().split(" : ")[0]  # "io_name : camera_name"
        vals = []
        for fourcc, v in self.camera_config["resolutions"][io_name].items():  # key is fourcc
            for resolution, v2 in v.items():  # dict
                for fps in v2: # list
                    s = "fourcc: {} / width x height: {} / fps: {}".format(fourcc, resolution, fps)
                    vals.append(s)
        return vals

    def set_preview_config_queue(self):
        self.set_preview_config()
        if not self.config_q.full():
            self.config_q.put(self.preview_config)

    # 統合する
    def set_preview_config(self):
        def get_fourcc_res_and_fps_from_concatenated_string(s):
            _fourcc, _wh, _fps = s.split("/")
            fourcc = _fourcc.split("fourcc:")[1].strip()
            width, height = _wh.split("width x height:")[1].strip().split("x")
            fps = _fps.split("fps:")[1].strip()
            return fourcc, int(width), int(height), int(fps)
        self.preview_config["io_name"], self.preview_config["camera_name"] = self.widgets["camera_pulldown"].get().split(" : ")
        self.preview_config["fourcc"], self.preview_config["width"], self.preview_config["height"],self.preview_config["fps"] \
            = get_fourcc_res_and_fps_from_concatenated_string(self.widgets["preview_fourcc_res_fps_pulldown"].get())

    def set_acquire_config(self):
        def get_fourcc_res_and_fps_from_concatenated_string(s):
            _fourcc, _wh, _fps = s.split("/")
            fourcc = _fourcc.split("fourcc:")[1].strip()
            width, height = _wh.split("width x height:")[1].strip().split("x")
            fps = _fps.split("fps:")[1].strip()
            return fourcc, int(width), int(height), int(fps)
        self.acquire_config["io_name"], self.acquire_config["camera_name"] = self.widgets["camera_pulldown"].get().split(" : ")
        self.acquire_config["fourcc"], self.acquire_config["width"], self.acquire_config["height"],self.acquire_config["fps"] \
            = get_fourcc_res_and_fps_from_concatenated_string(self.widgets["acquire_fourcc_res_fps_pulldown"].get())

    def change_camera_combobox(self, event):
        # refill settings combobox and set config queue
        vals = self.get_settings_from_camera_and_io_name()
        self.widgets["preview_fourcc_res_fps_pulldown"]["values"] = vals
        self.widgets["preview_fourcc_res_fps_pulldown"].current(0)

        self.widgets["acquire_fourcc_res_fps_pulldown"]["values"] = vals
        #中身をコピー
        self.widgets["acquire_fourcc_res_fps_pulldown"].set(
            self.widgets["preview_fourcc_res_fps_pulldown"].get()
        )


        self.set_preview_config_queue()
        self.set_acquire_config()

        # uncheck the camera config
        self.flags["camera_config_flag"].set(False)

        # renew the cameraconfigwindow
        self.camera_control_window.withdraw()
        self.camera_control_app.io_name = self.preview_config["io_name"]
        self.camera_control_app.initialize_camera_control_frame()

        # rebuild the focus menu
        self.frames["focus_control_frame"].pack_forget()
        self.initialize_focus_control()
        print("new io is",self.camera_control_app.io_name)

    def change_acquire_settings_combobox(self, event):
        print("changing acquire config to")
        print(self.widgets["acquire_fourcc_res_fps_pulldown"].get())
        self.set_acquire_config()

    def change_preview_settings_combobox(self, event):
        # set config queue
        self.set_preview_config_queue()

        if self.flags["sync_flag"].get():
            self.widgets["acquire_fourcc_res_fps_pulldown"].set(
                self.widgets["preview_fourcc_res_fps_pulldown"].get()
            )
        # if sync == True, change also the acquire combobox

    def camera_control_click(self):
        if self.flags["camera_config_flag"].get():
            self.camera_control_window.deiconify()
            self.camera_control_window.lift()
            self.camera_control_window.attributes("-topmost", True)
        else:
            self.camera_control_window.attributes("-topmost", False)
            self.camera_control_window.withdraw()

    def sync_click(self):
        if self.flags["sync_flag"].get():
            self.widgets["acquire_fourcc_res_fps_pulldown"].set(
                self.widgets["preview_fourcc_res_fps_pulldown"].get()
            )

    def update_multiprocess(self):
        # global image, result  # mandatory if not a class variable.
        if self.pause_q.qsize() != 0:
            self.flags["pause"] = self.pause_q.get()
            # print("pause is ",self.flags["pause"])
        if not self.flags["pause"]:
            # reset canvas
            self.frames["image_frame"].delete("bbox")
            self.frames["image_frame"].delete("text")

            self.orig_image = self.image_q.get()
            self.canvas_image = cv2.cvtColor(self.orig_image, cv2.COLOR_BGR2RGB)

            if self.vis:
                #ここで処理待ちが発生するので、result_qのqsizeが空の場合、１つ前のresultを使うことでfpsが向上すると思う
                if self.result_q.qsize() != 0:
                    self.result = self.result_q.get()
                    _result = self.result
                else:
                    if "_result" in locals():
                        self.result = _result

                if self.result is not None and self.selected_module_name == self.result["module_name"]:
                    self.canvas_image = self.vis(self.orig_image, self.result)
                    self.canvas_image = cv2.cvtColor(self.canvas_image, cv2.COLOR_BGR2RGB)
                    # self.canvas_imageがリスト（複数画像）だったらタイリングするコマンドをここに将来入れる

            input_size = [self.frames["image_frame"].winfo_width(), self.frames["image_frame"].winfo_height()]
            self.ratio = [input_size[0]/self.canvas_image.shape[1], input_size[1]/self.canvas_image.shape[0]]  # x,y fix image to canvas size.

            self.canvas_image = cv2.resize(self.canvas_image, dsize=tuple(input_size))
            self.canvas_image = Image.fromarray(self.canvas_image)
            self.canvas_image = ImageTk.PhotoImage(image=self.canvas_image)
            self.frames["image_frame"].create_image(0, 0, image=self.canvas_image, anchor="nw")

            s = "FPS: " + str(self.fps())
            self.frames["image_frame"].create_text(10, 10, anchor="nw",
                                         text=s, fill="red", font=('', '30', ''),tag="text")
        else:
            while self.result_q.qsize() != 0:
                _ = self.result_q.get()
            self.frames["image_frame"].delete("bbox")
            self.frames["image_frame"].delete("text")
            self.frames["image_frame"].create_text(10, 10, anchor="nw",
                                         text="loading model, please wait.", fill="red", font=('', '30', ''), tag="text")

        self.master.after(self.delay, self.update_multiprocess)