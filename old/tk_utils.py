import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox

from PIL import Image, ImageTk


class App(tk.Frame):
    def __init__(self, master=tk.Tk(), camera=None):
        super().__init__(master)

        self.camera = camera
        # self.camera_dict = camera_dict
        self.width = self.master.winfo_screenwidth()
        self.height = self.master.winfo_screenheight()

        self.control_frame = tk.Frame(self.master, bg="white")
        self.control_frame2 = tk.Frame(self.master, bg="white")
        self.image_frame = tk.Canvas(self.master, bg="black")

        self.configs_pulldown = None
        self.selected_config = tk.StringVar()

        self.initialize_frames()
        self.set_control_frames()

    def initialize_frames(self):
        self.master.title("main window")
        self.master.geometry("{}x{}+0+0".format(self.width, self.height))

        self.control_frame.pack(side=tk.TOP, anchor="n", fill="x")
        self.control_frame2.pack(side=tk.TOP, anchor="n", fill="x")
        self.image_frame.pack(side=tk.LEFT, expand=1, fill="both")

    def set_control_frames(self):
        val = []
        for k, v in self.camera.camera_dict.items():
            # k is config_00 and v is camera_name, io_name, ....
            io_name = v["io_name"]
            camera_name = v["camera_name"][:30] + "..."
            w = str(v["width"])
            h = str(v["height"])
            fourcc = v["fourcc"]
            fps = str(v["fps"])
            s = "  ".join([k, io_name, camera_name, fourcc, "width:" + w, "height:" + h, "fps:" + fps])
            val.append(s)
        val = tuple(val)

        self.configs_pulldown = ttk.Combobox(self.control_frame,
                                             values=val,
                                             textvariable=self.selected_config,
                                             state="readonly", height=20, width=100)
        self.configs_pulldown.current(0)  # init value
        # self.configs_pulldown.bind('<<ComboboxSelected>>', self.change_camera_settings)
        self.configs_pulldown.pack(side=tk.LEFT, fill="x")

    def change_camera_settings(self, event):
        # if dev/video changes, reconnect.
        # if not, just change h, w, fps.
        pass




    def update_image_frame(self):
        pass
