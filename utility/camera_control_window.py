import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox
from .v4l2_utils import V4L2Control, RetreiveV4L2Info

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
            # if self.focus_only and "focus" not in key.lower():
            #     continue

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

