import subprocess
import re

class V4L2Control:
    def __init__(self,io_name):
        self.io_name = io_name

    def set_value(self, io_name, key, value):
        s = "/usr/bin/v4l2-ctl -d {} -c {}={}".format(io_name, key, value)
        output = subprocess.run([s], shell=True, capture_output=True)
        #print(s)
        assert output.returncode == 0, "{} failed".format(s)

    def parse_int(self, key=None, step=1):
        # does not use tk.intvar
        def _parse_int(value):
            value = int(value) * int(step)
            self.set_value(self.io_name, key, value)
        return _parse_int

    def parse_bool(self, key=None, state=None):
        # state :tk.booleanvar
        def _parse_bool():
            if state.get():
                value = 1
            else:
                value = 0
            self.set_value(self.io_name, key, value)
        return _parse_bool

    def parse_menu(self, key=None, cb=None):
        # use comboboxselected bind. cb: ttk.combobox
        def _parse_menu(event):
            value = cb.get().split(":")[0]
            self.set_value(self.io_name, key, value)
        return _parse_menu


class RetreiveV4L2Info:
    def __init__(self):
        self.device_list = {}
        self.res_list = {}
        self.param_list = {}

        self.get_camera_configuration()

        self.config = {}
        self.config["devices"] = self.device_list
        self.config["resolutions"] = self.res_list
        self.config["parameters"] = self.param_list

    def get_camera_configuration(self):
        self.get_device_list()  # fills self.device_list
        self.get_res_list()  # fills self.res_list
        self.get_param_list()  # fills self.

    def get_device_list(self):

        output = subprocess.run(["/usr/bin/v4l2-ctl --list-devices"], shell=True, capture_output=True)
        assert output.returncode == 0, "v4l2-ctl command possibly failed check whether v4l-utility is installed"

        camera_ids = output.stdout.decode("utf-8")
        camera_ids = camera_ids.splitlines()  # to list
        d = {}
        camerano = 0
        for s in camera_ids:
            if len(s) == 0:  # empty line for next camera
                camerano += 1
                continue
            if s[-1] == ":":  # camera info
                camera_name = s[:-1]
                d[camera_name] = {}
                d[camera_name]["io_name"] = ""
                # d["camera_" + str(camerano)] = {}
                # d["camera_" + str(camerano)]["camera_name"] = s[:-1]
                # d["camera_" + str(camerano)]["io_name"] = []
            if s.startswith("\t"):  # /dev/videoN
                # just use the first /dev/videoN, may remove for future
                if len(d[camera_name]["io_name"]) == 0:
                    d[camera_name]["io_name"] = s[1:]
                # if len(d["camera_" + str(camerano)]["io_name"]) == 0:
                #     d["camera_" + str(camerano)]["io_name"].append(s[1:])
                #     d["camera_" + str(camerano)]["io_name"] = d["camera_" + str(camerano)]["io_name"][0]

        # remove surface camera
        d = {key: val for key, val in d.items() if key.find("PCI:0000") == -1}
        self.device_list = d

    def get_res_list(self):
        for key, val in self.device_list.items():
            command = "/usr/bin/v4l2-ctl --device {} --list-formats-ext".format(val["io_name"])
            output = subprocess.run([command], shell=True, capture_output=True)
            assert output.returncode == 0, "{} failed".format(command)
            camera_info = output.stdout.decode("utf-8")
            camera_info = camera_info.splitlines()  # to list
            i = 0
            fourcc = ""
            size = ""
            width = ""
            height = ""
            fps = ""

            for line in camera_info:
                # replace to regex search in future -> if [number]: get 'XXXX'
                if line.find("MJPG") != -1:
                    fourcc = "MJPG"
                elif line.find("YUYV") != -1:
                    fourcc = "YUYV"
                elif line.find("H264") != -1:
                    fourcc = "H264"
                else:
                    line = line.replace("\t", "")

                    if line.startswith("Size"):  # size: discrete 1920x1080
                        wh = line.split(" ")[-1].split("x")
                        width = wh[0]
                        height = wh[1]

                    elif line.startswith("Interval"):  # interval: discrete 0.0033s (30.000 fps)
                        # config_no = "config_{}".format(str(i).zfill(3))
                        fps = line.split(" ")[-2].split(".")[0][1:]  # (30.000
                        res = str(int(width)) + " x " + str(int(height))

                        # structure : self.res_list["/dev/videoN"]["MJPG"]["width x height"] list
                        if val["io_name"] not in self.res_list:
                            self.res_list[val["io_name"]] = {}
                        if fourcc not in self.res_list[val["io_name"]]:
                            self.res_list[val["io_name"]][fourcc] = {}
                        if res not in self.res_list[val["io_name"]][fourcc]:
                            self.res_list[val["io_name"]][fourcc][res] = []
                        self.res_list[val["io_name"]][fourcc][res].append(int(fps))

                        # if config_no not in self.res_list:
                        #     self.res_list[config_no] = {}
                        # self.res_list[config_no]["camera_name"] = val["camera_name"]
                        # self.res_list[config_no]["io_name"] = val["io_name"]
                        # self.res_list[config_no]["fourcc"] = fourcc
                        # self.res_list[config_no]["height"] = int(height)
                        # self.res_list[config_no]["width"] = int(width)
                        # self.res_list[config_no]["fps"] = int(fps)
                        # i += 1

    def get_param_list(self):

        def parse_int_item(s):
            param = {}
            param["type"] = "int"
            param["min"] = s.split("min=")[1].split(" ")[0]
            param["max"] = s.split("max=")[1].split(" ")[0]
            param["step"] = s.split("step=")[1].split(" ")[0]
            param["default"] = s.split("default=")[1].split(" ")[0]
            param["value"] = s.split("value=")[1].split(" ")[0]
            return param

        def parse_bool_item(s):
            param = {}
            param["type"] = "bool"
            param["default"] = s.split("default=")[1].split(" ")[0]
            param["value"] = s.split("value=")[1].split(" ")[0]
            return param

        def parse_menu_item(s):
            param = {}
            param["type"] = "menu"
            param["min"] = s.split("min=")[1].split(" ")[0]
            param["max"] = s.split("max=")[1].split(" ")[0]
            param["default"] = s.split("default=")[1].split(" ")[0]
            param["value"] = s.split("value=")[1].split(" ")[0]
            return param

        for key, val in self.device_list.items():

            command = "/usr/bin/v4l2-ctl --device {} -L".format(val["io_name"])
            output = subprocess.run([command], shell=True, capture_output=True)
            assert output.returncode == 0, "{} failed".format(command)
            params = output.stdout.decode("utf-8")
            params = params.splitlines()  # to list
            i = 0
            if val["io_name"] not in self.param_list:
                self.param_list[val["io_name"]] = {}
            for line in params:
                #print(line,flush=True)
                # if not "0x" in line:
                #     continue
                line = line.strip()
                if "(int)" in line:
                    self.param_list[val["io_name"]][line.split(" ")[0]] = parse_int_item(line)
                elif "(bool)" in line:
                    self.param_list[val["io_name"]][line.split(" ")[0]] = parse_bool_item(line)
                elif "(menu)" in line:
                    menu_key = line.split(" ")[0]
                    self.param_list[val["io_name"]][menu_key] = parse_menu_item(line)
                    self.param_list[val["io_name"]][menu_key]["items"] = []
                # the items in menu above
                # item no may not be incremental, so storing everything and creating empty e.g. 1: Manual Mode, 3: Aperture Priority Mode
                elif ":" in line:
                    # while len(self.param_list[val["io_name"]][menu_key]["items"]) < int(line.split(":")[0]):
                    #     self.param_list[val["io_name"]][menu_key]["items"].append("_")
                    self.param_list[val["io_name"]][menu_key]["items"].append(line)





