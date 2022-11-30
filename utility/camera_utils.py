import cv2
import atexit
import time
import collections
import numpy as np

class UVCCamera:
    def __init__(self, camera_config=None):
        self.camera_config = camera_config

        self.cap = None
        self.current_config = {}
        self.image = None
        self.denoise = None
        self.initialize_config()
        self.connect_camera()

    def initialize_config(self):
        # get the first element of the camera config. this will be also done with
        camera_name = next(iter(self.camera_config["devices"]))
        io_name = self.camera_config["devices"][camera_name]["io_name"]
        fourcc = next(iter(self.camera_config["resolutions"][io_name]))
        wh = next(iter(self.camera_config["resolutions"][io_name][fourcc]))
        w,h = wh.split(" x ")
        fps = self.camera_config["resolutions"][io_name][fourcc][wh][0]

        self.current_config["fourcc"] = fourcc
        self.current_config["camera_name"] = camera_name
        self.current_config["io_name"] = io_name
        self.current_config["width"] = int(w)
        self.current_config["height"] = int(h)
        self.current_config["fps"] = int(fps)

    def connect_camera(self):
        try:
            try:
                self.cap = cv2.VideoCapture(self.current_config["io_name"], cv2.CAP_V4L2)
            except Exception as e:
                print(e)
                raise RuntimeError("v4l2 not installed. running with backend:".format(self.cap.get(cv2.CAP_PROP_BACKEND)))

            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.current_config["fourcc"]))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.current_config["width"])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.current_config["height"])
            self.cap.set(cv2.CAP_PROP_FPS, self.current_config["fps"])
            # self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            # self.cap.set(cv2.CAP_PROP_FOCUS, 0)

            r, image = self.cap.read()
            if not r:
                raise RuntimeError("Image Retrieval from Camera Failed")
        except Exception as e:
            print(e)
            raise RuntimeError("Camera Connection Failed")
        finally:
            atexit.register(self.cap.release)

    def change_fourcc_res_fps(self):
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.current_config["fourcc"]))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.current_config["width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.current_config["height"])
        self.cap.set(cv2.CAP_PROP_FPS, self.current_config["fps"])

    def reconnect_camera(self):
        try:
            if self.cap.isOpened():
                self.cap.release()
        except Exception as e:
            pass
        self.connect_camera()

    def check_camera_setting_update(self,incoming_config):
        '''
        :param incoming_config: config_q used in outer space
        :return: 0 if no change, 1 if settings changes
        '''

        if type(incoming_config) is dict:
            print(self.current_config)
            print(incoming_config)
            if incoming_config["io_name"] != self.current_config["io_name"]:
                print("change camera",flush=True)
                self.current_config["io_name"] = incoming_config["io_name"]
                self.current_config["fourcc"] = incoming_config["fourcc"]
                self.current_config["width"] = incoming_config["width"]
                self.current_config["height"] = incoming_config["height"]
                self.current_config["fps"] = incoming_config["fps"]
                self.reconnect_camera()
            elif incoming_config["fourcc"] != self.current_config["fourcc"]:
                print("change fourcc",flush=True)
                self.current_config["fourcc"] = incoming_config["fourcc"]
                self.current_config["width"] = incoming_config["width"]
                self.current_config["height"] = incoming_config["height"]
                self.current_config["fps"] = incoming_config["fps"]
                self.change_fourcc_res_fps()
            elif (incoming_config["height"] != self.current_config["height"]) or \
                    (incoming_config["width"] != self.current_config["width"]):
                print("change res",flush=True)
                self.current_config["width"] = incoming_config["width"]
                self.current_config["height"] = incoming_config["height"]
                self.current_config["fps"] = incoming_config["fps"]
                self.change_fourcc_res_fps()
            elif incoming_config["fps"] != self.current_config["fps"]:
                print("change fps",flush=True)
                self.current_config["fps"] = incoming_config["fps"]
                self.change_fourcc_res_fps()





    def get_frame(self):
        if self.denoise:
            frames = []
            for i in range(4):
                ret, _frame = self.cap.read()
                frames.append(_frame)
            frame = np.median(frames, axis=0).astype(np.uint8)
        else:
            ret, frame = self.cap.read()
        return ret, frame


