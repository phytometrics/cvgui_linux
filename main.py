import importlib, inspect
import numpy as np
from multiprocessing import Process, Queue

from utility import v4l2_utils, tk_utils, camera_utils
from image_analysis_modules import PassThrough


def gui_process(camera_info, ia_module_dict, image_q1, result_q1, config_q1, module_q1, pause_q1):
    mainapp = tk_utils.MainApp(image_queue=image_q1,
                               result_queue=result_q1,
                               config_queue=config_q1,
                               module_queue=module_q1,
                               pause_queue=pause_q1,
                               ia_module_dict=ia_module_dict,
                               camera_config=camera_info.config)
    mainapp.mainloop()


def cam_process(image_q1, image_q2, config_q):
    cam = camera_utils.UVCCamera(camera_config=camera_info.config)

    while True:
        ret, image = cam.get_frame()
        if not ret:
            if image_q1.qsize() == 0:
                image_q1.put(np.zeros([10, 10, 3]))
            if image_q2.qsize() == 0:
                image_q2.put(np.zeros([10, 10, 3]))
            continue
        if image_q1.qsize() == 0:
            image_q1.put(image)
        if image_q2.qsize() == 0:
            image_q2.put(image)

        if config_q.qsize() != 0:
            cam.check_camera_setting_update(config_q.get())
            # clear queue upon camera change
            while image_q1.qsize() != 0:
                _ = image_q1.get()
            while image_q2.qsize() != 0:
                _ = image_q2.get()


def get_ia_module_dict():
    '''
    import modules defined as class inheriting BaseModule that are directly defined in __init__ under image_analysis_modules
    '''
    ia_module_dict = {}
    for name, cls in inspect.getmembers(importlib.import_module("image_analysis_modules"), inspect.isclass):
        print(cls)
        if cls.__bases__[0].__name__ == "BaseModule":
            ia_module_dict[cls.__name__] = {}
            ia_module_dict[cls.__name__]["func"] = cls
            ia_module_dict[cls.__name__]["description"] = cls.description
            ia_module_dict[cls.__name__]["vis"] = cls.vis
    return ia_module_dict


class IaProcess(Process):
    def __init__(self, image_q2, result_q, module_q, pause_q):
        super(IaProcess, self).__init__()
        self.image_q2 = image_q2
        self.result_q = result_q
        self.module_q = module_q
        self.pause_q = pause_q
        self.module_dict = get_ia_module_dict()
        self.module = PassThrough()
        self.module.load()

    def replace_module(self, name):
        del self.module
        # does tensorflow and keras frees gpu? may have to K.clear_session e.g.?
        self.module = self.module_dict[name]["func"]()
        self.module.load()
        while self.pause_q.qsize() != 0:
            _ = self.pause_q.get()
        self.pause_q.put(0)

    def run(self):
        # method name cannot be changed
        while True:
            if self.module_q.qsize() != 0:
                name = self.module_q.get()
                print("replacing model to:{}".format(name))
                self.replace_module(name)
                # must send a pause queue to ia to "wait for loading before visualization"

            image = self.image_q2.get()

            result = self.module.analyze(image)
            result["module_name"] = type(self.module).__name__
            self.result_q.put(result)


if __name__ == '__main__':
    camera_info = v4l2_utils.RetreiveV4L2Info()
    ia_module_dict = get_ia_module_dict()
    # pprint(camera_info.config["parameters"])
    image_q1 = Queue(maxsize=1)  # image queue for gui and saving
    image_q2 = Queue(maxsize=1)  # image queue for image analysis
    result_q1 = Queue(maxsize=1)  # result queue retreived from image analysis
    config_q1 = Queue(maxsize=1)  # gui -> camera settings control
    module_q1 = Queue(maxsize=1)
    pause_q1 = Queue(maxsize=1)

    # GUI process
    p1 = Process(target=gui_process, args=(camera_info, ia_module_dict, image_q1,
                                           result_q1, config_q1, module_q1, pause_q1))
    # camera input handling process
    p2 = Process(target=cam_process, args=(image_q1, image_q2, config_q1))

    # image analysis process
    p3 = IaProcess(image_q2, result_q1, module_q1, pause_q1)

    p1.start()
    p2.start()
    p3.start()

    # terminate p2 and p1 after p1 (GUI) is closed.
    p1.join()
    p2.terminate()
    p3.terminate()

