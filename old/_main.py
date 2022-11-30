import tkinter as tk
import subprocess
import cv2
from PIL import Image, ImageTk
import time
from utility import v4l2_utils, tk_utils, camera_utils, general_utils
from image_analysis_modules.yolox import yolox_onnx
from pprint import pprint
import numpy as np

# https://trafalbad.hatenadiary.jp/entry/2021/10/04/075331
# https://stackoverflow.com/questions/23599087/multiprocessing-python-core-foundation-error/23982497

from multiprocessing import Process, Queue, Value

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

class StomataYoloXM:
    description = "YoloX-m Stomata Detection Model 768 x 1280"

    def load(self):
        self.yolox = yolox_onnx.YoloxONNX(model_path="../image_analysis_modules/stomata/yolox_m.onnx")

    def analyze(self, image):
        result = {}
        bboxes, scores, class_ids = self.yolox.inference(image)
        result["bboxes"] = bboxes
        result["scores"] = scores
        result["class_ids"] = class_ids

        return result


class YoloXNano:
    description = "YoloX Nano 416x416 ONNX model"
    with open('../image_analysis_modules/yolox/coco_classes.txt', 'rt') as f:
        coco_classes = f.read().rstrip('\n').split('\n')

    def load(self):
        self.yolox = yolox_onnx.YoloxONNX(model_path="../image_analysis_modules/yolox/yolox_nano.onnx")

    def analyze(self, image):
        result = {}
        bboxes, scores, class_ids = self.yolox.inference(image)
        result["bboxes"] = bboxes
        result["scores"] = scores
        result["class_ids"] = class_ids

        return result

    @staticmethod
    def vis(img, result, conf=0.5, class_names=None):
        boxes, scores, cls_ids = result["bboxes"], result["scores"], result["class_ids"]
        img = yolox_onnx.vis(img, boxes, scores, cls_ids, conf=0.5, class_names=YoloXNano.coco_classes)
        return img


class YoloXS:
    description = "YoloX-S 640x640 ONNX model"
    with open('../image_analysis_modules/yolox/coco_classes.txt', 'rt') as f:
        coco_classes = f.read().rstrip('\n').split('\n')

    def load(self):
        self.yolox = yolox_onnx.YoloxONNX(model_path="../image_analysis_modules/yolox/yolox_s.onnx",
                                          input_shape=(640, 640))

    def analyze(self, image):
        result = {}
        bboxes, scores, class_ids = self.yolox.inference(image)
        result["bboxes"] = bboxes
        result["scores"] = scores
        result["class_ids"] = class_ids
        return result

    @staticmethod
    def vis(img, result, conf=0.5, class_names=None):
        boxes, scores, cls_ids = result["bboxes"], result["scores"], result["class_ids"]
        img = yolox_onnx.vis(img, boxes, scores, cls_ids, conf=0.5, class_names=YoloXS.coco_classes)
        return img


class PyDNet2:
    """
    https://github.com/mattpoggi/pydnet/blob/master/webcam.py
    https://github.com/PINTO0309/PINTO_model_zoo/tree/main/314_PyDNet2
    """

    description = "PyDNet2 float32 640x384 ONNX model (beta)"

    def load(self):
        import onnxruntime
        model_path = "../image_analysis_modules/pydnet2/model_float32_640_384.onnx"
        providers = ['CPUExecutionProvider']
        # モデル読み込み

        self.onnx_session = onnxruntime.InferenceSession(
            model_path,
            providers=providers,
        )
        self.input_name = self.onnx_session.get_inputs()[0].name
        self.output_name = self.onnx_session.get_outputs()[0].name

    def analyze(self, image):
        import cv2

        image = cv2.resize(image, dsize=(640, 384)).astype(np.float32) / 255.
        image = image.transpose((2, 0, 1))
        predictions = self.onnx_session.run(
            None,
            {self.input_name: image[None, :, :, :]},
        )
        prediction = predictions[0]
        result = {}
        result["depth"] = prediction
        return result

    @staticmethod
    def applyColorMap(img, cmap):
        import numpy as np
        from matplotlib import cm
        import cv2

        colormap = cm.get_cmap(cmap)
        colored = colormap(img)
        return np.float32(cv2.cvtColor(np.uint8(colored * 255), cv2.COLOR_RGBA2BGR)) / 255.

    @staticmethod
    def vis(img, result):
        color_scaling = 1 / 64.

        disp = result["depth"]
        disp_color = PyDNet2.applyColorMap(disp[0, :, :, 0] * color_scaling, 'magma')
        disp_color = (disp_color * 255).astype(np.uint8)
        disp_color = cv2.resize(disp_color, dsize=(img.shape[1], img.shape[0]))
        toShow = np.concatenate((img, disp_color), 1)
        return toShow


class DeepLabV3Plus:
    """
    https://github.com/PINTO0309/PINTO_model_zoo/blob/main/026_mobile-deeplabv3-plus/07_openvino/deeplabv3plus_usbcam.py
    """

    description = "DeepLabV3Plus Human Segmentation 513x513 OpenVINO model"

    def load(self):
        from openvino.inference_engine import IENetwork, IECore

        self.DEEPLAB_PALETTE = Image.open("../image_analysis_modules/deeplabv3plus/colorpalette.png").getpalette()
        model_xml = "image_analysis_modules/deeplabv3plus/513_513/deeplab_v3_plus_mnv2_decoder_513.xml"
        model_bin = "image_analysis_modules/deeplabv3plus/513_513/deeplab_v3_plus_mnv2_decoder_513.bin"
        device = "CPU"
        ie = IECore()
        net = ie.read_network(model_xml, model_bin)
        input_info = net.input_info
        self.input_blob = next(iter(input_info))
        self.exec_net = ie.load_network(network=net, device_name=device)

    def analyze(self, image):
        import cv2
        height, width, _ = image.shape
        prepimg_deep = cv2.resize(image, (513, 513))
        # prepimg_deep = cv2.cvtColor(prepimg_deep, cv2.COLOR_BGR2RGB)
        prepimg_deep = np.expand_dims(prepimg_deep, axis=0)
        prepimg_deep = prepimg_deep.astype(np.float32)
        prepimg_deep = np.transpose(prepimg_deep, [0, 3, 1, 2])
        prepimg_deep -= 127.5
        prepimg_deep /= 127.5

        # Run model - DeeplabV3-plus
        deeplabv3_predictions = self.exec_net.infer(inputs={self.input_blob: prepimg_deep})

        # Get results
        predictions = deeplabv3_predictions['Output/Transpose']
        # Segmentation
        outputimg = np.uint8(predictions[0][0])
        outputimg = cv2.resize(outputimg, (width, height))
        outputimg = Image.fromarray(outputimg, mode="P")
        outputimg.putpalette(self.DEEPLAB_PALETTE)
        outputimg = outputimg.convert("RGB")
        outputimg = np.asarray(outputimg)
        result = {}
        result["mask"] = outputimg
        return result

    @staticmethod
    def vis(img, result):
        return cv2.addWeighted(img, 0.5, result["mask"], 0.5, 0)


class DeepLabV3:
    """
    https://docs.openvino.ai/latest/omz_models_model_deeplabv3.html
    https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/segmentation_demo/python/segmentation_demo.py
    https://github.com/openvinotoolkit/open_model_zoo/blob/master/data/palettes/pascal_voc_21cl_colors.txt
    """

    description = "DeepLabV3 513x513 FP16 Pascal VOC OpenVINO model"

    pascalvoc_color = [
        (0, 0, 0),  # background
        (128, 0, 0),  # aeroplane
        (0, 128, 0),  # bicycle
        (128, 128, 0),  # bird
        (0, 0, 128),  # boat
        (128, 0, 128),  # bottle
        (0, 128, 128),  # bus
        (128, 128, 128),  # car
        (64, 0, 0),  # cat
        (192, 0, 0),  # chair
        (64, 128, 0),  # cow
        (192, 128, 0),  # diningtable
        (64, 0, 128),  # dog
        (192, 0, 128),  # horse
        (64, 128, 128),  # motorbike
        (192, 128, 128),  # person
        (0, 64, 0),  # pottedplant
        (128, 64, 0),  # sheep
        (0, 192, 0),  # sofa
        (128, 192, 0),  # train
        (0, 64, 128)  # tvmonitor
    ]
    # for 0-1 coloring
    pascalvoc_color = [tuple(y / 255. for y in list(x)) for x in pascalvoc_color]

    def load(self):
        from openvino.inference_engine import IENetwork, IECore

        # self.DEEPLAB_PALETTE = Image.open("image_analysis_modules/deeplabv3/colorpalette.png").getpalette()
        model_xml = "image_analysis_modules/deeplabv3/FP16/deeplabv3.xml"
        model_bin = "image_analysis_modules/deeplabv3/FP16/deeplabv3.bin"
        device = "CPU"
        ie = IECore()
        net = ie.read_network(model_xml, model_bin)
        input_info = net.input_info
        self.input_blob = next(iter(input_info))
        self.exec_net = ie.load_network(network=net, device_name=device)

    def analyze(self, image):
        import cv2
        import numpy as np
        from PIL import Image
        from skimage.color import label2rgb

        height, width, _ = image.shape
        prepimg_deep = cv2.resize(image, (513, 513))
        # prepimg_deep = cv2.cvtColor(prepimg_deep, cv2.COLOR_BGR2RGB)
        prepimg_deep = np.expand_dims(prepimg_deep, axis=0)
        # prepimg_deep = prepimg_deep.astype(np.float32)
        prepimg_deep = np.transpose(prepimg_deep, [0, 3, 1, 2])

        deeplabv3_predictions = self.exec_net.infer(inputs={self.input_blob: prepimg_deep})
        self.prediction = deeplabv3_predictions['ArgMax/Squeeze'][0]
        # self.prediction = label2rgb(self.prediction,colors=self.pascalvoc_color)
        # overlay = cv2.addWeighted(image, 0.5, self.prediction, 0.5, 0)

        result = {}
        result["mask"] = self.prediction
        return result

    @staticmethod
    def vis(img, result):
        from skimage.color import label2rgb
        mask = result["mask"]
        mask = label2rgb(mask, colors=DeepLabV3.pascalvoc_color)
        mask = (mask * 255).astype(np.uint8)
        mask = cv2.resize(mask, dsize=(img.shape[1], img.shape[0]))
        overlay = cv2.addWeighted(img, 0.5, mask, 0.5, 0)
        return overlay


class Nothing:
    description = "Select Image Analysis Module"

    def load(self):
        pass

    def analyze(self, image):
        result = {}
        return result

    @staticmethod
    def vis(image, result):
        # use for annotation in tkinter
        return image


def get_ia_module_dict():
    # module_list should be automatically retreived
    ia_module_list = [
        Nothing,
        YoloXNano,
        YoloXS,
        PyDNet2,
        DeepLabV3Plus,
        DeepLabV3
    ]
    ia_module_dict = {}
    for cls in ia_module_list:
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
        self.module = Nothing()
        self.module.load()

    def replace_module(self, name):
        del self.module
        # does tensorflow and keras frees gpu? may have to K.clear_session e.g.?
        self.module = self.module_dict[name]["func"]()
        self.module.load()
        while self.pause_q.qsize() != 0:
            _ = self.pause_q.get()
        self.pause_q.put(0)

    def run(self):  # this name cannot be changed
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


# def ia_process(image_q, result_q, module_q):
#     if module_q.qsize() != 0:
#         module_name = module_q.get()
#         # moduleの名前からfunctionをロード
#         selected_class = yoloxNano
#     #yolox
#     core_module = selected_class()
#     while True:
#         image = image_q.get()
#         core_module.loop(image)
#

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

    # run = Value("i", 1)  # 0 pause, 1 run

    p1 = Process(target=gui_process, args=(camera_info, ia_module_dict, image_q1,
                                           result_q1, config_q1, module_q1, pause_q1))
    p2 = Process(target=cam_process, args=(image_q1, image_q2, config_q1))
    p3 = IaProcess(image_q2, result_q1, module_q1, pause_q1)
    p1.start()
    p2.start()
    p3.start()
    p1.join()
    p2.terminate()
    p3.terminate()
    # terminate cam_loop when tkinter window is closed
