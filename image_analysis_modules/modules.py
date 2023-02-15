import cv2
import os
from PIL import Image
from skimage.color import label2rgb
from skimage.measure import label, regionprops, find_contours
from skimage.draw import polygon
from skimage.transform import resize
from scipy.special import expit
import numpy as np
from matplotlib import cm


from .yolox import yolox_onnx
from .iseed import yolov5_utils
from .base import BaseModule
from utility.general_utils import make_grid

try:
    import torch
except Exception as e:
    print("pytorch not installed. ignoring")
    print(e)
try:
    from pycoral.utils.edgetpu import make_interpreter
    from pycoral.adapters import classify, common, segment
    from pycoral.utils.dataset import read_label_file
    import tflite_runtime.interpreter as tflite
    import platform
except Exception as e:
    print("pycoral libraries must be installed to use tpu")
    print(e)
try:
    import onnxruntime
except Exception as e:
    print("onnxruntime is not installed")

class PassThrough(BaseModule):
    description = "Do nothing" \

    def load(self):
        pass

    def analyze(self, image):
        result = {}
        return result

    @staticmethod
    def vis(image, result):
        # use for annotation in tkinter
        return image

class DeepLabSlimTPU(BaseModule):
    description = "deep lab slim cityscape in tpu"
    path = os.path.dirname(__file__)
    model_file = os.path.join(path, 'deeplabslimtpu/deeplabv3_mnv2_pascal_quant_edgetpu.tflite')

    def create_pascal_label_colormap(self):
        """Creates a label colormap used in PASCAL VOC segmentation benchmark.
        Returns:
          A Colormap for visualizing segmentation results.
        """
        colormap = np.zeros((256, 3), dtype=int)
        indices = np.arange(256, dtype=int)

        for shift in reversed(range(8)):
            for channel in range(3):
                colormap[:, channel] |= ((indices >> channel) & 1) << shift
            indices >>= 3

        return colormap

    def label_to_color_image(self,label):

        """Adds color defined by the dataset colormap to the label.
        Args:
          label: A 2D array with integer type, storing the segmentation label.
        Returns:
          result: A 2D array with floating type. The element of the array
            is the color indexed by the corresponding element in the input label
            to the PASCAL color map.
        Raises:
          ValueError: If label is not of rank 2 or its value is larger than color
            map maximum entry.
        """
        if label.ndim != 2:
            raise ValueError('Expect 2-D input label')

        colormap = self.create_pascal_label_colormap()

        if np.max(label) >= len(colormap):
            raise ValueError('label value too large.')

        return colormap[label]

    def load(self):
        self.interpreter = make_interpreter(self.model_file)
        self.interpreter.allocate_tensors()

    def analyze(self, image):
        result = {}
        h, w, c = image.shape
        image = cv2.resize(image, dsize=(513, 513))
        # mask = Image.fromarray(mask)
        # common.set_input(self.interpreter, mask)
        common.input_tensor(self.interpreter)[:, :] = image
        self.interpreter.invoke()
        pred = segment.get_output(self.interpreter)
        if len(pred.shape) == 3:
            pred = np.argmax(pred, axis=-1)
        pred = cv2.resize(pred, dsize=(w, h))
        pred = self.label_to_color_image(pred).astype(np.uint8)
        result["pred"] = pred
        return result

    @staticmethod
    def vis(image, result):
        orig_image = image.copy()
        image = cv2.resize(image, dsize=(640, 480))
        mask = result["pred"]
        images = []
        images.append(image)
        images.append(mask)
        images = np.array(images)

        image = make_grid(images, nrow=2,padding=1)
        return image


class Coleochaete(BaseModule):

    description = "Coleochaete anomaly analysis"
    reference = ["https://github.com/danielgatis/rembg/tree/main/rembg","https://github.com/xavysp/LDC/blob/main/main.py"]
    path = os.path.dirname(__file__)
    model_path1 = os.path.join(path, 'coleochaete/u2netp.onnx')
    model_path2 = os.path.join(path, 'coleochaete/LDC_BIPED_480x400.onnx')

    def image_normalization(self, img, img_min=0, img_max=255,
                            epsilon=1e-12):
        """This is a typical image normalization function
        where the minimum and maximum of the image is needed
        source: https://en.wikipedia.org/wiki/Normalization_(image_processing)
        :param img: an image could be gray scale or color
        :param img_min:  for default is 0
        :param img_max: for default is 255
        :return: a normalized image, if max is 255 the dtype is uint8
        """

        img = np.float32(img)
        # whenever an inconsistent image
        img = (img - np.min(img)) * (img_max - img_min) / \
              ((np.max(img) - np.min(img)) + epsilon) + img_min
        return img

    def count_parameters(self, model=None):
        if model is not None:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        else:
            print("Error counting model parameters line 32 img_processing.py")
            raise NotImplementedError

    def load(self):
        providers = ['CPUExecutionProvider']
        self.onnx_session1 = onnxruntime.InferenceSession(
            self.model_path1,
            providers=providers,
        )
        self.input_name1 = self.onnx_session1.get_inputs()[0].name
        self.output_name1 = self.onnx_session1.get_outputs()[0].name

        self.onnx_session2 = onnxruntime.InferenceSession(
            self.model_path2,
            providers=providers,
        )
        self.input_name2 = self.onnx_session2.get_inputs()[0].name
        # self.output_names2 = self.onnx_session2.get_outputs()[0].name
        self.output_names2 = [x.name for x in self.onnx_session2.get_outputs()]


    def analyze(self, image):
        result = {}
        result["prop"] = {}

        # https://github.com/danielgatis/rembg/tree/main/rembg
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        # ldcの方は引くだけ
        orig_image = image.copy()
        h,w,c = image.shape

        image = cv2.resize(image, dsize=(320,320))
        image = image.astype(np.float32)
        image = image / np.max(image)
        image[:, :, 0] = (image[:, :, 0] - mean[0]) / std[0]
        image[:, :, 1] = (image[:, :, 1] - mean[1]) / std[1]
        image[:, :, 2] = (image[:, :, 2] - mean[2]) / std[2]
        image = image.transpose((2, 0, 1))
        # end of preprocess

        pred = self.onnx_session1.run(
                        None,
                        {self.input_name1: image[None, :, :, :]},
                    )[0][0][0]
        pred = resize(pred, (h,w), preserve_range=True)
        # pred = cv2.resize(pred, dsize=(w, h))
        pred = pred > 0.5
        label_image = label(pred)
        pred = pred.astype(np.float32)
        pred *= 255
        pred = pred.astype(np.uint8)
        pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
        #pred = cv2.resize(pred, dsize=(w,h))
        result["pred"] = pred
        # get contour
        props = regionprops(label_image)
        if props:
            idx = np.argmax([x.area for x in props])
            prop = props[idx]
            #contour = find_contours(mask, 0.5)[0]
            # propそのものを入れるとバグる
            result["prop"]["solidity"] = prop.solidity
            result["prop"]["centroid"] = prop.centroid  # r,c
            result["prop"]["bbox"] = prop.bbox  # (min_row, min_col, max_row, max_col)
            # get image with only the largest area
            mask = label_image == prop.label
            mask = mask.astype(np.float32)
            mask *= 255
            mask = mask.astype(np.uint8)
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            #mask = cv2.resize(mask, dsize=(w, h))

            result["mask"] = mask

            # create polar2linear
            # https://qiita.com/Kazuhito/items/065fd25d56c0238a1d72
            # キュービック補間 + 外れ値塗りつぶし + 極座標へリニアマッピング
            flags = cv2.INTER_CUBIC + cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LINEAR
            # 引き数：画像, 変換後サイズ(幅、高さ)、中心座標(X座標、Y座標)、半径、変換フラグ
            #
            r = np.linalg.norm(prop.image.shape)/2
            polar = cv2.warpPolar(orig_image,(orig_image.shape[0],orig_image.shape[1]),(prop.centroid[1], prop.centroid[0]),r,flags)
            #rotate 90
            polar = polar.transpose(1,0,2)[::-1]
            result["polar"] = polar


            #LDC inference
            # image = orig_image.copy()[prop.bbox[0]:prop.bbox[2],prop.bbox[1]:prop.bbox[3]]
            #creat a mask image and then infer to suppress background noise


            image = orig_image.copy()
            image = cv2.resize(image, dsize=(480,400))
            image = image.astype(np.float32)
            image -= [160.913,160.275,162.239]#,137.86]

            image = image.transpose((2, 0, 1))

            #use only the last layer
            edge = self.onnx_session2.run(
                [self.output_names2[-1]],
                {self.input_name2: image[None, :, :, :]})[0][0][0]
            edge = expit(edge)
            edge = np.uint8(self.image_normalization(edge))
            edge = cv2.resize(edge, (w,h))
            edge = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
            edge = cv2.bitwise_and(edge, mask)
            result["edge"] = edge

            flags = cv2.INTER_CUBIC + cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LINEAR
            # 引き数：画像, 変換後サイズ(幅、高さ)、中心座標(X座標、Y座標)、半径、変換フラグ
            #
            r = np.linalg.norm(edge.shape) / 2
            polar_edge = cv2.warpPolar(edge, (orig_image.shape[0], orig_image.shape[1]),
                                  (prop.centroid[1], prop.centroid[0]), r, flags)
            # rotate 90
            polar_edge = polar_edge.transpose(1, 0, 2)[::-1]
            result["polar_edge"] = polar_edge


            # averageをここから計算できるけれど、最後の層だけ使うのがが単純に一番よさそう。
            # preds2 = self.onnx_session2.run(
            #     self.output_names2,
            #     {self.input_name2: image[None, :, :, :]},
            # )  # outputs, batch, c, h, w
            #
            # edge_maps = []
            # for pred2 in preds2:
            #     tmp = pred2[0][0]
            #     tmp = expit(pred2)
            #     edge_maps.append(tmp)
            # edge_maps = np.array(edge_maps)
            #
            # edge_result = []
            # fuse_num = edge_maps.shape[0]

            # i_shape = tmp.shape[:2][::-1]
            # for i in range(edge_maps):
            #     tmp_img = tmp[i]
            #     tmp_img = np.uint8(self.image_normalization(tmp_img))
            #     tmp_img = cv2.bitwise_not(tmp_img)
            #     if not tmp_img.shape[1] == i_shape[0] or not tmp_img.shape[0] == i_shape[1]:
            #         tmp_img = cv2.resize(tmp_img, (i_shape[0], i_shape[1]))
            #     edge_result.append(tmp_img)
            #
            #     if i == fuse_num:
            #         # print('fuse num',tmp.shape[0], fuse_num, i)
            #         _fuse = tmp_img
            #         _fuse = _fuse.astype(np.uint8)






            # print(len(pred2))
            # print(pred2.shape)
            # pred2 = resize(pred2, (h, w), preserve_range=True)
            # pred2 *=255.
            # pred2 = pred2.astype(np.uint8)
            # pred2 = cv2.cvtColor(pred2, cv2.COLOR_GRAY2BGR)
            # result["edge"] = pred2
        return result

    @staticmethod
    def vis(image, result):

        images = []
        images.append(image)
        images.append(result["pred"])

        if "mask" in result.keys():
            mask = result["mask"]
            images.append(mask)
            if "solidity" in result["prop"].keys():
                if result["prop"]["solidity"]>0.95:
                    color = (255, 0, 0)
                else:
                    color = (0, 0, 255)
                cv2.putText(mask,
                            org=(10,200),
                            color=color,
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=3.0,
                            text="Sol.:" + str(np.round(result["prop"]["solidity"],decimals=2)),
                            thickness=2,
                            )
            if "centroid" in result["prop"].keys():
                center = [int(x) for x in result["prop"]["centroid"]]
                radius = 10
                cv2.circle(mask, (center[1],center[0]), radius, (0,255,0), thickness=1, lineType=cv2.LINE_8, shift=0)
            if "bbox" in result["prop"].keys():
                bbox = [int(x) for x in result["prop"]["bbox"]]
                cv2.rectangle(mask, (bbox[1], bbox[0]), (bbox[3], bbox[2]), (0, 255, 0))

            images.append(result["polar"])
            images.append(result["edge"])
            images.append(result["polar_edge"])


        try:
            images = np.array(images)
            n = -(-len(images)//2)
            image = make_grid(images,nrow=2,padding=1)
        except:
            pass
        return image

class Coleochaete2(BaseModule):

    description = "Coleochaete anomaly analysis2"
    path = os.path.dirname(__file__)
    model_file = os.path.join(path, 'coleochaete/u2netp_256x256_full_integer_quant_edgetpu.tflite')


    def load(self):
        self.interpreter = make_interpreter(self.model_file)
        self.interpreter.allocate_tensors()

    def analyze(self, image):
        result = {}
        # bboxes, scores, class_ids = self.yolox_module.inference(image)
        # result["bboxes"] = bboxes
        # result["scores"] = scores
        # result["class_ids"] = class_ids
        h,w,c = image.shape
        image = cv2.resize(image, dsize=(256,256))
        # mask = Image.fromarray(mask)
        # common.set_input(self.interpreter, mask)
        common.input_tensor(self.interpreter)[:, :] = image
        self.interpreter.invoke()
        pred = segment.get_output(self.interpreter)
        pred = pred[...,0] # 320,320,1 -> 320
        pred = cv2.resize(pred, dsize=(w, h))
        pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)

        result["pred"] = pred

        return result

    @staticmethod
    def vis(image, result):
        orig_image  = image.copy()
        #image = cv2.resize(image, dsize=(, 256))
        mask = result["pred"]
        images = []
        images.append(image)
        images.append(mask)
        images = np.array(images)

        image = make_grid(images, nrow=2,padding=1)
        return image

class YoloXNano(BaseModule):

    description = "YoloX Nano 416x416 ONNX model"
    path = os.path.dirname(__file__)
    with open(os.path.join(path, 'yolox/coco_classes.txt'), 'rt') as f:
        coco_classes = f.read().rstrip('\n').split('\n')

    def load(self):
        self.yolox_module = yolox_onnx.YoloxONNX(model_path=os.path.join(self.path, "yolox/yolox_nano.onnx"))

    def analyze(self, image):
        result = {}
        bboxes, scores, class_ids = self.yolox_module.inference(image)
        result["bboxes"] = bboxes
        result["scores"] = scores
        result["class_ids"] = class_ids

        return result

    @staticmethod
    def vis(img, result, conf=0.5):
        boxes, scores, cls_ids = result["bboxes"], result["scores"], result["class_ids"]
        # img = YoloXNano.yolox_module.vis(img, boxes, scores, cls_ids, conf=0.5, class_names=YoloXNano.coco_classes)
        img = yolox_onnx.annotate(img, boxes, scores, cls_ids, conf=conf, class_names=YoloXNano.coco_classes)
        return img

class YoloXS(BaseModule):
    description = "YoloX-S 640x640 ONNX model"

    path = os.path.dirname(__file__)
    with open(os.path.join(path, 'yolox/coco_classes.txt'), 'rt') as f:
        coco_classes = f.read().rstrip('\n').split('\n')

    def load(self):
        self.yolox_module = yolox_onnx.YoloxONNX(model_path=os.path.join(self.path,"yolox/yolox_s.onnx"),
                                          input_shape=(640, 640))

    def analyze(self, image):
        result = {}
        bboxes, scores, class_ids = self.yolox_module.inference(image)
        result["bboxes"] = bboxes
        result["scores"] = scores
        result["class_ids"] = class_ids
        return result

    @staticmethod
    def vis(img, result, conf=0.5):
        boxes, scores, cls_ids = result["bboxes"], result["scores"], result["class_ids"]
        img = yolox_onnx.annotate(img, boxes, scores, cls_ids, conf=conf, class_names=YoloXS.coco_classes)
        return img

class ArabidopsisStomataQuantification(BaseModule):
    description = "Arabidopsis Stomata Detection Model 1280 x 1280"
    path = os.path.dirname(__file__)

    def load(self):
        self.yolox_module = yolox_onnx.YoloxONNX(model_path=os.path.join(self.path,"stomata/ara_yolox_s_1280.onnx"),
                                          input_shape=(1280, 1280),class_score_th=0.01)
    def analyze(self, image):
        result = {}
        bboxes, scores, class_ids = self.yolox_module.inference(image)
        result["bboxes"] = bboxes
        result["scores"] = scores
        result["class_ids"] = class_ids
        return result

    @staticmethod
    def vis(img, result, conf=0.01):
        boxes, scores, cls_ids = result["bboxes"], result["scores"], result["class_ids"]
        img = yolox_onnx.annotate(img, boxes, scores, cls_ids, conf=conf, class_names=["open","close"])
        return img

class iSeed(BaseModule):
    description = "iSeed Counter 640x640 yolov5 onnx"
    references = "https://github.com/ultralytics/yolov5/blob/4d8d84b0ea7147aca64e7c38ce1bdb5fbb9c5a53/utils/general.py#L769"
    path = os.path.dirname(__file__)
    model_path = os.path.join(path,"iseed","best.onnx")
    label_path = os.path.join(path, "iseed", "label.txt")

    def load(self):
        # torch hub prob does not work in multiprocess
        #self.model = torch.hub.load('ultralytics/yolov5', 'custom', self.model_file)

        providers = ['CPUExecutionProvider']
        self.onnx_session = onnxruntime.InferenceSession(
            self.model_path,
            providers=providers,
        )
        self.input_name = self.onnx_session.get_inputs()[0].name
        self.output_name = self.onnx_session.get_outputs()[0].name

    def analyze(self, image):
        result = {}

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image, ratio = yolov5_utils.preprocess(image, (640,640))
        image /= 255.
        # Run an inference
        preds = self.onnx_session.run(
                        None,
                        {self.input_name: image[None, :, :, :]},
                    )[0]
        if len(preds):
            preds = yolov5_utils.non_max_suppression(preds,conf_thres=0.3, iou_thres=0.45, max_det=2000)[0]
            # to original scale
            preds[:, :4] = preds[:, :4] / ratio
            result["preds"] = preds
        else:
            preds = None
            result["preds"] = []
        return result

    @staticmethod
    def vis(image, result):
        if len(result["preds"])>0:
            for pred in result["preds"]:
                xmin, ymin, xmax, ymax, conf, cls = pred
                # cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)),
                #               color=(255, 0, 0), thickness=1,
                # )
                cv2.circle(image, (int(np.average([xmin,xmax])), int(np.average([ymin,ymax]))), 5, (255, 0, 0), thickness=-1)

        cv2.putText(image,
                    text="Seeds:"+str(len(result["preds"])),
                    org=(0, 100),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.0,
                    color=(0, 0, 0),
                    thickness=2,
                    lineType=cv2.LINE_4)

        return image


class iNaturalistBirdClassification:
    description = "iNaturalist Bird Classification edge coral TPU"
    path = os.path.dirname(__file__)
    #info["reference"] = "https://github.com/google-coral/pycoral/blob/master/examples/classify_image.py"

    model_path = os.path.join(path, "inatbird", "mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite")
    label_path = os.path.join(path, "inatbird", "inat_bird_labels.txt")

    def load_edgetpu_delegate(self,options=None):
        _EDGETPU_SHARED_LIB = {
            'Linux': 'libedgetpu.so.1',
            'Darwin': 'libedgetpu.1.dylib',
            'Windows': 'edgetpu.dll'
        }[self.platform.system()]
        return self.tflite.load_delegate(_EDGETPU_SHARED_LIB, options or {})

    def make_interpreter(self,device=None):
        delegates = [self.load_edgetpu_delegate({'device': device} if device else {})]
        return self.tflite.Interpreter(model_path=self.model_path, experimental_delegates=delegates)

    def load(self):
        self.interpreter = self.make_interpreter()
        self.interpreter.allocate_tensors()
        self.size = self.common.input_size(self.interpreter)
        self.labels = self.read_label_file(self.label_path)

        self.params = self.common.input_details(self.interpreter, 'quantization_parameters')
        self.scale = self.params['scales']
        self.zero_point = self.params['zero_points']
        self.mean = 128.0  # args.input_mean
        self.std = 128.0  # args.input_std

    def analyze(self, image, topk=3, threshold=0):
        image = cv2.resize(image, self.size)
        if abs(self.scale * self.std - 1) < 1e-5 and abs(self.mean - self.zero_point) < 1e-5:
            # Input data does not require preprocessing.
            self.common.set_input(self.interpreter, image)
        else:
            # Input data requires preprocessing
            normalized_input = (np.asarray(image) - self.mean) / (self.std * self.scale) + self.zero_point
            np.clip(normalized_input, 0, 255, out=normalized_input)
            self.common.set_input(self.interpreter, normalized_input.astype(np.uint8))

        self.interpreter.invoke()
        classes = self.classify.get_classes(self.interpreter, top_k, threshold)

        result = {}
        result["classes"] = classes
        return result
    @staticmethod
    def vis(image, result):
        y = 50
        for cls in result["classes"]:

            cv2.putText(image,
                        text='sample text',
                        org=(100, y),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1.0,
                        color=(0, 255, 0),
                        thickness=2,
                        lineType=cv2.LINE_4)
            y += 50
        return image

# class PyDNet2:
#     """
#     https://github.com/mattpoggi/pydnet/blob/master/webcam.py
#     https://github.com/PINTO0309/PINTO_model_zoo/tree/main/314_PyDNet2
#     """
#
#     description = "PyDNet2 float32 640x384 ONNX model (beta)"
#
#     def load(self):
#         model_path = "image_analysis_modules/pydnet2/model_float32_640_384.onnx"
#         providers = ['CPUExecutionProvider']
#         # モデル読み込み
#
#         self.onnx_session = onnxruntime.InferenceSession(
#             model_path,
#             providers=providers,
#         )
#         self.input_name = self.onnx_session.get_inputs()[0].name
#         self.output_name = self.onnx_session.get_outputs()[0].name
#
#     def analyze(self, image):
#         image = cv2.resize(image, dsize=(640, 384)).astype(np.float32) / 255.
#         image = image.transpose((2, 0, 1))
#         predictions = self.onnx_session.run(
#             None,
#             {self.input_name: image[None, :, :, :]},
#         )
#         prediction = predictions[0]
#         result = {}
#         result["depth"] = prediction
#         return result
#
#     @staticmethod
#     def applyColorMap(img, cmap):
#
#         colormap = cm.get_cmap(cmap)
#         colored = colormap(img)
#         return np.float32(cv2.cvtColor(np.uint8(colored * 255), cv2.COLOR_RGBA2BGR)) / 255.
#
#     @staticmethod
#     def vis(img, result):
#         color_scaling = 1 / 64.
#
#         disp = result["depth"]
#         disp_color = PyDNet2.applyColorMap(disp[0, :, :, 0] * color_scaling, 'magma')
#         disp_color = (disp_color * 255).astype(np.uint8)
#         disp_color = cv2.resize(disp_color, dsize=(img.shape[1], img.shape[0]))
#         toShow = np.concatenate((img, disp_color), 1)
#         return toShow
#

# class DeepLabV3Plus:
#     """
#     https://github.com/PINTO0309/PINTO_model_zoo/blob/main/026_mobile-deeplabv3-plus/07_openvino/deeplabv3plus_usbcam.py
#     """
#
#     description = "DeepLabV3Plus Human Segmentation 513x513 OpenVINO model"
#
#     def load(self):
#         from openvino.inference_engine import IENetwork, IECore
#
#         self.DEEPLAB_PALETTE = Image.open("image_analysis_modules/deeplabv3plus/colorpalette.png").getpalette()
#         model_xml = "image_analysis_modules/deeplabv3plus/513_513/deeplab_v3_plus_mnv2_decoder_513.xml"
#         model_bin = "image_analysis_modules/deeplabv3plus/513_513/deeplab_v3_plus_mnv2_decoder_513.bin"
#         device = "CPU"
#         ie = IECore()
#         net = ie.read_network(model_xml, model_bin)
#         input_info = net.input_info
#         self.input_blob = next(iter(input_info))
#         self.exec_net = ie.load_network(network=net, device_name=device)
#
#     def analyze(self, image):
#         height, width, _ = image.shape
#         prepimg_deep = cv2.resize(image, (513, 513))
#         # prepimg_deep = cv2.cvtColor(prepimg_deep, cv2.COLOR_BGR2RGB)
#         prepimg_deep = np.expand_dims(prepimg_deep, axis=0)
#         prepimg_deep = prepimg_deep.astype(np.float32)
#         prepimg_deep = np.transpose(prepimg_deep, [0, 3, 1, 2])
#         prepimg_deep -= 127.5
#         prepimg_deep /= 127.5
#
#         # Run model - DeeplabV3-plus
#         deeplabv3_predictions = self.exec_net.infer(inputs={self.input_blob: prepimg_deep})
#
#         # Get results
#         predictions = deeplabv3_predictions['Output/Transpose']
#         # Segmentation
#         outputimg = np.uint8(predictions[0][0])
#         outputimg = cv2.resize(outputimg, (width, height))
#         outputimg = Image.fromarray(outputimg, mode="P")
#         outputimg.putpalette(self.DEEPLAB_PALETTE)
#         outputimg = outputimg.convert("RGB")
#         outputimg = np.asarray(outputimg)
#         result = {}
#         result["mask"] = outputimg
#         return result
#
#     @staticmethod
#     def vis(img, result):
#         return cv2.addWeighted(img, 0.5, result["mask"], 0.5, 0)


# class DeepLabV3:
#     """
#     https://docs.openvino.ai/latest/omz_models_model_deeplabv3.html
#     https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/segmentation_demo/python/segmentation_demo.py
#     https://github.com/openvinotoolkit/open_model_zoo/blob/master/data/palettes/pascal_voc_21cl_colors.txt
#     """
#
#     description = "DeepLabV3 513x513 FP16 Pascal VOC OpenVINO model"
#
#     pascalvoc_color = [
#         (0, 0, 0),  # background
#         (128, 0, 0),  # aeroplane
#         (0, 128, 0),  # bicycle
#         (128, 128, 0),  # bird
#         (0, 0, 128),  # boat
#         (128, 0, 128),  # bottle
#         (0, 128, 128),  # bus
#         (128, 128, 128),  # car
#         (64, 0, 0),  # cat
#         (192, 0, 0),  # chair
#         (64, 128, 0),  # cow
#         (192, 128, 0),  # diningtable
#         (64, 0, 128),  # dog
#         (192, 0, 128),  # horse
#         (64, 128, 128),  # motorbike
#         (192, 128, 128),  # person
#         (0, 64, 0),  # pottedplant
#         (128, 64, 0),  # sheep
#         (0, 192, 0),  # sofa
#         (128, 192, 0),  # train
#         (0, 64, 128)  # tvmonitor
#     ]
#     # for 0-1 coloring
#     pascalvoc_color = [tuple(y / 255. for y in list(x)) for x in pascalvoc_color]
#
#     def load(self):
#         from openvino.inference_engine import IENetwork, IECore
#
#         # self.DEEPLAB_PALETTE = Image.open("image_analysis_modules/deeplabv3/colorpalette.png").getpalette()
#         model_xml = "image_analysis_modules/deeplabv3/FP16/deeplabv3.xml"
#         model_bin = "image_analysis_modules/deeplabv3/FP16/deeplabv3.bin"
#         device = "CPU"
#         ie = IECore()
#         net = ie.read_network(model_xml, model_bin)
#         input_info = net.input_info
#         self.input_blob = next(iter(input_info))
#         self.exec_net = ie.load_network(network=net, device_name=device)
#
#     def analyze(self, image):
#         height, width, _ = image.shape
#         prepimg_deep = cv2.resize(image, (513, 513))
#         # prepimg_deep = cv2.cvtColor(prepimg_deep, cv2.COLOR_BGR2RGB)
#         prepimg_deep = np.expand_dims(prepimg_deep, axis=0)
#         # prepimg_deep = prepimg_deep.astype(np.float32)
#         prepimg_deep = np.transpose(prepimg_deep, [0, 3, 1, 2])
#
#         deeplabv3_predictions = self.exec_net.infer(inputs={self.input_blob: prepimg_deep})
#         self.prediction = deeplabv3_predictions['ArgMax/Squeeze'][0]
#         # self.prediction = label2rgb(self.prediction,colors=self.pascalvoc_color)
#         # overlay = cv2.addWeighted(image, 0.5, self.prediction, 0.5, 0)
#
#         result = {}
#         result["mask"] = self.prediction
#         return result
#
#     @staticmethod
#     def vis(img, result):
#         mask = result["mask"]
#         mask = label2rgb(mask, colors=DeepLabV3.pascalvoc_color)
#         mask = (mask * 255).astype(np.uint8)
#         mask = cv2.resize(mask, dsize=(img.shape[1], img.shape[0]))
#         overlay = cv2.addWeighted(img, 0.5, mask, 0.5, 0)
#         return overlay
#
