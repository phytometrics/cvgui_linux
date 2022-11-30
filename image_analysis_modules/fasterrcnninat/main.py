import onnxruntime
import os
import cv2
import numpy as np

#very slow. should not use now

class FasterRCNNiNAT:
    description = "Faster RCNN 600x600 iNaturalist 2854 class"
    path = os.path.dirname(__file__)
    input_size = (600,600)

    def load(self):

        # model_path = os.path.join(self.path,"fasterrcnninat/fastercnn_resnet101_inat.onnx")
        model_path = "/Users/todayousuke/Google ドライブ（yosuke@phytometrics.jp）/仕事/cameraio/image_analysis_modules/fasterrcnninat/fastercnn_resnet101_inat.onnx"
        providers = ['CPUExecutionProvider']

        self.onnx_session = onnxruntime.InferenceSession(
            model_path,
            providers=providers,
        )
        self.input_name = self.onnx_session.get_inputs()[0].name
        self.output_name = self.onnx_session.get_outputs()[0].name

    def preprocess(self, image):
        padded_image = np.ones(
            (self.input_size[0], self.input_size[1], 3), dtype=np.uint8) * 114
        self.ratio = min(self.input_size[0] / image.shape[0],
                    self.input_size[1] / image.shape[1])
        resized_image = cv2.resize(
            image,
            (int(image.shape[1] * self.ratio), int(image.shape[0] * self.ratio)),
            interpolation=cv2.INTER_LINEAR,
        )
        resized_image = resized_image.astype(np.uint8)

        padded_image[:int(image.shape[0] * self.ratio), :int(image.shape[1] *
                                                        self.ratio)] = resized_image
        return padded_image

    def analyze(self, image):
        image = self.preprocess(image)
        predictions = self.onnx_session.run(
            None,
            {self.input_name: image[None, :, :, :]},
        )
        prediction = predictions[0]
        result = {}
        result = prediction
        return result

inat = FasterRCNNiNAT()
inat.load()

image = cv2.imread("/Users/todayousuke/Google ドライブ（yosuke@phytometrics.jp）/仕事/cameraio/image_analysis_modules/fasterrcnninat/sakura.jpeg")
result = inat.analyze(image)
print(result)