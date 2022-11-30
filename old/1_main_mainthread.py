import tkinter as tk
import subprocess
import cv2
from PIL import Image, ImageTk

from utility import v4l2_utils, camera_utils, general_utils, tk_utils

def update():
    global image
    # variable frame should be the original image retreived from videocapture.read. BGR numpy array

    ret, frame = app.cam.get_frame()
    if ret:
        image = frame.copy()
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # bboxes, scores, class_ids = yolox.inference(image)
        #
        # image = yolox_onnx.vis(image, bboxes, scores, class_ids, conf=0.5, class_names=coco_classes)

        input_size = (app.image_frame.winfo_width(), app.image_frame.winfo_height())
        image = cv2.resize(image, dsize=input_size)
        image = ImageTk.PhotoImage(Image.fromarray(image))
        app.image_frame.create_image(0, 0, image=image, anchor="nw")

    # s = "FPS: " + str(fps())
    # app.image_frame.create_text(10, 10, anchor="nw", text=s, fill="red", font=('', '30', ''))

    app.after(1, update)


if __name__ == '__main__':
    fps = general_utils.FPS()
    #
    # yolox = yolox_onnx.YoloxONNX(model_path="image_analysis_modules/yolox_s/yolox_s.onnx")
    # with open('image_analysis_modules/yolox_s/coco_classes.txt', 'rt') as f:
    #     coco_classes = f.read().rstrip('\n').split('\n')

    camera_dict = v4l2_utils.get_devices()
    camera_dict = v4l2_utils.get_resolution(camera_dict)
    _cam = camera_utils.UVCCamera(camera_dict=camera_dict)
    app = tk_utils.App(camera=_cam)

    update()
    app.mainloop()

