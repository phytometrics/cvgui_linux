import os

class BaseModule:
    info = {}
    info["description"] = __module__
    info["reference"] = "not defined"
    info["license"] = "not defined"
    path = os.path.dirname(__file__)

    def load(self):
        pass

    def analyze(self, image):
        result = {}
        return result

    @staticmethod
    def vis(image, result):
        # use for annotation in tkinter
        return image

