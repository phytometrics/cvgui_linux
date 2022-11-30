import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.375
        self.input_size = (416, 416)
        self.mosaic_scale = (0.5, 1.5)
        self.random_size = (10, 20)
        self.test_size = (416, 416)
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.enable_mixup = False
        self.num_classes= 2854 
        self.save_history_ckpt = False
        self.max_epoch = 300
        self.data_dir = "/home2/inat2017/YOLOX/dataset/"
        self.train_ann = "train_2017_bboxes_reindex.json"
        self.val_ann = "val_2017_bboxes_reindex.json"
        self.test_ann = "val_2017_bboxes_reindex.json"
        self.eval_interval = 1
        self.data_num_workers = 1
