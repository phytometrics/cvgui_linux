import numpy as np
import cv2
import time
# https://teratail.com/questions/375733?sort=3

def preprocess(image, input_size, swap=(2, 0, 1)):
    # from yolox preprocesss
    if len(image.shape) == 3:
        padded_image = np.ones(
            (input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_image = np.ones(input_size, dtype=np.uint8) * 114

    ratio = min(input_size[0] / image.shape[0],
                input_size[1] / image.shape[1])
    resized_image = cv2.resize(
        image,
        (int(image.shape[1] * ratio), int(image.shape[0] * ratio)),
        interpolation=cv2.INTER_LINEAR,
    )
    resized_image = resized_image.astype(np.uint8)

    padded_image[:int(image.shape[0] * ratio), :int(image.shape[1] *
                                                    ratio)] = resized_image
    padded_image = padded_image.transpose(swap)
    padded_image = np.ascontiguousarray(padded_image, dtype=np.float32)
    return padded_image, ratio

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    #y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    #y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def bbox_iou(box1, box2):
    xs1 = max(box1[0], box2[0])
    ys1 = max(box1[1], box2[1])
    xs2 = min(box1[0] + box1[2], box2[0] + box2[2])
    ys2 = min(box1[1] + box1[3], box2[1] + box2[3])

    intersections = max(ys2 - ys1, 0) * max(xs2 - xs1, 0)
    unions = (box1[2] * box1[3]) + (box2[2] * box2[3]) - intersections
    return intersections / unions


def box_iou(box1, box2):
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    inter = (np.min(box1[:, None, 2:], box2[:, 2:]) - np.max(box1[:, None, :2], box2[:, :2])).clip(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

def nms(boxes, scores, nms_thr):
    # from yolox nms
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [np.zeros((0, 6))] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence

        if labels and len(labels[xi]):
            l = labels[xi]
            v = np.zeros(len(l), nc + 5)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = np.concatenate((x, v), 0)

        if not x.shape[0]:
            continue

        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        box = xywh2xyxy(x[:, :4])

        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = np.concatenate((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf = np.max(x[:, 5:], 1, keepdims=True)
            j = np.argmax(x[:, 5:], axis=1)
            j = j.reshape((j.shape[0], 1))
            x = np.concatenate((box, conf, j), 1)[conf.reshape(-1) > conf_thres]

        if classes is not None:
            x = x[(x[:, 5:6] == np.array(classes)).any(1)]

        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = nms(boxes, scores, iou_thres)  # NMS　### ここが問題, indexを返すようにしたい
        i = np.array(i)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = np.dot(weights, x[:, :4]).float() / weights.sum(1, keepdims=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break
    return output