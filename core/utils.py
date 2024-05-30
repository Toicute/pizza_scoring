'''
========================================================================
For some reason these utils need to move from Yolov7 source code to here
========================================================================
'''
import time

import cv2
import torch
import torchvision
from torch.nn import functional as F
import numpy as np

from .tmp_detectron import Boxes


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def merge_bases(rois, coeffs, attn_r, num_b, location_to_inds=None):
    # merge predictions
    # N = coeffs.size(0)
    if location_to_inds is not None:
        rois = rois[location_to_inds]
    N, B, H, W = rois.size()
    if coeffs.dim() != 4:
        coeffs = coeffs.view(N, num_b, attn_r, attn_r)
    # NA = coeffs.shape[1] //  B
    coeffs = F.interpolate(coeffs, (H, W),
                           mode="bilinear").softmax(dim=1)
    # coeffs = coeffs.view(N, -1, B, H, W)
    # rois = rois[:, None, ...].repeat(1, NA, 1, 1, 1)
    # masks_preds, _ = (rois * coeffs).sum(dim=2) # c.max(dim=1)
    masks_preds = (rois * coeffs).sum(dim=1)
    return masks_preds


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def non_max_suppression_mask_conf(prediction, attn, bases, pooler, hyp, conf_thres=0.1, iou_thres=0.6, merge=False, classes=None, agnostic=False, mask_iou=None, vote=False):

    if prediction.dtype is torch.float16:
        prediction = prediction.float()  # to FP32
    nc = prediction[0].shape[1] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates
    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [None] * prediction.shape[0]
    output_mask = [None] * prediction.shape[0]
    output_mask_score = [None] * prediction.shape[0]
    output_ac = [None] * prediction.shape[0]
    output_ab = [None] * prediction.shape[0]
    
    def RMS_contrast(masks):
        mu = torch.mean(masks, dim=-1, keepdim=True)
        return torch.sqrt(torch.mean((masks - mu)**2, dim=-1, keepdim=True))
    
    
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence
        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])
        
        # If none remain process next image
        if not x.shape[0]:
            continue
            
        a = attn[xi][xc[xi]]
        base = bases[xi]

        bboxes = Boxes(box)
        pooled_bases = pooler([base[None]], [bboxes])
        
        pred_masks = merge_bases(pooled_bases, a, hyp["attn_resolution"], hyp["num_base"]).view(a.shape[0], -1).sigmoid()

        if mask_iou is not None:
            mask_score = mask_iou[xi][xc[xi]][..., None]
        else:
            temp = pred_masks.clone()
            temp[temp < 0.5] = 1 - temp[temp < 0.5]
            mask_score = torch.exp(torch.log(temp).mean(dim=-1, keepdims=True))#torch.mean(temp, dim=-1, keepdims=True)
        
        x[:, 5:] *= x[:, 4:5] * mask_score # x[:, 4:5] *   * mask_conf * non_mask_conf  # conf = obj_conf * cls_conf

        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
            mask_score = mask_score[i]
            if attn is not None:    
                pred_masks = pred_masks[i]
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]


        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue
        
        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        # scores *= mask_score
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
            
        
        all_candidates = []
        all_boxes = []
        if vote:
            ious = box_iou(boxes[i], boxes) > iou_thres
            for iou in ious: 
                selected_masks = pred_masks[iou]
                k = min(10, selected_masks.shape[0])
                _, tfive = torch.topk(scores[iou], k)
                all_candidates.append(pred_masks[iou][tfive])
                all_boxes.append(x[iou, :4][tfive])
        #exit()
            
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy
            except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                print(x, i, x.shape, i.shape)
                pass

        output[xi] = x[i]
        output_mask_score[xi] = mask_score[i]
        output_ac[xi] = all_candidates
        output_ab[xi] = all_boxes
        if attn is not None:
            output_mask[xi] = pred_masks[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output, output_mask, output_mask_score, output_ac, output_ab