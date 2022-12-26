# Copyright (C) 2020-2022 Intel Corporation
#
# SPDX-License-Identifier: MIT
import torch
import cv2
from models.common import DetectMultiBackend
from utils.general import check_img_size
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages
from skimage.measure import find_contours, approximate_polygon
from utils.augmentations import (Albumentations, augment_hsv, classify_albumentations, classify_transforms, copy_paste,
                                 letterbox, mixup, random_perspective)
from pred import return_res
import os
import numpy as np
import sys
weights='best_yolov5.pt'
vid_stride=1
half=False,
bs = 1  # use FP16 half-precision inference
dnn=False # use OpenCV DNN for ONNX inference
labels=["Basophil","Eosinophil","Erythroblast","Intrusion","Lymphocyte","Monocyte","Myelocyte","Neutrophil","Platelles","RBC"]

class ModelLoader:
    def __init__(self):
        COCO_MODEL_PATH = os.path.join('./', "best_yolov5.pt")
        if COCO_MODEL_PATH is None:
            raise OSError('Model path env not found in the system.')

        # Limit gpu memory to 30% to allow for other nuclio gpu functions. Increase fraction as you like
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model =  DetectMultiBackend(weights)
        
        self.labels = labels

    def infer(self, image, threshold):
        stride, names, pt = self.model.stride, self.model.names, self.model.pt
        
        imgsz = check_img_size((640,640), s=stride)
        shape_im=image.shape
        im = letterbox(image, imgsz, stride=stride, auto=pt)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous
        self.model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))
        masks,classes,conf=return_res(self.model,im,imgsz,self.device)
       # output = self.model.detect([image], verbose=1)[0]

        results = []
        MASK_THRESHOLD = 0.5
        for i in range(len(classes)):
            score = conf[i].item()
            class_id = classes[i].item()
            mask =cv2.resize(masks[i],(shape_im[1],shape_im[0]))
            mask = mask.astype(np.uint8)
            contours = find_contours(mask, MASK_THRESHOLD)
            contour = contours[0]
            contour = np.flip(contour, axis=1)
            # Approximate the contour and reduce the number of points
            contour = approximate_polygon(contour, tolerance=2.5)
            if len(contour) < 6:
                continue
            label = self.labels[class_id]
            results.append({
                    "confidence": str(score),
                    "label": label,
                    "points": contour.ravel().tolist(),
                    "type": "polygon",
                })

        return results