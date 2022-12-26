import  torch
from models.common import DetectMultiBackend
from utils.general import check_img_size
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, scale_segments,
                           strip_optimizer)
from utils.segment.general import masks2segments, process_mask, process_mask_native
from pathlib import Path
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages
visualize=False
augment=False
classes=None
retina_masks=False
agnostic_nms=False
max_det=1000
webcam=False
conf_thres=0.25
iou_thres=0.45
def return_res(model,im,im0s,imgsz,device):
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    with dt[0]:
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]
    with dt[1]:
        #visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred, proto = model(im, augment=augment, visualize=visualize)[:2] 
    with dt[2]:
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)
    for i, det in enumerate(pred):
        s += '%gx%g ' % im.shape[2:]  # per image
        seen += 1
        if webcam:  # batch_size >= 1
            im0 = im0s[i].copy()
            s += f'{i}: '
        else:
            im0 = im0s.copy()

    if len(det):
        if retina_masks:
            # scale bbox first the crop masks
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size
            masks = process_mask_native(proto[i], det[:, 6:], det[:, :4], im0.shape[:2])  # HWC
        else:
            masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size 
    return masks , det[:, 5] ,det[:, 4]          
