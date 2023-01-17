import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, check_requirements, non_max_suppression_detection, \
    scale_coords, xyxy2xywh, set_logging, increment_path, refind_Zhang
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized


def detect(save_img=True):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    # Directories

    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    if save_img:
        save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    # half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    # if half:
    #     model.half()  # to FP16

    # Second-stage classifier
    # Set Dataloader
    vid_path, vid_writer = None, None

    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    """
      path: the path of image file or video
      img: [c,w,h]
      img0s: original image [w,h,c]
      cap: only for video reading. if reading image, it will be None;
      """
    Target = []
    Target_candidate = []
    frame = 0
    tar_num = 1
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred_high, pred_candidates, _ = non_max_suppression_detection(prediction=pred, conf_factor=opt.conf_factor, iou_thres = opt.iou_thres, merge_thres=0.5, possible_targer_n=tar_num, wh_range=[20, imgsz],merge=True, classes=opt.classes, agnostic=opt.agnostic_nms)
        # # my post-processing v1
        if dataset.mode == 'image':
            frame = int(Path(path).name.split('.')[0].split('_')[-1])+1
        elif dataset.mode == 'video':
            frame = getattr(dataset, 'frame', 0)

        if frame == 0:
            raise Exception(f'ERROR: {source} does not meet the required format!')
        Target, Target_candidate = refind_Zhang(Target, Target_candidate, pred_high, pred_candidates, frame, tar_num=tar_num, conj_thres=0.2)

        t2 = time_synchronized()
        # Process detections
        for i, (det, det_other) in enumerate(zip(Target, Target_candidate)):  # detections per image
            # p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            p, s, im0 = path, '', im0s
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):  # xyxy: bbox
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1, margin=5)
                        plot_one_box(xyxy, im0, label=label, color=[255, 0, 0], line_thickness=1, margin=5)

            if len(det_other) > 0 and len(det) < tar_num:
                # Rescale boxes from img_size to im0 size
                det_other[:, :4] = scale_coords(img.shape[2:], det_other[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det_other):  # xyxy: bbox

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1, margin=5)
                        plot_one_box(xyxy, im0, label=label, color=[0, 0, 255], line_thickness=1, margin=5)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


#data/raw_Testimage_det1/*/*.png
#data/raw_Testimage_det1/sequence20_frame00050
#data/raw_Testvideo_det
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='runs/train/all_exp_yolov5s_det/weights/best.pt', help='model.pt path(s)')
    # parser.add_argument('--source', type=str, default='data/raw_Testimage_det1/sequence09_frame00100', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='data/brainlab_image/BewegungDraht_C1_Left', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-factor', type=float, default=0.1, help='object confidence factor')
    parser.add_argument('--iou-thres', type=float, default=0.1, help='IOU threshold for NMS')
    parser.add_argument('--device', default='1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--project', default='runs/mydetect_test_brainlab/all_exp_yolov5s_det', help='save results to project/name')
    parser.add_argument('--name', default='exp_Zhang', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        detect()
