import random

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo
import torch
import cv2
import numpy as np

floor_coordinate = []


class Detector:
    def __init__(self, model_type='OD'):
        self.cfg = get_cfg()
        self.model_type = model_type
        # load model config
        if model_type == 'OD':  # object detection
            self.cfg.merge_from_file(model_zoo.get_config_file(r"COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(r"COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
        elif model_type == 'IS':  # instance segmentation
            self.cfg.merge_from_file(model_zoo.get_config_file(r"COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                r"COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        elif model_type == 'LVIS':  # LVIS segmentation
            self.cfg.merge_from_file(
                model_zoo.get_config_file(r"LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                r"LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml")
        elif model_type == 'PS':
            self.cfg.merge_from_file(model_zoo.get_config_file(r"COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                r"COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.DEVICE = 'cuda'

        self.predictor = DefaultPredictor(self.cfg)

    def onImage(self, imagePath, cnt):
        image = cv2.imread(imagePath)
        image = cv2.resize(image, dsize=(640, 480))
        if self.model_type != 'PS':

            predictions = self.predictor(image)

            viz = Visualizer(image[:, :, ::-1], metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
                             instance_mode=ColorMode.IMAGE_BW)

            output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))
        else:
            predictions, segmentInfor = self.predictor(image)['panoptic_seg']
            row_indices, col_indices = torch.where(predictions == 17)
            for row_indice, col_indice in zip(row_indices, col_indices):
                floor_coordinate.append([row_indice.item(), col_indice.item()])
            # Print the row and column indices where the value 5 occurs
            # for row_idx, col_idx in zip(row_indices, col_indices):
            #     print(row_idx, col_idx)
            viz = Visualizer(image[:, :, ::-1], metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))
            output = viz.draw_panoptic_seg_predictions(predictions.to("cpu"), segmentInfor)
        cv2.imwrite('img' + str(cnt) + '.jpg', output.get_image()[:, :, ::-1])
        out = output.get_image()[:, :, ::-1]
        # print(out.shape)
        # print(np.array(floor_coordinate)[:, 0].max())
        # print(np.array(floor_coordinate)[:, 1].max())
        # # out[floor_coordinate] = 0
        for coord in floor_coordinate:
            row, col = coord# print()
            out[row][col] = 0
        cv2.imshow('img', out)

        cv2.waitKey(0)


model = Detector(model_type="PS")

img_path = r"E:\juntion2023_2\AI-Junction\AI-Junction\data_btc\data\frames\30\192_168_5_102_frame_30.jpg"
model.onImage(img_path, 2)
