#!/usr/bin/python3
# coding=utf-8

import os
import os.path as osp
import numpy as np
import cv2
import json
import pickle
import matplotlib.pyplot as plt

import sys

cur_dir = os.path.dirname(__file__)
sys.path.insert(0, osp.join(cur_dir, 'PythonAPI'))
from crowdposetools.coco import COCO
from crowdposetools.cocoeval import COCOeval
from sklearn.model_selection import train_test_split


class Dataset(object):
    def __init__(self, train=True):
        dataset_name = 'JTA'
        additional_name = 'SyMPose_IOSB_CrowdPose'
        self.num_kps = 14
        self.kps_names=[
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
            "head_top",
            "neck",
        ],
        self.kps_symmetry = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]
        self.kps_lines = [
                        (0, 1),
                        (0, 2),
                        (1, 3),
                        (2, 4),
                        (3, 5),
                        (6, 7),
                        (6, 8),
                        (7, 9),
                        (8, 10),
                        (9, 11),
                        (12, 13),
                    ],
        self.sigmas = np.array([.79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89, .79, .79]) / 10.0

        # self.human_det_path = osp.join('../data', dataset_name, 'dets', 'human_detection.json')  # human detection result
        self.human_det_path = osp.join('../../crowdPE', 'dets', 'human_detection_test.json')
        self.img_path = osp.join('../data', dataset_name, additional_name, 'images')
        self.train_annot_path = osp.join('../data',dataset_name, additional_name, 'annotations', 'train_jta.json')
        self.num_val_split = 5
        self.val_annot_path = osp.join('../data',dataset_name, additional_name, 'annotations', 'train_jta.json')
        self.test_annot_path = osp.join('../data', dataset_name, additional_name,'annotations', 'iosb_crowdpose_test.json')
        self.train_data = []
        if train:
            coco = COCO(self.train_annot_path)
            # train_data = []
            for aid in coco.anns.keys():
                ann = coco.anns[aid]
                imgname = coco.imgs[ann['image_id']]['file_name']
                joints = ann['keypoints']

                if (ann['image_id'] not in coco.imgs) or ann['iscrowd'] or (np.sum(joints[2::3]) == 0) or (
                        ann['num_keypoints'] == 0):
                    continue

                # sanitize bboxes
                x, y, w, h = ann['bbox']
                img = coco.loadImgs(ann['image_id'])[0]
                width, height = img['width'], img['height']
                x1 = np.max((0, x))
                y1 = np.max((0, y))
                x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
                y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
                if ann['area'] > 0 and x2 >= x1 and y2 >= y1:
                # if x2 >= x1 and y2 >= y1:
                    bbox = [x1, y1, x2 - x1, y2 - y1]
                else:
                    continue

                data = dict(image_id=ann['image_id'], imgpath=imgname, id=aid, bbox=bbox, joints=joints, score=1)

                self.train_data.append(data)

    def load_train_data(self):
        # coco = COCO(self.train_annot_path)
        # train_data = []
        # for aid in coco.anns.keys():
        #     ann = coco.anns[aid]
        #     imgname = coco.imgs[ann['image_id']]['file_name']
        #     joints = ann['keypoints']
        #
        #     if (ann['image_id'] not in coco.imgs) or ann['iscrowd'] or (np.sum(joints[2::3]) == 0) or (
        #             ann['num_keypoints'] == 0):
        #         continue
        #
        #     # sanitize bboxes
        #     x, y, w, h = ann['bbox']
        #     img = coco.loadImgs(ann['image_id'])[0]
        #     width, height = img['width'], img['height']
        #     x1 = np.max((0, x))
        #     y1 = np.max((0, y))
        #     x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
        #     y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
        #     if ann['area'] > 0 and x2 >= x1 and y2 >= y1:
        #         bbox = [x1, y1, x2 - x1, y2 - y1]
        #     else:
        #         continue
        #
        #     data = dict(image_id=ann['image_id'], imgpath=imgname, id=aid, bbox=bbox, joints=joints, score=1)
        #
        #
        #     train_data.append(data)
        train_data, val_data = train_test_split(self.train_data, test_size=0.2)
        return train_data, val_data

    def load_val_data_with_annot(self):
        coco = COCO(self.val_annot_path)
        val_data = []
        for aid in coco.anns.keys():
            ann = coco.anns[aid]
            if ann['image_id'] not in coco.imgs:
                continue
            imgname = coco.imgs[ann['image_id']]['file_name']
            bbox = ann['bbox']
            joints = ann['keypoints']
            data = dict(image_id=ann['image_id'], imgpath=imgname, bbox=bbox, joints=joints, score=1)
            val_data.append(data)

        return val_data

    def load_annot(self, db_set):
        if db_set == 'train':
            coco = COCO(self.train_annot_path)
        elif db_set == 'val':
            coco = COCO(self.val_annot_path)
        elif db_set == 'test':
            coco = COCO(self.test_annot_path)
        else:
            print('Unknown db_set')
            assert 0

        return coco

    def load_test_data(self, score=False):
        coco = COCO(self.test_annot_path)
        test_data = []
        for aid in coco.anns.keys():
            ann = coco.anns[aid]
            imgname = coco.imgs[ann['image_id']]['file_name']
            joints = ann['keypoints']

            if (ann['image_id'] not in coco.imgs) or ann['iscrowd'] or (np.sum(joints[2::3]) == 0) or (
                    ann['num_keypoints'] == 0):
                continue

            # sanitize bboxes
            x, y, w, h = ann['bbox']
            img = coco.loadImgs(ann['image_id'])[0]
            width, height = img['width'], img['height']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if ann['area'] > 0 and x2 >= x1 and y2 >= y1:
            # if x2 >= x1 and y2 >= y1:
                bbox = [x1, y1, x2 - x1, y2 - y1]
            else:
                continue
            if score:
                data = dict(image_id=ann['image_id'], imgpath=imgname[7:], id=aid, bbox=bbox, joints=joints, score=1)
            else:
                data = dict(image_id=ann['image_id'], imgpath=imgname[7:], id=aid, bbox=bbox, joints=joints)

            test_data.append(data)

        return test_data


    def load_imgid(self, annot):
        return annot.imgs

    def imgid_to_imgname(self, annot, imgid, db_set):
        imgs = annot.loadImgs(imgid)
        imgname = [i['file_name'] for i in imgs]
        return imgname

    ##TODO: maybe need to change
    def evaluation(self, result, gt, result_dir, db_set):
        result_path = osp.join(result_dir, 'result.json')
        with open(result_path, 'w') as f:
            json.dump(result, f)

        result = gt.loadRes(result_path)
        cocoEval = COCOeval(gt, result, iouType='keypoints')

        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        result_path = osp.join(result_dir, 'result.pkl')
        with open(result_path, 'wb') as f:
            pickle.dump(cocoEval, f, 2)
            print("Saved result file to " + result_path)

    def evaluation_stats(self, result, annot, eval_dir):
        eval_path = osp.join(eval_dir, 'eval.json')
        with open(eval_path, 'w') as f:
            json.dump(result, f)

        result = annot.loadRes(eval_path)
        cocoEval = COCOeval(annot, result, iouType='keypoints')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        return cocoEval


    def vis_keypoints(self, img, kps, kp_thresh=0.4, alpha=1):

        # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
        cmap = plt.get_cmap('rainbow')
        colors = [cmap(i) for i in np.linspace(0, 1, len(self.kps_lines) + 2)]
        colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

        # Perform the drawing on a copy of the image, to allow for blending.
        kp_mask = np.copy(img)

        # Draw mid shoulder / mid hip first for better visualization.
        mid_shoulder = (kps[:2, 0] + kps[:2, 1]) / 2.0
        sc_mid_shoulder = np.minimum(kps[2, 0], kps[2, 1])
        mid_hip = (kps[:2, 6] + kps[:2, 7]) / 2.0
        sc_mid_hip = np.minimum(kps[2, 6], kps[2, 7])
        head_bot_idx = 13
        if sc_mid_shoulder > kp_thresh and kps[2, head_bot_idx] > kp_thresh:
            cv2.line(
                kp_mask, tuple(mid_shoulder.astype(np.int32)), tuple(kps[:2, head_bot_idx].astype(np.int32)),
                color=colors[len(self.kps_lines)], thickness=2, lineType=cv2.LINE_AA)
        if sc_mid_shoulder > kp_thresh and sc_mid_hip > kp_thresh:
            cv2.line(
                kp_mask, tuple(mid_shoulder.astype(np.int32)), tuple(mid_hip.astype(np.int32)),
                color=colors[len(self.kps_lines) + 1], thickness=2, lineType=cv2.LINE_AA)


        # Draw the keypoints.
        for l in range(len(self.kps_lines)):
            i1 = self.kps_lines[l][0]
            i2 = self.kps_lines[l][1]
            p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
            p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
            if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
                cv2.line(
                    kp_mask, p1, p2,
                    color=colors[l], thickness=2, lineType=cv2.LINE_AA)
            if kps[2, i1] > kp_thresh:
                cv2.circle(
                    kp_mask, p1,
                    radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
            if kps[2, i2] > kp_thresh:
                cv2.circle(
                    kp_mask, p2,
                    radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

        # Blend the keypoints.
        return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)


dbcfg = Dataset(False)
