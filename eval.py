
import os
import sys
import datasets
import argparse
import numpy as np
import bilateral_solver
from PIL import Image
from crf import dense_crf
from scipy.spatial import distance
from tools import mask_rgb_compose, iou


def compute_IoU(gt_mask, pred_mask):
    if len(gt_mask.shape) == 3:
        gt_mask = gt_mask[:,:,1]
    gt_mask = gt_mask.astype(np.bool)
    pred_mask = pred_mask.astype(np.bool)

    if np.isclose(np.sum(pred_mask),0) and np.isclose(np.sum(gt_mask),0):
        return 1
    else:
        return np.sum((pred_mask & gt_mask)) / \
                np.sum((pred_mask | gt_mask),dtype=np.float32)


parser = argparse.ArgumentParser("TokenCut Video Evaluation")
parser.add_argument("--dataset", type=str, default='DAVIS', choices=['DAVIS','FBMS', 'SegTrackv2',None], help="Dataset name?",)
parser.add_argument("--mask_dir", type=str, help="The predicted mask folder")
parser.add_argument("--gt_dir", type=str, help="The ground truth mask folder")
parser.add_argument("--resolution", type=str, default='480p', help="The video resolution used in davis dataset")

args = parser.parse_args()

img_dir, anno_dir, _, flow_dir, video_list = datasets.VideoSet(args.dataset, resolution=args.resolution, is_train=False)

CategoryIou = {}
num_frame = 0
for vid_id in range(len(video_list)):
    vid_name = video_list[vid_id]
    gts = sorted(os.listdir(os.path.join(args.gt_dir, vid_name)))
    gts = [x for x in gts if x.endswith('.png')]
    masks = sorted(os.listdir(os.path.join(args.mask_dir, vid_name)))
    masks = [m for m in masks if m.replace('.bmp', '.png').replace('.jpg','.png') in gts]
    assert len(gts) > 0, 'Ground truth file not found.'
    assert len(gts) == len(masks)
    for i in range(len(gts)):
        gt = np.array(Image.open(os.path.join(args.gt_dir, vid_name, gts[i])))
        mask = np.array(Image.open(os.path.join(args.mask_dir, vid_name, masks[i])))

        iou = compute_IoU(gt_mask=gt, pred_mask=mask)
        try:
            CategoryIou[vid_name].append(iou)
        except:
            CategoryIou[vid_name] = [iou]
        num_frame += 1 

tot_ious = 0
tot_maes = 0
per_cat_iou = []
for cat, list_iou in CategoryIou.items():
    print("Category {}: IoU is {:.2%}".format(cat, np.mean(list_iou)))
    tot_ious += np.sum(list_iou)
    per_cat_iou.append(np.mean(list_iou))
print("The Average over the dataset: IoU is {:.2%}".format(tot_ious/float(num_frame)))
print("The Average over sequences IoU is {:.2%}".format(np.mean(per_cat_iou)))
print("Success: Processed {} frames".format(num_frame))

