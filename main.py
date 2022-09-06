import os
import glob
import time 
import tools
import torch
import graph
import networks
import datasets
import datetime
import argparse
import numpy as np
import PIL.Image as Image
from torchvision import transforms 
from datasets import extract_feat_info


    
parser = argparse.ArgumentParser("Visualize Self-Attention maps")

parser.add_argument("--arch", default="vit_small", type=str, choices=["vit_small", "vit_base", "moco_vit_small", "mae_vit_base"], help="Model architecture.")
parser.add_argument("--dataset", type=str, default='DAVIS', choices=['DAVIS','FBMS', 'SegTrackv2',None], help="Dataset name?",)
parser.add_argument("--resolution", type=str, default='480p', help="dataset resolution, for DAVIS, 480p and 1080p are possible",)
parser.add_argument("--patch-size", default=16, type=int, help="Patch resolution of the model.")

parser.add_argument("--min-size", type=int, default=320, help="minimum resolution of image size")
parser.add_argument("--tau", type=float, default=0.3, help="hyper-parameter used in similarities")
parser.add_argument("--gap", type=int, default=1, help="frame gap between flow")
parser.add_argument("--fusion-mode", type=str, default='mean', choices=['mean', 'max', 'min', 'img', 'flow'], help="frame gap between flow")
parser.add_argument("--flow-model", type=str, default='RAFT', choices=['RAFT', 'ARFlow'], help="frame gap between flow")
parser.add_argument("--alpha", type=float, default=0.5, help="hyper-parameter used in similarities")
parser.add_argument("--max-frame", type=int, default=90, help="Max number of frame when building graph")

## --- parameters used in bilateral solver
parser.add_argument("--bs", action="store_true", help="Using bilateral solver")
parser.add_argument('--sigma-spatial', type=float, default=16, help='sigma spatial in the bilateral solver')
parser.add_argument('--sigma-luma', type=float, default=16, help='sigma luma in the bilateral solver')
parser.add_argument('--sigma-chroma', type=float, default=8, help='sigma chroma in the bilateral solver')

## --- parameters used in crf
parser.add_argument("--crf", action="store_true", help="Using crf")
parser.set_defaults(crf=True)
parser.add_argument("--single-frame", action="store_true", help="Build graph for each frame")

parser.add_argument("--out-dir", type=str, default="./output", help="Output directory to store predictions and visualizations.")

args = parser.parse_args()

print(f'args: {args}')
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

## load models + datasets
model = networks.get_model(args.arch, args.patch_size, device)
transform = transforms.Compose([ transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406),
                                                     (0.229, 0.224, 0.225)),])

img_dir, anno_dir, flow_img_dir, flow_dir, video_list = datasets.VideoSet(args.dataset, resolution=args.resolution, is_train=False, gap=args.gap, flow_model=args.flow_model)

if not os.path.exists(args.out_dir) : 
    os.makedirs(args.out_dir, exist_ok=True)
logger = tools.get_logger(args.out_dir)
logger.info (args)
logger.info(f"Running TokenCut-Video on the dataset {args.dataset} with {args.arch}: in total {len(video_list)} videos...")


start_time = time.time()
if args.dataset == 'DAVIS':
    postfix = args.resolution + '/tokencut/'
else:
    postfix = ''

for vid_id in range(len(video_list)) : 
    
    vid_name = video_list[vid_id]
    logger.info(f"Running TokenCut on video No.{vid_id} (video name: {vid_name})")

    out_vis = os.path.join(args.out_dir, 'coarse/' , postfix, vid_name)
    out_vis_rgb = os.path.join(args.out_dir, 'rgb/', postfix, vid_name)
    if args.bs:
        args.crf = False
        out_vis_refine = os.path.join(args.out_dir, 'bs/', postfix, vid_name)
    elif args.crf:
        out_vis_refine = os.path.join(args.out_dir, 'crf/', postfix, vid_name)

    os.makedirs(out_vis_rgb, exist_ok=True)
    if not os.path.exists(out_vis) :
        os.makedirs(out_vis, exist_ok=True)
        os.makedirs(out_vis_rgb, exist_ok=True)
        os.makedirs(out_vis_refine, exist_ok=True)
    
    #if os.path.exists(os.path.join(args.out_dir,vid_name + '.gif')): 
    #    continue

    # extract features from frame
    img_names, nb_node, nb_img, feat_h, feat_w, feats, arr_h, arr_w, frame_id, pil, _ = extract_feat_info(vid_name,  
                                                                                            img_dir,
                                                                                            args.patch_size,
                                                                                            args.min_size,
                                                                                            args.arch,
                                                                                            model,
                                                                                            transform
                                                                                            )
    # extract features from flow
    img_names, _, nb_flow, feat_h_flow, feat_w_flow, feats_flow, arr_h_flow, arr_w_flow, frame_id, _, flow = extract_feat_info(vid_name,
                                                                                            img_dir,
                                                                                            args.patch_size,
                                                                                            args.min_size,
                                                                                            args.arch,
                                                                                            model,
                                                                                            transform,
                                                                                            flow_img_dir,
                                                                                            flow_dir)
    assert nb_flow == nb_img 
    assert feat_h == feat_h_flow
    assert feat_w == feat_w_flow

    logger.info(f"Building the graph, {nb_node} nodes")

    if not args.single_frame:
        foreground = graph.build_graph(nb_img, nb_node, feats, feats_flow, frame_id, arr_w, arr_h, args.tau, args.alpha, fusion_mode = args.fusion_mode, max_frame=args.max_frame)
    else:
        foreground = graph.build_graph_single_frame(nb_img, feats, feats_flow, frame_id, arr_w, arr_h, args.tau, args.alpha, fusion_mode = args.fusion_mode)

    foreground = foreground.reshape(nb_img, feat_h, feat_w)
    

    logger.info(f"Generating masks for video {vid_name}")

    for img_id in range(nb_img) :
        rgb, mask_coarse, mask_refine = tools.vis_mask_pil(pil[img_id],
                                 foreground[img_id],
                                 args)
                                 
        rgb.save(os.path.join(out_vis_rgb, '%05d.jpg' % img_id))
        Image.fromarray(mask_coarse.astype(np.uint8) * 255).save(os.path.join(out_vis, img_names[img_id].replace('.jpg','.png')))
        Image.fromarray(mask_refine.astype(np.uint8) * 255).save(os.path.join(out_vis_refine, img_names[img_id].replace('.jpg','.png')))
        
        if img_id % 10 == 9 : 
            print (f"{img_id} / {nb_img} ...")
    del pil, feats, arr_h, arr_w, foreground
    msg = f"ffmpeg -f image2 -framerate 5  -i {out_vis_rgb}/%05d.jpg {os.path.join(args.out_dir, vid_name)}.gif"
    os.system(msg)

end_time = time.time()
logger.info(f'Time cost: {str(datetime.timedelta(milliseconds=int((end_time - start_time)*1000)))}')
