import os
import shutil
import argparse
import time
import torch
import glob as gb
import numpy as np
import graph
import tools
import networks
import datetime
from torchvision import transforms 
from datasets import extract_feat_info
from PIL import Image


parser = argparse.ArgumentParser("quick-start")

parser.add_argument("--arch", default="vit_small", type=str, choices=["vit_small", "vit_base", "moco_vit_small", "mae_vit_base"], help="Model architecture.")
parser.add_argument("--video-path", type=str, help="Video path")
parser.add_argument("--patch-size", default=16, type=int, help="Patch resolution of the model.")

parser.add_argument("--min-size", type=int, default=320, help="minimum resolution of image size")
parser.add_argument("--tau", type=float, default=0.3, help="hyper-parameter used in similarities")
parser.add_argument("--gap", type=int, default=1, help="frame gap between flow")
parser.add_argument("--fusion-mode", type=str, default='mean', choices=['mean', 'max', 'min', 'img', 'flow'], help="frame gap between flow")
parser.add_argument("--flow-model", type=str, default='RAFT', help="frame gap between flow")
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
start_time = time.time()

# load models + datasets
model = networks.get_model(args.arch, args.patch_size, device)
transform = transforms.Compose([ transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406),
                                                     (0.229, 0.224, 0.225)),])

# Extract frame from mp4 video

if not os.path.exists(args.out_dir): 
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, f'RAFT_Flows_gap{args.gap}'), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, f'RAFT_FlowImages_gap{args.gap}'), exist_ok=True)
if args.video_path[-4:] == '.mp4' or args.video_path[-4:] == '.MP4':
    data_path = os.path.join(args.out_dir, 'frames')
    if not os.path.exists(data_path): 
        os.makedirs(data_path, exist_ok=True)
    tools.get_frames(args.video_path, data_path)
else:
    data_path = args.video_path


gap = [args.gap] # gap = 3 for FBMS
reverse = [0]
shutil.copytree(data_path, './raft/input', dirs_exist_ok=True)
os.chdir(os.path.abspath("./raft")) ## cannot do it in the loop, as we cannot cd raft twice...

for r in reverse:
    for g in gap:
        print('===> Runing {}, gap {}'.format(args.video_path, g))
        flow_model = './raft-things.pth'  # model
        raw_flow_path = './RAFT_Flows_gap{}'.format(g)   # where to raw flow
        flow_path = './RAFT_FlowImages_gap{}'.format(g)   # where to save the image flow
        
        cmd = f"python predict.py  --gap {g} --model {flow_model} --path {'./input'} --outroot {flow_path} --reverse {0} --raw_outroot {raw_flow_path} --resize {args.min_size} "
        os.system(cmd)
        break
    
### don't forget to go back to main directory
os.chdir(os.path.abspath(".."))
#print (os.getcwd())
shutil.rmtree('./raft/input')
shutil.copytree(os.path.join('./raft', f'RAFT_Flows_gap{args.gap}/input'), os.path.join(args.out_dir, f'RAFT_Flows_gap{args.gap}'),dirs_exist_ok=True)
shutil.copytree(os.path.join('./raft', f'RAFT_FlowImages_gap{args.gap}/input'), os.path.join(args.out_dir, f'RAFT_FlowImages_gap{args.gap}'), dirs_exist_ok=True)
shutil.rmtree(os.path.join('./raft', f'RAFT_FlowImages_gap{args.gap}'))
shutil.rmtree(os.path.join('./raft', f'RAFT_Flows_gap{args.gap}'))

flow_img_dir = os.path.join(args.out_dir, f'RAFT_FlowImages_gap{args.gap}')
flow_dir = os.path.join(args.out_dir, f'RAFT_Flows_gap{args.gap}')

## extract features from frame
img_names, nb_node, nb_img, feat_h, feat_w, feats, arr_h, arr_w, frame_id, pil, _ = extract_feat_info('',  
                                                                                        data_path,
                                                                                        args.patch_size,
                                                                                        args.min_size,
                                                                                        args.arch,
                                                                                        model,
                                                                                        transform,
                                                                                        )
### extract features from flow
img_names, _, nb_flow, feat_h_flow, feat_w_flow, feats_flow, arr_h_flow, arr_w_flow, frame_id, _, flow = extract_feat_info('',
                                                                                        data_path,
                                                                                        args.patch_size,
                                                                                        args.min_size,
                                                                                        args.arch,
                                                                                        model,
                                                                                        transform,
                                                                                        flow_img_dir,
                                                                                        flow_dir,
                                                                                        )
assert nb_flow == nb_img 
assert feat_h == feat_h_flow
assert feat_w == feat_w_flow

print(f"Building the graph, {nb_node} nodes")

if not args.single_frame:
    foreground = graph.build_graph(nb_img, nb_node, feats, feats_flow, frame_id, arr_w, arr_h, args.tau, args.alpha, fusion_mode = args.fusion_mode, max_frame=args.max_frame)
else:
    foreground = graph.build_graph_single_frame(nb_img, feats, feats_flow, frame_id, arr_w, arr_h, args.tau, args.alpha, fusion_mode = args.fusion_mode)

foreground = foreground.reshape(nb_img, feat_h, feat_w)


print(f"Generating masks for input video")
out_vis = os.path.join(args.out_dir, 'coarse/')
out_vis_rgb = os.path.join(args.out_dir, 'rgb/')
if args.bs:
    args.crf = False
    out_vis_refine = os.path.join(args.out_dir, 'bs/')
elif args.crf:
    out_vis_refine = os.path.join(args.out_dir, 'crf/')
os.makedirs(out_vis_rgb, exist_ok=True)
if not os.path.exists(out_vis) :
    os.makedirs(out_vis, exist_ok=True)
    os.makedirs(out_vis_rgb, exist_ok=True)
    os.makedirs(out_vis_refine, exist_ok=True)

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
msg = f"ffmpeg -f image2 -framerate 5  -i {out_vis_rgb}/%05d.jpg {os.path.join(args.out_dir,'segmentation')}.gif"
os.system(msg)

end_time = time.time()
print(f'Time cost: {str(datetime.timedelta(milliseconds=int((end_time - start_time)*1000)))}')
