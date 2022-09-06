import os 
from PIL import Image
from tqdm import  tqdm
import glob
import imageio

vis_path = './test/FlowImages_gap1/test3/'
video_path = './test3.mp4'
frame_path = os.path.join(vis_path, "*.png")
video_path = os.path.join(video_path)
imgs = [Image.open(f) for f in tqdm(sorted(glob.glob(frame_path)))]
imageio.mimsave(video_path, imgs)
print(f"Video saved at {video_path}.")
