import os 
import cv2
import sys
import logging
import numpy as np
import bilateral_solver
import PIL.Image as Image
from crf import dense_crf
from datetime import datetime
from scipy.spatial import distance

def get_logger(out_dir):
    logger = logging.getLogger('Exp')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    ts = str(datetime.now()).split(".")[0].replace(" ", "_")

    ts = ts.replace(":", "_").replace("-", "_")
    file_path = os.path.join(out_dir, "run_{}.log".format(ts)) if os.path.isdir(out_dir) else out_dir.replace('.pth.tar', '')
    file_hdlr = logging.FileHandler(file_path)
    file_hdlr.setFormatter(formatter)

    strm_hdlr = logging.StreamHandler(sys.stdout)
    strm_hdlr.setFormatter(formatter)

    logger.addHandler(file_hdlr)
    logger.addHandler(strm_hdlr)
    return logger


def iou(pred, target) : 
    
    intersection = np.logical_and(pred, target)
    union = np.logical_or(pred, target)
    
    return intersection.sum() / (union.sum() + 1e-7)


def mask_rgb_compose(org, mask, mask_color = [173, 216, 230]) : 
    
    rgb = np.copy(org)
    rgb[mask] = (rgb[mask] * 0.3 + np.array(mask_color) * 0.7).astype(np.uint8)
    
    return Image.fromarray(rgb)

def vis_mask_pil(pil, mask_coarse, args) : 
    w, h = pil.size
      
    mask_coarse = np.array(Image.fromarray(mask_coarse.astype(np.uint8) * 255).resize((w, h))) > 128

    if args.crf:
        img = np.array(pil.convert('RGB'))
        mask_refine = dense_crf(img, mask_coarse)
    elif args.bs:
        _, mask_refine = bilateral_solver.bilateral_solver_output(np.array(pil), mask_coarse, args.sigma_spatial, args.sigma_luma, args.sigma_chroma)
    else:
        raise NotImplementedError
    
    out = mask_rgb_compose(np.array(pil), mask_refine == 1)
    return out, mask_coarse, mask_refine


TAG_FLOAT = 202021.25
def read_flo(file):
    assert type(file) is str, "file is not str %r" % str(file)
    assert os.path.isfile(file) is True, "file does not exist %r" % str(file)
    assert file[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]
    f = open(file,'rb')
    flo_number = np.fromfile(f, np.float32, count=1)[0]
    assert flo_number == TAG_FLOAT, 'Flow number %r incorrect. Invalid .flo file' % flo_number
    w = np.fromfile(f, np.int32, count=1)
    h = np.fromfile(f, np.int32, count=1)
    data = np.fromfile(f, np.float32, count=2*w[0]*h[0])
    # Reshape data into 3D array (columns, rows, bands)
    flow = np.resize(data, (int(h), int(w), 2))
    f.close()
    return flow


def get_frames(in_path, out_path=None):
    frames=[]
    vidcap = cv2.VideoCapture(in_path)
    success,image = vidcap.read()
    count = 0
    while success:
        frames.append(image)
        cv2.imwrite(os.path.join(out_path, "frame%d.jpg" % count), image)     # save frame as JPEG file
        success,image = vidcap.read()
        count += 1  
