import os 
import PIL.Image as Image 
import numpy as np 
from networks import feature_extractor
from tools import read_flo

def VideoSet(dataset_name='DAVIS', resolution='480p', is_train=True, gap=1, flow_model='RAFT') : 
    if dataset_name == 'DAVIS':
        img_dir = os.path.join('./data/DAVIS/JPEGImages/' + resolution) # JPG
        anno_dir = os.path.join('./data/DAVIS/Annotations/' + resolution) # PNG
        flow_img_dir = os.path.join(f'./data/DAVIS/{flow_model}_FlowImages_gap{gap}/' + resolution) # PNG
        flow_dir = os.path.join(f'./data/DAVIS/{flow_model}_Flows_gap{gap}/' + resolution) # PNG
        video_list = np.load('./data/DAVIS/train_vid.npy') if is_train else np.load('./data/DAVIS/val_vid.npy')
    elif dataset_name == 'FBMS':
        img_dir = os.path.join('./data/FBMS/JPEGImages')
        anno_dir = os.path.join('./data/FBMS/Annotations')
        flow_img_dir = os.path.join(f'./data/FBMS/{flow_model}_FlowImages_gap{gap}/') # PNG
        flow_dir = os.path.join(f'./data/FBMS/{flow_model}_Flows_gap{gap}/') # flo
        video_list = np.load('./data/FBMS/train_vid.npy') if is_train else np.load('./data/FBMS/val_vid.npy')
    elif dataset_name == 'SegTrackv2':
        img_dir = os.path.join('./data/SegTrackv2/JPEGImages')
        anno_dir = os.path.join('./data/SegTrackv2/GroundTruth')
        flow_img_dir = os.path.join(f'./data/SegTrackv2/{flow_model}_FlowImages_gap{gap}/') # PNG
        flow_dir = os.path.join(f'./data/SegTrackv2/{flow_model}_Flows_gap{gap}/') # flo
        video_list = np.load('./data/SegTrackv2/train_vid.npy') if is_train else np.load('./data/SegTrackv2/val_vid.npy')
    else: 
        raise NotImplementedError
        
    return img_dir, anno_dir, flow_img_dir, flow_dir, video_list


def resize_img(I, patch_size, min_size = 480, mode=Image.LANCZOS) :

    w, h = I.size
    
    ## resize img, the largest dimension is maxSize
    w_ratio, h_ratio = w / min_size, h / min_size
    min_ratio = min(w_ratio, h_ratio)
    
    w, h= w / min_ratio, h / min_ratio
    w_resize = round(w/ patch_size) * patch_size
    h_resize = round(h/ patch_size) * patch_size

    return I.resize((w_resize, h_resize), resample=mode)
    
def extract_feat_info(vid_name, img_dir, patch_size, min_size, arch, model, transform, flow_img_dir=None, flow_dir=None) : 
    
    ## this will save features, frame id, each feature's w and h
    feats = []
    frame_id = []
    w = []
    h = []
    pil = []
    flo = []

    if 'DAVIS' in img_dir:
        nb_img = len(os.listdir(os.path.join(img_dir, vid_name)))
        img_names = ['%05d.jpg' % i for i in range(nb_img)]
        if flow_img_dir is not None:
            nb_flow = len(os.listdir(os.path.join(flow_img_dir, vid_name)))
            flow_img_names = ['%05d.jpg' % i for i in range(nb_flow)]
            flow_names = ['%05d.flo' % i for i in range(nb_flow)]
            for i in range(nb_img - nb_flow):
                flow_img_names.append(flow_img_names[-1]) 
                flow_names.append(flow_names[-1]) 
    elif 'FBMS' in img_dir:
        nb_img = len(os.listdir(os.path.join(img_dir, vid_name)))
        img_names = sorted(os.listdir(os.path.join(img_dir, vid_name)))
        if flow_img_dir is not None:
            flow_img_names = sorted(os.listdir(os.path.join(flow_img_dir, vid_name)))
            flow_names = sorted(os.listdir(os.path.join(flow_dir, vid_name)))
            for i in range(nb_img - len(flow_img_names)):
                flow_img_names.append(flow_img_names[-1]) 
                flow_names.append(flow_names[-1]) 
            assert len(img_names) == len(flow_img_names)
    elif 'SegTrackv2' in img_dir:
        nb_img = len(os.listdir(os.path.join(img_dir, vid_name)))
        img_names = sorted(os.listdir(os.path.join(img_dir, vid_name)))
        if flow_img_dir is not None:
            flow_img_names = sorted(os.listdir(os.path.join(flow_img_dir, vid_name)))
            flow_names = sorted(os.listdir(os.path.join(flow_dir, vid_name)))
            for i in range(nb_img - len(flow_img_names)):
                flow_img_names.append(flow_img_names[-1]) 
                flow_names.append(flow_names[-1]) 
            assert len(img_names) == len(flow_img_names)
    else:
        nb_img = len(os.listdir(os.path.join(img_dir)))
        img_names = sorted(os.listdir(os.path.join(img_dir, vid_name)))
        if flow_img_dir is not None:
            flow_img_names = sorted(os.listdir(os.path.join(flow_img_dir)))
            flow_names = sorted(os.listdir(os.path.join(flow_dir)))
            for i in range(nb_img - len(flow_img_names)):
                flow_img_names.append(flow_img_names[-1]) 
                flow_names.append(flow_names[-1]) 
            assert len(img_names) == len(flow_img_names)


    for i in range(nb_img) : 
        img_name = img_names[i] 
        if flow_img_dir is not None:
            flow_img_name = flow_img_names[i] 
            flow_name = flow_names[i] 

        img = Image.open(os.path.join(img_dir, vid_name, img_name)).convert('RGB')
        pil.append(img)
        if flow_dir is not None and flow_img_dir is not None:
            flow = Image.open(os.path.join(flow_img_dir, vid_name, flow_img_name)).convert('RGB')
            I = resize_img(flow, patch_size, min_size)
            pil.append(flow)
            flow = read_flo(os.path.join(flow_dir, vid_name, flow_name))
            flo.append(flow)
        else:
            I = resize_img(img, patch_size, min_size)
        img_tensor = transform(I).cuda()
        hh, ww = np.meshgrid(np.arange(img_tensor.shape[1] // patch_size),
                             np.arange(img_tensor.shape[2] // patch_size),
                             indexing='ij')
        feat = feature_extractor(arch, model, img_tensor)
        nb_feat = feat.shape[0]
        feats.append(feat)
        frame_id.append(np.ones(nb_feat) * i)
        w.append(ww.reshape(-1))
        h.append(hh.reshape(-1))

    if len(flo)>0:
        flo = np.stack(flo, axis=0)

    nb_feat_total = nb_img * nb_feat
    feats = np.stack(feats,axis=0).reshape(nb_feat_total, -1)
    feats = feats / (np.sqrt(np.sum(feats**2, axis=1, keepdims=True)) + 1e-7) ## normalization
    w = np.stack(w,axis=0).reshape(nb_feat_total, -1)
    h = np.stack(h,axis=0).reshape(nb_feat_total, -1)
    frame_id = np.stack(frame_id,axis=0).reshape(nb_feat_total, -1)
    return img_names, nb_feat_total, nb_img, img_tensor.shape[1] // patch_size, img_tensor.shape[2] // patch_size, feats, h, w, frame_id, pil, flo
            
