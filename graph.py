import torch
import tools 
import numpy as np
import torch.nn.functional as F
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from scipy.sparse import csc_matrix, csr_matrix, lil_matrix, coo_matrix, diags

def build_graph(nb_img, nb_node, feats, feats_flow, frame_id, arr_w, arr_h, tau, frame_sigma=300, alpha=0.5, fusion_mode='mean', max_frame=100) :
    
    ## first allocate row and column array in order to accelerate
    ## tmp should be higher if the graph has more edges
    tmp = 100000
    batch_size = feats.shape[0] // nb_img

    ## Due to the limit of computational resources, we build multiple nonoverlaping graph in case the video is too long.
    frame_treshold = max_frame 

    
    nb_graph = nb_img // frame_treshold if nb_img % frame_treshold == 0 else nb_img // frame_treshold + 1
    foregrounds = []
    for graph_id in range(nb_graph):
        edge_start = graph_id * frame_treshold * batch_size
        edge_end = min(edge_start + frame_treshold * batch_size, nb_node)
        row = np.zeros(batch_size * frame_treshold * tmp, np.int32)
        column = np.zeros(batch_size *frame_treshold * tmp, np.int32)
        idx = 0
        num_frame = min(frame_treshold, (edge_end-edge_start+1)//batch_size) 
        for batch_id in range(num_frame) : 
            start = edge_start + batch_size * batch_id
            end = start + batch_size
            if graph_id == nb_graph - 1 and batch_id == num_frame: 
                end = edge_end

            edge_img = feats[start : end] @ feats[edge_start:edge_end].T 
            edge_flow = feats_flow[start : end] @ feats_flow[edge_start:edge_end].T 
            if fusion_mode == 'mean':
                edge = (1-alpha) * edge_img + alpha * edge_flow
            elif fusion_mode == 'max':
                edge = np.maximum(edge_img, edge_flow)
            elif fusion_mode == 'min':
                edge = np.minimum(edge_img, edge_flow)
            elif fusion_mode == 'img':
                edge = edge_img
            elif fusion_mode == 'flow':
                edge = edge_flow

            idx_row, idx_column = np.where(edge > tau)
            row[idx : idx + len(idx_row)] = idx_row + batch_size * batch_id
            column[idx : idx + len(idx_row)] = idx_column 

            idx += len(idx_row)
            
            if batch_id % 10 == 9 : 
                print (f"{batch_id + frame_treshold*graph_id} / {nb_img + 1} ...")
        row = row[: idx]
        column = column[: idx]

        ## build coo matrix
        graph = coo_matrix((np.ones(idx, np.float32), (row, column)), shape=((edge_end-edge_start), (edge_end-edge_start)))

        W = graph.tocsr().tolil()
        del graph, row, column, edge, edge_img, edge_flow
        D = diags(np.asarray(W.sum(axis=1)).flatten())
        E = (D - W).tocsr().tocsc()
        D = D.tocsr().tocsc()
        del  W
        _, eigenvectors = eigsh(E, 2, D, which='SM', v0=np.ones((edge_end-edge_start), np.float64) * 1/(edge_end - edge_start)**0.5) ## second smallest eigenvector
        eigenvectors = eigenvectors[:, 1]

        max_eig = eigenvectors.max()
        max_abs_eig = np.abs(eigenvectors).max()

        eigenvectors = (eigenvectors > eigenvectors.mean())
        foreground = eigenvectors == 1 if max_abs_eig == max_eig else eigenvectors == 0
        del D, E, eigenvectors
        foregrounds.append(foreground)
        del foreground
    foreground = np.concatenate(foregrounds,axis=0) 
    return foreground


def build_graph_single_frame(nb_img, feats, feats_flow, frame_id, feat_w, feat_h, tau, alpha=0.5, eps=1e-5, fusion_mode='mean'):
    batch_size = feats.shape[0] // nb_img
    feats = torch.from_numpy(feats)

    feats_flow = torch.from_numpy(feats_flow)
    mask = []
    for batch_id in range(nb_img):
        start = batch_size * batch_id
        end = start + batch_size

        edge_img = feats[start : end] @ feats[start:end].T
        edge_flow = feats_flow[start : end] @ feats_flow[start:end].T

        if fusion_mode == 'mean':
            edge = alpha * edge_img + alpha * edge_flow
        elif fusion_mode == 'max':
            edge = np.maximum(edge_img, edge_flow)
        elif fusion_mode == 'min':
            edge = np.minimum(edge_img, edge_flow)
        elif fusion_mode == 'img':
            edge = edge_img
        elif fusion_mode == 'flow':
            edge = edge_flow
        A = edge.unsqueeze(0)

        ## Using lobpcg
        A = A > tau
        A = A.float()
        A = A + eps
        d_i = torch.sum(A, dim=2)
        D = torch.diag_embed(d_i, dim1=1)
        X = (D-A) / (D + eps)
        eigval, eigvec = torch.lobpcg(A=D-A, B=D, k=2, largest=False)
        second_smallest_vec = eigvec[0,:,1].cpu().numpy()

        avg = np.sum(second_smallest_vec) / len(second_smallest_vec)
        bipartition = second_smallest_vec > avg

        seed = np.argmax(np.abs(second_smallest_vec))

        if bipartition[seed] != 1:
            bipartition = np.logical_not(bipartition)

        bipartition = bipartition.astype(float)
        mask.append(bipartition)
    mask = np.stack(mask, axis=0)
    return mask
