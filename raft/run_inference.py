import os
import glob as gb
import numpy as np

data_path = "../data/DAVIS"
gap = [1] # gap = 3 for FBMS
reverse = [0]
rgbpath = data_path + '/JPEGImages/480p'  # path to the dataset
folder = gb.glob(os.path.join(rgbpath, '*'))

for r in reverse:
  for g in gap:
    for f in folder:
      print('===> Runing {}, gap {}'.format(f, g))
      mode = 'raft-things.pth'  # model
      if r==1:
        raw_outroot = data_path + '/RAFT_Flows_gap{}/480p'.format(g)  # where to raw flow
        outroot = data_path + '/RAFT_FlowImages_gap{}/480p'.format(g)  # where to save the image flow
      elif r==0:
        raw_outroot = data_path + '/RAFT_Flows_gap{}/480p'.format(g)   # where to raw flow
        outroot = data_path + '/RAFT_FlowImages_gap{}/480p'.format(g)   # where to save the image flow
      os.system("python predict.py "
                "--gap {} --model {} --path {} "
                "--outroot {} --reverse {} --raw_outroot {} ".format(g, mode, f, outroot, r, raw_outroot))

