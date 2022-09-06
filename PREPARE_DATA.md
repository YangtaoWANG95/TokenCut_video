## Download dataset

### DAVIS2016

To download at [DAVIS](https://davischallenge.org/davis2016/code.html).

```
wget https://graphics.ethz.ch/Downloads/Data/Davis/DAVIS-data.zip
unzip DAVIS-data.zip
```
The dataset should be organized as 
```
./data/DAVIS/├── JPEGImages/
                ├── 1080p/(50 category folders)
                ├── 480p/(50 category folders)
             ├── Annotations/
                ├── 1080p/(50 category folders)
                ├── 480p/(50 category folders)
             ├── FlowImages/
                ├── 1080p/(50 category folders)
                ├── 480p/(50 category folders)
             ├── train_vid.npy
             ├── val_vid.npy
                        ...
```


### FBMS

To download at [FBMS](https://lmb.informatik.uni-freiburg.de/resources/datasets/)

```
wget https://lmb.informatik.uni-freiburg.de/resources/datasets/fbms/FBMS_Trainingset.zip
wget https://lmb.informatik.uni-freiburg.de/resources/datasets/fbms/FBMS_Testset.zip
python ./data/FBMS/FBMS_clean.py
```

The dataset should be organized as 
```
./data/FBMS/ ├── JPEGImages/
             ├── Annotations/
             ├── train_vid.npy
             ├── val_vid.npy
             ├── RAFT_FlowImages_gap3
             ├── ARFlow_FlowImages_gap3 (optinal)
                        ...
```

### SegTrackv2

To download at [SegTrackv2](https://web.engr.oregonstate.edu/~lif/SegTrack2/dataset.html)

```
wget https://web.engr.oregonstate.edu/~lif/SegTrack2/SegTrackv2.zip
python ./data/SegTrackv2/SegTrack_clean.py
```

The dataset should be organized as 
```
./data/SegTrackv2/├── GroundTruth/
                  ├── ImageSets/
                  ├── train_vid.npy
                  ├── val_vid.npy
                  ├── RAFT_FlowImages_gap1
                  ├── ARFlow_FlowImages_gap1 (optinal)
                        ...
```
### Generate Optical Flow using RAFT
Taken from: https://github.com/charigyang/motiongrouping/tree/main/raft

Please modify the datapath in `run_inference.py` and generate optical flow for each dataset separately. Note that using `gap=3` for`FBMS` dataset.
```
cd raft
python run_inference.py
```

### Generate Optical Flow using ARFlow
Download ARFlow from: https://github.com/lliuz/ARFlow.

Please follow the instruction in [ARFlow](https://github.com/lliuz/ARFlow), to generate the environment.
Their code has been developed under Python3, PyTorch 1.1.0 and CUDA 9.0 on Ubuntu 16.04.

```
# Install python packages
pip3 install -r requirements.txt
```


Please replace the original `inference.py`, modify the datapath in `run_inference.py` and generate optical flow for each dataset separately. Note that using `gap=3` for`FBMS` dataset.

```
git clone https://github.com/lliuz/ARFlow.git
cd ARFlow
python run_inference.py
```
