Python
(Official version is too old)
 * Cython==0.24
 * PyYAML==3.11
 * argparse==1.2.1
 * easydict==1.6
 * future==0.15.2
 * h5py==2.6.0
 * matplotlib==1.5.1
 * numpy==1.11.0
 * prettytable==0.7.2
 * scikit-image==0.12.3
 * scipy==0.17.0

(Higher version is also valid)
 * Cython==0.29.28
 * PyYAML==5.1.2
 * easydict==1.9
 * future==0.18.2
 * h5py==3.1.0
 * matplotlib==3.3.4
 * numpy==1.19.5
 * prettytable==2.5.0
 * scikit-image==0.15.0
 * scipy==1.5.4
 * imageio==2.5.0

Installation
--------------
Python:

1. pip install -r python/requirements.txt(or you can install the libraries manually, **doesn't need to be the same version as indicated above**)
2. export PYTHONPATH=$(pwd)/python/lib
3. `cd data`
4. `sh get_davis.sh`(or creat a symbolic link to your DAVIS path)
5. `sh get_davis_results.sh`(I used the authors' prediciton for evaluation)
6. evaluation: `cd python; python tools/eval.py ../data/DAVIS/Results/Segmentations/480p/tokencut/ ./`(the first path to the prediction, if you want to use another path, please check the `ROOT/python/lib/davis/configure.py`)
7. evaluation results stored in: `./python/tokencut.h5` 
8. Evaluation h5 file: `python tools/eval_view.py  ./tokencut.h5 --eval_set test`
<!-- 9. (optional) To show the results of each sequence: `cd python; python experiments/eval_sequences.py` -->

Documentation
----------------
See source code for documentation.

The directory is structured as follows:

 * `ROOT/python/tools`: contains scripts for evaluating segmentation.
     - `eval.py` : evaluate a technique and store results in HDF5 file

 * `ROOT/python/lib/davis`: contains scripts for evaluating segmentation.
     - `configure.py` : configuration for data path (**Check The current JPEGImage path is 1080p, Annotation and Segementation path is 480p**)

 * `ROOT/python/experiments`: contains several demonstrative examples.

 * `ROOT/data` :
     - `get_davis.sh`: download input images and annotations.
     - `get_davis_cvpr2016_results.sh`: download the CVPR 2016 submission results.

