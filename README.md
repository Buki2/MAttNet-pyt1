# PyTorch v1.x Implementation of MAttNet

This is not the official code!

The official code is <https://github.com/lichengunc/MAttNet>  
Please follow their citation and license information.  
Thank them for sharing such useful code :D

## Build for Pytorch 1.x

### Done

- Update the code from python2 to python3
- Update the code from pytorch 0.x to 1.x
- Replace the source code of pyutils/mask-faster-rcnn with pyutils/pytorch-faster-rcnn <https://github.com/ruotianluo/pytorch-faster-rcnn>
  - i.e., replace all the functions or packages from mask-faster-rcnn with that of pytorch-faster-rcnn

### Notice

<!-- Incomplete replacement (just being able to execute extract_mrcn_head_feats.py and extract_mrcn_ann_feats.py) -->

This version of the code can work in my environment. If there are some bugs, we can communicate in the issue section.

Apologize for the lack of elegance in the code and commands, but I don't have much time to sort them out. :(

## Let's begin!

### Prerequisites

```
conda create -n matt python=3.7
source activate matt
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=9.0 -c pytorch
```

### Data preparation

```
data
├── images
│   └── mscoco
│       └── images
│           └── train2014  # contains images
├── refcoco
│   ├── instances.json
│   ├── refs(google).p
│   └── refs(unc).p
├── refcoco+
│   ├── instances.json
│   └── refs(unc).p
└── refcocog
    ├── instances.json
    ├── refs(google).p
    └── refs(umd).p
```

### Installation

1. Clone this repository
2. Prepare the submodule *refer*

(Please refer to <https://github.com/lichengunc/refer> for details.)

```
## go to this directory
pip install Cython
make
```

3. Prepare the submodule *pytorch-faster-rcnn*

(Please refer to <https://github.com/ruotianluo/pytorch-faster-rcnn> for details.)

```
## go to this directory
cd data
git clone http://github.com/pdollar/coco.git
cd coco/PythonAPI
make
cd ../../..
mkdir -p output/res101/coco_2014_train_minus_refer_valtest+coco_2014_valminusminival/notime
cd output/res101/coco_2014_train_minus_refer_valtest+coco_2014_valminusminival/notime/
## download and put the file res101_mask_rcnn_iter_1250000.pth here
cd ../../../..
cd data/coco
mkdir annotations
cd annotations
## download and put the file instances_minival2014.json here

## back to the main directory of MAttNet
pip install pyyaml
pip install scipy==1.2.1
pip install pillow
pip install scikit-image
pip install opencv-python
pip install runipy
pip install matplotlib
pip install opencv-python-headless
pip install easydict==1.6
pip install tensorboardX
## You could use cv/mrcn_detection.ipynb to test if you've get Mask R-CNN ready.
runipy cv/mrcn_detection.ipynb
```

The pre-trained model *res101_mask_rcnn_iter_1250000.pth* can be found at <http://bvision.cs.unc.edu/licheng/MattNet/pytorch_mask_rcnn/res101_mask_rcnn_iter_1250k.zip> (Please refer to <https://github.com/lichengunc/mask-faster-rcnn> for details.)   
The annotations *instances_minival2014.json* can be found at <https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip?dl=0> (Please refer to <https://github.com/rbgirshick/py-faster-rcnn/blob/master/data/README.md> for details.)

4. Prepare the submodule *refer-parser2*

(Please refer to <https://github.com/lichengunc/refer-parser2> for details.)

```
## go to this directory
cd cache
wget http://bvision.cs.unc.edu/licheng/MattNet/refer-parser2/cache/parsed_atts.zip
chmod -R 775 parsed_atts.zip
unzip parsed_atts.zip
```


### Prepare some files for training

1. well-organized annotation files

```
python tools/prepro.py --dataset refcoco --splitBy unc

# output:  
# cache/prepro/data.json  
# cache/prepro/data.h5
```

2. features of each image
```
CUDA_VISIBLE_DEVICES=GPU_ID python tools/extract_mrcn_head_feats.py --dataset refcoco --splitBy unc

# output:  
# cache/feats/refcoco_unc/mrcn/res101_coco_minus_refer_notime/xxx.h5
```
<!-- before it:
```
## copy mask-faster-rcnn/lib/utils/mask_utils.py to the corresponding directory in pytorch-faster-rcnn
``` -->

3. features of the **ground-truth** detection boxes

```
CUDA_VISIBLE_DEVICES=GPU_ID python tools/extract_mrcn_ann_feats.py --dataset refcoco --splitBy unc

# output:  
# cache/feats/refcoco_unc/mrcn/res101_coco_minus_refer_notime_ann_feats.h5
```

4. features of the **real** detection boxes

only needed if you want to evaluate the automatic comprehension

*I think this step is necessary, because there are no ground truth of all the detection boxes in the real scene*

```
CUDA_VISIBLE_DEVICES=GPU_ID python tools/run_detect.py --dataset refcoco --splitBy unc --conf_thresh 0.65
CUDA_VISIBLE_DEVICES=GPU_ID python tools/extract_mrcn_det_feats.py --dataset refcoco --splitBy unc
```

### Training

```
./experiments/scripts/train_mattnet.sh GPU_ID refcoco unc
```

### Evaluation

```
./experiments/scripts/eval_easy.sh GPU_ID refcoco unc
./experiments/scripts/eval_dets.sh GPU_ID refcoco unc
```

## Contribution
Yuqi Bu

Xin Wu
