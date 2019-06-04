# FuseNet

This repository contains PyTorch implementation of FuseNet-SF5 architecture from the paper
[FuseNet: incorporating depth into semantic segmentation via fusion-based CNN architecture](https://pdfs.semanticscholar.org/9360/ce51ec055c05fd0384343792c58363383952.pdf). 


## Installation
Prerequisites:
- python 3.6
- Nvidia GPU + CUDA cuDNN

## Datasets 

### [NYU-Depth V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
- Simply, create a directory named datasets in the main project directory and in datasets directory download the preprocessed dataset, in HDF5 format, with 40 semantic-segmentation and 10 scene classes here: [train + test set](https://vision.in.tum.de/webarchive/hazirbas/fusenet-pytorch/nyu/nyu_class_10_db.h5)
- Preprocessed dataset contains 1449 (train: 795, test: 654) RGB-D images with 320x240 resolution, their semantic-segmentation and scene-type annotations.
- Depth image values have been normalized so that they fall into 0-255 range. 

## Training
- To train FuseNet, run `fusenet_train.py` by providing the path of the dataset. 
- If you would like to train a FuseNet model with the classification head, provide `--use_class True`
- Example training commands can be found below.

### Training from scratch

```
python fusenet_train.py --dataroot ./datasets/nyu_class_10_db.h5 --batch_size 8 --lr 0.005 --num_epochs 125
```

### Resuming training from a checkpoint
```

python fusenet_train.py --dataroot ./datasets/nyu_class_10_db.h5 --resume_train True --batch_size 8 \
                        --load_checkpoint ./checkpoints/may27_first_run/nyu/best_model.pth.tar --lr 0.005 --num_epochs 25
```

## Inference
- Model's semantic segmentation performance on the given dataset will be evaluated in three accuracy measures: global pixel-wise classification accuracy, 
intersection over union, and mean accuracy.
- vis_results is used to visualize the results on the test set
- Example run command:
```
python fusenet_test.py --dataroot ./datasets/nyu_class_10_db.h5 --load_checkpoint ./checkpoints/rgb_only/nyu/best_model.pth.tar --vis_results True
```



## Citing FuseNet
Caner Hazirbas, Lingni Ma, Csaba Domokos and Daniel Cremers, _"FuseNet: Incorporating Depth into Semantic Segmentation via Fusion-based CNN Architecture"_, in proceedings of the 13th Asian Conference on Computer Vision, 2016.

    @inproceedings{fusenet2016accv,
     author    = "C. Hazirbas and L. Ma and C. Domokos and D. Cremers",
     title     = "FuseNet: incorporating depth into semantic segmentation via fusion-based CNN architecture",
     booktitle = "Asian Conference on Computer Vision",
     year      = "2016",
     month     = "November",
    }
