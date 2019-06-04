## [High Quality Monocular Depth Estimation via Transfer Learning (arXiv 2018)](https://arxiv.org/abs/1812.11941)
**[Ibraheem Alhashim](https://ialhashim.github.io/)** and **Peter Wonka**


## Requirements
* This code is tested with Keras 2.2.4, Tensorflow 1.13, CUDA 9.0, on a machine with an NVIDIA Titan V and 16GB+ RAM running on Windows 10 or Ubuntu 16.
* Other packages needed `keras pillow matplotlib scikit-learn scikit-image opencv-python pydot` and `GraphViz`.


## Data
* [NYU Depth V2 (50K)](https://s3-eu-west-1.amazonaws.com/densedepth/nyu_data.zip) (4.1 GB): You don't need to extract the dataset since the code loads the entire zip file into memory when training.

## Training with DenseNet 169 encoder
* Train from scratch: 
```
python train.py --data nyu --bs 5 --full 
```

* Train from a previous checkpoint 
```
python train.py --data nyu --bs 5 --full --checkpoint ./models/1557344811-n10138-e20-bs5-lr0.0001-densedepth_nyu/weights.04-0.12.h5
```

## Training with DenseNet 121 encoder
```
python train.py --data nyu --bs 5 --full --dnetVersion small
```

## Training with ResNet50 encoder
```
python train.py --data nyu --bs 5 --name resnet50_nyu --full --resnet
```

## Evaluation
* Download, but don't extract, the ground truth test data from [here](https://s3-eu-west-1.amazonaws.com/densedepth/nyu_test.zip) (1.4 GB). The call evaluate.py with your model checkpoint

```
python evaluate.py --model ./models/1557483797-n10138-e20-bs5-lr0.0001-densedepth_nyu/weights.06-0.12.h5
```

## Reference
Corresponding paper to cite:
```
@article{Alhashim2018,
  author    = {Ibraheem Alhashim and Peter Wonka},
  title     = {High Quality Monocular Depth Estimation via Transfer Learning},
  journal   = {arXiv e-prints},
  volume    = {abs/1812.11941},
  year      = {2018},
  url       = {https://arxiv.org/abs/1812.11941},
  eid       = {arXiv:1812.11941},
  eprint    = {1812.11941}
}
```
