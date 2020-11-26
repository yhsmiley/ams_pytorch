# Adaptive Model Streaming in PyTorch

**Still very much work in progress**

TODO:
- [ ] convert to object detection using yolov4
- [ ] mlflow/tensorboard to visualize training progress
- [ ] database for easy retrieval of wanted images

## Features

* Hyperparameters defined by "params.json"
* Progress bar, checkpoint saving/loading (utils.py)
* Pretrained teacher models available for download 


## Install

* Build docker image
  ```
  docker build -t ams_pytorch .
  ```


## Organization:

* train.py: main entrance for train/eval
* experiments/: json files for each experiment
* model/: teacher and student DNNs, dataloader 


## Key notes about usage for your experiments:

* Download pretrained teacher model checkpoints from [here](https://pytorch.org/docs/stable/torchvision/models.html) and extract into experiments folder
* Download some sample imagenet data and extract into data-imagenet folder (separate train and test subfolders).
* Call train.py to start training ResNet18 with state-of-the-art deeper models distilled
* Hyperparameters are defined in params.json files.


## Train (Dataset: ImageNet)

Note: all the hyperparameters can be found and modified in 'params.json' under 'model_dir'

- Train a ResNet18 model with knowledge distilled from a pre-trained Densenet teacher
```
python3 train.py --model_dir experiments/resnet18_distill/densenet161_teacher
```


## References

[Real-Time Video Inference on Edge Devices via
Adaptive Model Streaming](https://arxiv.org/abs/2006.06628)

https://github.com/modelstreaming/ams

https://github.com/peterliht/knowledge-distillation-pytorch
