# Adaptive Model Streaming in PyTorch

**Still very much work in progress**

TODO:
- [x] convert to object detection using yolov4
- [ ] mlflow/tensorboard to visualize training progress
- [ ] database for easy retrieval of wanted images

## Features

* Hyperparameters defined by "params.json"
* Progress bar, checkpoint saving/loading (tools/utils.py)
* Pretrained Yolov4 models available for download


## Install

* Build docker image
  ```
  docker build -t ams_pytorch .
  ```


## Organization:

* train.py: main entrance for train/eval
* experiments/: json files for each experiment
* model/: teacher and student networks, dataloader 


## Key notes about usage for your experiments:

* Download some sample imagenet data (make sure these classes also exist in coco) and extract into data-imagenet folder (separate train and test subfolders).
* Call train.py to start training Yolov4-tiny with Yolov4
* Hyperparameters are defined in params.json files.


## References

[Real-Time Video Inference on Edge Devices via
Adaptive Model Streaming](https://arxiv.org/abs/2006.06628)

https://github.com/modelstreaming/ams

https://github.com/peterliht/knowledge-distillation-pytorch
