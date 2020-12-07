# -*- coding: utf-8 -*-
from easydict import EasyDict

Cfg = EasyDict()

Cfg.batch_size = 64
Cfg.subdivisions = 16
Cfg.num_epochs = 100
Cfg.width = 608
Cfg.height = 608

Cfg.learning_rate = 0.00261
Cfg.burn_in = 10 # initial burn_in will be processed for the first x iterations, current_learning rate = learning_rate * pow(iterations / burn_in, power) = 0.001 * pow(iterations/1000, 4)
# Cfg.steps = [60, 80] # at these numbers of iterations the learning rate will be multiplied by scales factor
# Cfg.scales = [0.1, 0.1]

Cfg.momentum = 0.9
Cfg.decay = 0.0005
Cfg.saturation = 1.5
Cfg.exposure = 1.5
Cfg.hue = 0.1

Cfg.letter_box = 0 # keeps aspect ratio of loaded images during training
Cfg.jitter = 0.2 # randomly changes size of image and its aspect ratio from x(1 - 2*jitter) to x(1 + 2*jitter)
Cfg.classes = 80
Cfg.flip = 1
Cfg.blur = 0
Cfg.gaussian = 0
Cfg.boxes = 60  # number of bboxes

Cfg.teacher_bs = 64
Cfg.teacher_bbox_thresh = 0.5
Cfg.teacher_nms_thresh = 0.4

Cfg.num_classes = 80
Cfg.num_workers = 0

Cfg.keep_checkpoint_max = 10
