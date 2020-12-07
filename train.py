import torch
import random
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR

from tools import utils
from evaluate import evaluate_kd


def train_kd(model, optimizer, loss_fn, dataloader, params):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: 
        dataloader:
        params: (Params) hyperparameters
    """

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):
            # move to GPU
            train_batch = train_batch.cuda(non_blocking=True)
            labels_batch = labels_batch.cuda(non_blocking=True)

            # compute model output, compute loss
            output_batch = model(train_batch)

            loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = loss_fn(output_batch, labels_batch)

            # compute gradients of all variables wrt loss
            loss.backward()

            # after going through 1 batch size of images
            if (i+1) % params.subdivisions == 0:
                # performs updates using accumulated gradients
                optimizer.step()
                # clear previous gradients
                optimizer.zero_grad()

            # compute all metrics on this batch
            summary_batch = {}
            summary_batch['loss'] = loss.data.item()
            summary_batch['loss_xy'] = loss_xy.data.item()
            summary_batch['loss_wh'] = loss_wh.data.item()
            summary_batch['loss_obj'] = loss_obj.data.item()
            summary_batch['loss_cls'] = loss_cls.data.item()
            summary_batch['loss_l2'] = loss_l2.data.item()
            summ.append(summary_batch)

            # update the average loss
            loss_avg.update(loss.data.item())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)


def train_and_evaluate_kd(model, train_dataloader, val_dataloader, optimizer, loss_fn, params, model_dir):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
    """

    # learning rate setup
    def burnin_schedule(i):
        factor = 1.0

        if i < params.burn_in:
            factor = pow(i / params.burn_in, 4)
        # elif i < params.steps[0]:
        #     factor = 1.0
        # else:
        #     factor = 1.0
        #     max_idx = max_idx = params.steps.index(max(step for step in params.steps if step <= i))
        #     for scale in params.scales[:max_idx+1]:
        #         factor *= scale
        return factor

    best_val_acc = 0.0

    scheduler = LambdaLR(optimizer, burnin_schedule)

    for epoch in range(params.num_epochs):

        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train_kd(model, optimizer, loss_fn, train_dataloader, params)

        # No need to validate in real ams, just overfit to training set
        # Evaluate for one epoch on validation set
        val_metrics = evaluate_kd(model, loss_fn, val_dataloader, params)

        scheduler.step()

        val_acc = val_metrics['AP']
        is_best = val_acc >= best_val_acc

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict()},
                               is_best=is_best,
                               checkpoint=model_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new highest AP")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = model_dir / "metrics_val_best.json"
            utils.save_dict_to_json(val_metrics, str(best_json_path))

        # Save latest val metrics in a json file in the model directory
        last_json_path = model_dir / "metrics_val_last.json"
        utils.save_dict_to_json(val_metrics, str(last_json_path))


if __name__ == '__main__':
    import argparse
    from torch import nn, optim
    from datetime import datetime
    from torchvision import models
    from model.loss import Yolo_loss
    from model.darknet2pytorch import Darknet
    from model.data_loader import fetch_dataloader

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=Path, default='experiments/yolov4_tiny_distill/yolov4_teacher', help="Directory containing params.json")
    parser.add_argument('--restore_file', type=Path, default='yolov4-tiny.pth.tar', help="Optional, name of the file in model_dir containing student weights to reload before training")

    args = parser.parse_args()
    model_dir = args.model_dir

    # Update default params with loaded parameters from json file
    json_path = model_dir / 'params.json'
    assert json_path.is_file(), "No json configuration file found at {}".format(str(json_path))
    params = utils.Params(str(json_path))

    # use GPU if available
    params.cuda = torch.cuda.is_available()
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set the random seed for reproducible experiments
    seed = 230
    random.seed(seed)
    torch.manual_seed(seed)
    if params.cuda: torch.cuda.manual_seed(seed)

    #  enable cudnn autotuner
    torch.backends.cudnn.benchmark = True

    # Set the logger
    train_logs_dir = model_dir / 'train_logs'
    train_logs_dir.mkdir(exist_ok=True)
    now = datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    utils.set_logger(str(train_logs_dir / f'train_{now}.log'))

    # train a yolov4 tiny with knowledge distillation
    if params.model_version == 'yolov4_tiny_distill':
        cfgfile = str(model_dir / 'yolov4-tiny.cfg')
        student_model = Darknet(cfgfile).cuda()
        optimizer = optim.Adam(student_model.parameters(), lr=params.learning_rate/params.batch_size, betas=(0.9, 0.999), eps=1e-08)
        loss_fn = Yolo_loss(n_classes=params.num_classes, device=params.device, batch=params.batch_size // params.subdivisions, image_size=params.width, which_yolo='yolov4-tiny')

    # Specify the pre-trained teacher models for knowledge distillation
    if params.teacher == "yolov4":
        cfgfile = str(model_dir / 'yolov4.cfg')
        teacher_model = Darknet(cfgfile).cuda()
        teacher_checkpoint = model_dir / 'yolov4.pth.tar'

    # reload student weights from restore_file if specified
    if args.restore_file is not None:
        restore_path = model_dir / args.restore_file
        logging.info("Restoring parameters from {}".format(str(restore_path)))
        utils.load_checkpoint(restore_path, student_model, optimizer)

    # reload teacher weights
    utils.load_checkpoint(teacher_checkpoint, teacher_model)

    # Log down parameters used
    logging.info("Parameters: " + str(params.dict))

    # Create the input data pipeline
    logging.info("Loading the datasets...")
    train_dl, dev_dl = fetch_dataloader(params, teacher_model)
    logging.info("- done.")

    # Train the student model with KD
    logging.info("Experiment - model version: {}".format(params.model_version))
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate_kd(student_model, train_dl, dev_dl, optimizer, loss_fn, params, model_dir)
