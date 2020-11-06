import os
import torch
import random
import logging
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

import utils
from evaluate import evaluate_kd


def train_kd(model, optimizer, loss_fn, dataloader, metrics, params):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: 
        dataloader: 
        metrics: (dict) 
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
            # move to GPU if available
            if params.cuda:
                train_batch = train_batch.cuda(non_blocking=True)
                labels_batch = labels_batch.cuda(non_blocking=True)
            # convert to torch Variables
            train_batch = Variable(train_batch)
            labels_batch = Variable(labels_batch)

            # compute model output, compute loss
            output_batch = model(train_batch)

            loss = loss_fn(output_batch, labels_batch)

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()

                # compute all metrics on this batch
                summary_batch = {metric:metrics[metric](output_batch, labels_batch) for metric in metrics}
                summary_batch['loss'] = loss.data.item()
                summ.append(summary_batch)

            # update the average loss
            loss_avg.update(loss.data.item())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)


def train_and_evaluate_kd(model, train_dataloader, val_dataloader, optimizer, loss_fn, metrics, params, model_dir, restore_file=None):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) - file to restore (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(model_dir, restore_file)
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0

    # learning rate schedulers:
    if params.model_version == "resnet18_distill":
        scheduler = StepLR(optimizer, step_size=50, gamma=0.1, verbose=True)

    for epoch in range(params.num_epochs):

        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train_kd(model, optimizer, loss_fn, train_dataloader, metrics, params)

        # Evaluate for one epoch on validation set
        val_metrics = evaluate_kd(model, loss_fn, val_dataloader, metrics, params)

        scheduler.step()

        val_acc = val_metrics['accuracy']
        is_best = val_acc>=best_val_acc

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict()},
                               is_best=is_best,
                               checkpoint=model_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)


if __name__ == '__main__':
    import argparse
    import torch.nn as nn
    import torch.optim as optim
    import torchvision.models as models
    import model.data_loader as data_loader
    from model.metrics import metrics

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='experiments/resnet18_distill/densenet161_teacher', help="Directory containing params.json")
    parser.add_argument('--restore_file', default=None, help="Optional, name of the file in model_dir containing weights to reload before training")

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    random.seed(230)
    torch.manual_seed(230)
    if params.cuda: torch.cuda.manual_seed(230)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # train a 18-layer ResNet with knowledge distillation
    if params.model_version == 'resnet18_distill':
        model = models.resnet18().cuda() if params.cuda else models.resnet18()
        optimizer = optim.SGD(model.parameters(), lr=params.learning_rate, momentum=0.9, weight_decay=5e-4)
        loss_fn = nn.CrossEntropyLoss()

    # Specify the pre-trained teacher models for knowledge distillation
    if params.teacher == "densenet161":
        teacher_model = models.densenet161().cuda()
        teacher_checkpoint = 'experiments/base_densenet161/pretrained.pth.tar'

    elif params.teacher == "resnext50":
        teacher_model = models.resnext50_32x4d().cuda()
        teacher_checkpoint = 'experiments/base_resnext50/pretrained.pth.tar'

    utils.load_checkpoint(teacher_checkpoint, teacher_model)

    # Create the input data pipeline
    logging.info("Loading the datasets...")
    train_dl, dev_dl = data_loader.fetch_dataloader(params, teacher_model)
    logging.info("- done.")

    # Train the model with KD
    logging.info("Experiment - model version: {}".format(params.model_version))
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate_kd(model, train_dl, dev_dl, optimizer, loss_fn, metrics, params, args.model_dir, args.restore_file)
