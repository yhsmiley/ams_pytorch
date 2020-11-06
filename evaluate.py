"""Evaluates the model"""

import torch
import logging
import numpy as np
from torch.autograd import Variable
from model.metrics import metrics


def evaluate_kd(model, loss_fn, dataloader, metrics, params):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []

    # compute metrics over the dataset
    for i, (data_batch, labels_batch) in enumerate(dataloader):

        # move to GPU if available
        if params.cuda:
            data_batch, labels_batch = data_batch.cuda(non_blocking=True), labels_batch.cuda(non_blocking=True)
        # fetch the next evaluation batch
        data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)
        
        # compute model output
        output_batch = model(data_batch)

        loss = loss_fn(output_batch, labels_batch)
        # loss = 0.0  #force validation loss to zero to reduce computation time

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data.cpu().numpy()
        labels_batch = labels_batch.data.cpu().numpy()

        # compute all metrics on this batch
        summary_batch = {metric: metrics[metric](output_batch, labels_batch) for metric in metrics}
        summary_batch['loss'] = loss.data.item()
        # summary_batch['loss'] = loss
        summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]} 
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean


if __name__ == '__main__':
    """
    Evaluate the model on a dataset for one pass.
    """

    import os
    import utils
    import argparse
    import torchvision.models as models
    import model.data_loader as data_loader

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='experiments/resnet18_distill/densenet161_teacher', help="Directory of params.json")
    parser.add_argument('--restore_file', default='best.pth.tar', help="name of the file in model_dir containing weights to load")

    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()     # use GPU is available

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda: torch.cuda.manual_seed(230)
        
    # Get the logger
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Loading the dataset...")

    if params.teacher == "densenet161":
        teacher_model = models.densenet161().cuda()
        teacher_checkpoint = 'experiments/base_densenet161/pretrained.pth.tar'
    elif params.teacher == "resnext50":
        teacher_model = models.resnext50_32x4d().cuda()
        teacher_checkpoint = 'experiments/base_resnext50/pretrained.pth.tar'
    utils.load_checkpoint(teacher_checkpoint, teacher_model)

    # fetch dataloaders
    train_dl, dev_dl = data_loader.fetch_dataloader(params, teacher_model)

    logging.info("- done.")

    # Define the model graph
    if params.model_version == 'resnet18_distill':
        model = models.resnet18().cuda() if params.cuda else models.resnet18()
    
    logging.info("Starting evaluation...")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file), model)

    # Evaluate
    test_metrics = evaluate_kd(model, dev_dl, metrics, params)
    save_path = os.path.join(args.model_dir, "metrics_test_{}.json".format(args.restore_file.split('.')[0]))
    utils.save_dict_to_json(test_metrics, save_path)
