"""Evaluates the model"""

import cv2
import torch
import logging
import numpy as np
from model.metrics import metrics
from tools.coco_eval import CocoEvaluator
from tools.coco_utils import convert_to_coco_api


def evaluate_kd(model, loss_fn, dataloader, params):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    model.eval()

    # COCO Evaluation
    evaluator = evaluate(model, dataloader, loss_fn, params)
    stats = evaluator.coco_eval['bbox'].stats

    coco_summ = {'AP': stats[0], 'AP50': stats[1], 'AP75': stats[2], 'AP_small': stats[3], 'AP_medium': stats[4], 'AP_large': stats[5], 'AR1': stats[6], 'AR10': stats[7], 'AR100': stats[8], 'AR_small': stats[9], 'AR_medium': stats[10], 'AR_large': stats[11]}
    coco_metrics = " ; ".join(f"{k}: {v:05.3f}" for k, v in coco_summ.items())
    logging.info("- COCO Eval metrics : " + coco_metrics)

    # # summary for current eval loop
    # summ = []

    # # compute metrics over the dataset
    # for data_batch, labels_batch in dataloader:
    #     # move to GPU
    #     data_batch = data_batch.cuda(non_blocking=True)
    #     labels_batch = labels_batch.cuda(non_blocking=True)
        
    #     # compute model output
    #     output_batch = model(data_batch)

    #     # compute all metrics on this batch
    #     summary_batch = {}
    #     summary_batch['loss'] = loss.data.item()
    #     summ.append(summary_batch)

    # # compute mean of all metrics in summary
    # metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]} 
    # metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    # logging.info("- Eval metrics : " + metrics_string)

    # return metrics_mean
    return coco_summ

@torch.no_grad()
def evaluate(model, data_loader, loss_fn, params):
    coco_ds = convert_to_coco_api(data_loader.dataset, bbox_fmt='coco')
    coco_evaluator = CocoEvaluator(coco_ds, iou_types = ["bbox"], bbox_fmt='coco')

    # summary for current eval loop
    summ = []

    for images, targets in data_loader:
        model_input = [np.array(cv2.resize(img, (params.width, params.height))) for img in images]
        model_input = np.stack(model_input, axis=0)
        model_input = np.divide(model_input, 255, dtype=np.float32)
        model_input = torch.from_numpy(model_input.transpose(0, 3, 1, 2))
        model_input = model_input.cuda(non_blocking=True).contiguous()

        outputs = model(model_input)

        # all_targets = []
        # for target in targets:
        #     if len(target['boxes']):
        #         boxes = target['boxes'].copy()
        #         boxes[..., 2:] = boxes[..., 2:] + boxes[..., :2]
        #         labels = np.expand_dims(target['labels'], axis=1)
        #         boxes = np.hstack([boxes, labels])

        #         out_bboxes = np.zeros([params.boxes, 5])
        #         out_bboxes[:min(boxes.shape[0], params.boxes)] = boxes[:min(boxes.shape[0], params.boxes)]

        #         all_targets.append(out_bboxes)
        #     else:
        #         out_bboxes = np.zeros([params.boxes, 5])
        #         all_targets.append(out_bboxes)

        # all_targets = torch.from_numpy(np.array(all_targets))
        # all_targets = all_targets.cuda(non_blocking=True)

        # loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = loss_fn(outputs, all_targets)

        # # compute all metrics on this batch
        # summary_batch = {}
        # summary_batch['loss'] = loss.data.item()
        # summary_batch['loss_xy'] = loss_xy.data.item()
        # summary_batch['loss_wh'] = loss_wh.data.item()
        # summary_batch['loss_obj'] = loss_obj.data.item()
        # summary_batch['loss_cls'] = loss_cls.data.item()
        # summary_batch['loss_l2'] = loss_l2.data.item()
        # summ.append(summary_batch)

        res = {}
        for img, target, boxes, confs in zip(images, targets, outputs[0], outputs[1]):
            img_height, img_width = img.shape[:2]
            # boxes = output[...,:4].copy()  # output boxes in yolo format
            boxes = boxes.squeeze(2).cpu().detach().numpy()
            boxes[...,2:] = boxes[...,2:] - boxes[...,:2] # Transform [x1, y1, x2, y2] to [x1, y1, w, h]
            boxes[...,0] = boxes[...,0] * img_width
            boxes[...,1] = boxes[...,1] * img_height
            boxes[...,2] = boxes[...,2] * img_width
            boxes[...,3] = boxes[...,3] * img_height
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # confs = output[...,4:].copy()
            confs = confs.cpu().detach().numpy()
            labels = np.argmax(confs, axis=1).flatten()
            labels = torch.as_tensor(labels, dtype=torch.int64)
            scores = np.max(confs, axis=1).flatten()
            scores = torch.as_tensor(scores, dtype=torch.float32)
            res[target["image_id"]] = {
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
            }
        coco_evaluator.update(res)

    # gather the stats from all processes
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    return coco_evaluator


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
    test_metrics = evaluate_kd(model, dev_dl, params)
    save_path = os.path.join(args.model_dir, "metrics_test_{}.json".format(args.restore_file.split('.')[0]))
    utils.save_dict_to_json(test_metrics, save_path)
