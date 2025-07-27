import torch
from torch._utils import ExceptionWrapper
import logging
import numpy as np
import os

def get_logger(filename=None):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(asctime)s - %(levelname)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
    if filename is not None:
        handler = logging.FileHandler(filename)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logging.getLogger().addHandler(handler)
    return logger


def save_model(epoch, args, logger, model, optimizer=None, loss=None, best_model=False, best_name="best"):
    if args.save_model == 0:
        return
    model = model.module if hasattr(model, 'module') else model
    if best_model:
        model_state_file = os.path.join(args.path_log, f"pytorch_model.bin.{best_name}")
    else:
        model_state_file = os.path.join(args.path_log, f"pytorch_model.bin.{epoch}")
    torch.save({
        "epoch": epoch,
        "loss": loss if loss is not None else "None",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else "None"
    }, model_state_file)
    logger.info("Model saved to %s", model_state_file)
    return model_state_file

def load_model(args, logger, model, stage, optimizer=None):
    model = model.module if hasattr(model, 'module') else model
    if args.resume_path is not None:
        model_state_file = args.resume_path
    else:
        if stage == 1:
            model_state_file = args.load_retrieval_model_path
        elif stage == 2:
            model_state_file = args.load_grounding_model_path
        elif stage == 0:
            model_state_file = args.load_uni_model_path
        else:
            raise ValueError("Invalid stage")
    checkpoint = torch.load(model_state_file, map_location='cpu')
    # load architecture params from checkpoint.
    model.load_state_dict(checkpoint['model_state_dict']) if 'model_state_dict' in checkpoint else model.load_state_dict(checkpoint)
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    resume_epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0
    resume_loss = checkpoint['loss'] if 'loss' in checkpoint else 0
    if args.local_rank == 0:
        logger.info("Model loaded from %s", model_state_file)
    return model, optimizer, resume_epoch, resume_loss

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]