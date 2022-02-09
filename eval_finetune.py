# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os

import numpy as np

import torch

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
import torch.nn as nn
import CXR_dataset
from torch.utils.data import DataLoader, SequentialSampler
import utils
from vision_transformer import DINOHead, CLS_head
from torchvision import models as torchvision_models

import vision_transformer as vit_o
from main_dino import get_args_parser

parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
args = parser.parse_args()

logger = logging.getLogger(__name__)

class AverageMeter(object):
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


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, iteration, model, optimizer, scheduler, best_auc):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
    torch.save({'iteration': iteration, 'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_auc': best_auc}, model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)

def save_model_latest(args, iteration, model, optimizer, scheduler, best_auc):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint_latest.bin" % args.name)
    torch.save({'iteration': iteration, 'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_auc': best_auc}, model_checkpoint)
    logger.info("Saved latest model checkpoint to [DIR: %s]", args.output_dir)

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

def valid(args, model, writer, test_loader):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", 1)

    model.eval()

    pred, true = [], []
    y_score_1 = []

    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])

    loss_fct = nn.BCEWithLogitsLoss()

    for step, batch in enumerate(epoch_iterator):
        x, y = batch
        x = x.to(args.device)
        y = y.to(args.device).float()

        with torch.no_grad():
            output = model(x)
            labels = y

            loss = loss_fct(output.view(-1), labels.view(-1))
            eval_losses.update(loss.item())

            output = torch.sigmoid(output + 0.0)
            prob_np = output.detach().cpu().numpy()
            preds = np.round(prob_np)

            for x in range(len(labels)):
                true.append(np.asarray(labels.cpu())[x])
            for x in range(len(labels)):
                pred.append(np.asarray(preds)[x])

            # Calculate score for AUC
            for x in range(len(labels)):
                y_sc = prob_np[x]
                y_score_1.append(y_sc)

        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    y_score = np.array(y_score_1)

    auc = roc_auc_score(true, y_score)

    logger.info("\n")
    logger.info("External validation Results")
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("External validation AUC (TB): %2.5f" % auc)

    logger.info(
        classification_report(y_true=true, y_pred=pred, target_names=['Normal', 'Tuberculosis'],
                              digits=4, labels=list(range(2))))

    writer.add_scalar("test/loss", scalar_value=eval_losses.avg)
    return eval_losses.avg, auc


def evaluate(args, model):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    # Prepare dataset
    testset = CXR_dataset.CXR_Dataset(args.data_path, transforms=None, mode='test', labeled=True)

    test_sampler = SequentialSampler(testset)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=1,
                             num_workers=8,
                             pin_memory=True) if testset is not None else None

    # Load weights
    state_dict = torch.load(args.pretrained_dir, map_location="cpu")
    print("Take key {} in provided checkpoint dict".format(args.checkpoint_key))
    state_dict = state_dict[args.checkpoint_key]
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    msg_s = model.load_state_dict(state_dict, strict=False)
    print('Weights found at {} and loaded with msg: {}'.format('CheXpert and pre-training', msg_s))

    loss, auc = valid(args, model, writer, test_loader)

    logger.info("Best AUC: \t%f" % auc)
    logger.info("End Validation!")


def main():
    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    # Define model and load weights
    if 'vit' in args.arch:
        model = vit_o.__dict__[args.arch](patch_size=args.patch_size)
        embed_dim = model.embed_dim
        inter_dim = 384

    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch]()

        if 'resne' in args.arch:
            embed_dim = model.fc.weight.shape[1]
            inter_dim = 2048
        elif 'densenet' in args.arch:
            embed_dim = model.classifier.weight.shape[1]
            inter_dim = 1920
        elif 'eff' in args.arch:
            embed_dim = model.classifier[1].weight.shape[1]
            inter_dim = 1792

    model = utils.MultiCropWrapper(
        model,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head), CLS_head(inter_dim, 256, 1), args)

    model = model.cuda()

    # Training
    evaluate(args, model)


if __name__ == "__main__":
    main()
