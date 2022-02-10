import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path
import csv

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import models as torchvision_models

import utils
import vision_transformer as vit_o
from vision_transformer import DINOHead, CLS_head

torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument('--use_original', default=True, type=bool,
                        help="Whether to use original ViT (DINO, not ours) model or not")
    parser.add_argument('--fine_tune', default=False, type=bool,
                        help="Fine-tune network or not")
    parser.add_argument('--arch', default='vit_small', type=str,
                        choices=['vit_tiny', 'vit_small', 'vit_base'] + torchvision_archs + torch.hub.list("facebookresearch/xcit"),
                        help="""Name of architecture to train. For quick experiments with ViTs,
            we recommend using vit_small.""")
    parser.add_argument('--patch_size', default=8, type=int, help="""Size in pixels    # Original - 8, ours - 16
            of input square patches - default 16 (for 16x16 patches). Using smaller
            values leads to better performance but requires more memory. Applies only
            for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
            mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.9995, type=float, help="""Ba``se EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.01, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.01, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=8, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=5, type=int, help='Number of epochs of training.')
    parser.add_argument('--ssl_epoch', default=5, type=int, help='Number of epochs of self-supervised learning.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.00005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=1, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.75, 1.),
                        help="""Scale range of the cropped save before resizing, relatively to the origin save.
                Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
                recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
                local views to generate. Set this parameter to 0 to disable multi-crop training.
                When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.2, 0.6),
                        help="""Scale range of the cropped save before resizing, relatively to the origin save.
            Used for small local view cropping of multi-crop.""")

    # Misc
    # PATH TO DATA
    parser.add_argument('--data_path', default='/PATH/DATA/', type=str,
        help='Please specify the directory to your data')
    # PATH TO SAVE MODEL WEIGHTS
    parser.add_argument('--output_dir', default="/PATH/SAVE/", type=str,
                        help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")

    parser.add_argument('--total_folds', type=int, default=0,
                        help='Total folds to include')
    parser.add_argument("--pretrained_dir", type=str, default='./pretrained_weights/pertrain.ckpt',
                        help="Where to search for pretrained ViT models on CheXpert features.")
    parser.add_argument("--checkpoint_key", default="student", type=str,
                        help='Key to use in the checkpoint (example: "student", "teacher")')

    parser.add_argument('--lam', type=float, default=0.5,
                        help="Loss scaling")
    parser.add_argument("--correct", default=500, type=int,
                        help="correction steps")

    parser.add_argument('--alpha', default=True, type=bool,
                        help="Use alpha weight")

    return parser

import CXR_dataset

def train_dino(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    transform = DataAugmentationDINO(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
    )
    dataset = CXR_dataset.CXR_Dataset(args.data_path, transforms=transform, mode='train', labeled=False)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    l_dataset = CXR_dataset.CXR_Dataset(args.data_path, transforms=transform, mode='train', labeled=True)
    l_sampler = torch.utils.data.DistributedSampler(l_dataset, shuffle = True)
    l_data_loader = torch.utils.data.DataLoader(
        l_dataset,
        sampler=l_sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    print(f"Unlabeled Data loaded: there are {len(dataset)} images.")
    print(f"Labeled Data loaded: there are {len(l_dataset)} images.")

    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vit_o.__dict__.keys():
        student = vit_o.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,  # stochastic depth
        )
        teacher = vit_o.__dict__[args.arch](patch_size=args.patch_size)
        embed_dim = student.embed_dim
        inter_dim = 384

    elif args.arch == 'deit':
        student = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_384', pretrained=True,
                                 drop_path_rate=args.drop_path_rate)
        teacher = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_384', pretrained=True)
        embed_dim = student.embed_dim

    # if the network is a XCiT
    elif args.arch in torch.hub.list("facebookresearch/xcit"):
        student = torch.hub.load('facebookresearch/xcit', args.arch,
                                 pretrained=False, drop_path_rate=args.drop_path_rate)
        teacher = torch.hub.load('facebookresearch/xcit', args.arch, pretrained=False)
        embed_dim = student.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()

        if 'resne' in args.arch:
            embed_dim = student.fc.weight.shape[1]
            inter_dim = 2048
        elif 'densenet' in args.arch:
            embed_dim = student.classifier.weight.shape[1]
            inter_dim = 1920
        elif 'eff' in args.arch:
            embed_dim = student.classifier[1].weight.shape[1]
            inter_dim = 1792
    else:
        print(f"Unknow architecture: {args.arch}")

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(student, DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ), CLS_head(inter_dim, 256, 1), args)
    teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head), CLS_head(inter_dim, 256, 1), args)

    # Load weights
    state_dict = torch.load(args.pretrained_dir, map_location="cpu")
    print("Take key {} in provided checkpoint dict".format(args.checkpoint_key))
    if args.checkpoint_key == 'each':
        if args.total_folds == 1:
            state_dict_t = state_dict['student']
            state_dict_s = state_dict['student']
        else:
            state_dict_t = state_dict['teacher']
            state_dict_s = state_dict['student']
    else:
        if args.total_folds == 1:
            state_dict_t = state_dict['student']
            state_dict_s = state_dict['student']
        else:
            state_dict_t = state_dict[args.checkpoint_key]
            state_dict_s = state_dict[args.checkpoint_key]
    # remove `module.` prefix
    state_dict_t = {k.replace("module.", ""): v for k, v in state_dict_t.items()}
    state_dict_s = {k.replace("module.", ""): v for k, v in state_dict_s.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    # state_dict = {k.replace(".backbone.", "___"): v for k, v in state_dict.items()}
    # state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    # state_dict = {k.replace("___", ".backbone."): v for k, v in state_dict.items()}
    msg_t = teacher.load_state_dict(state_dict_t, strict=False)
    print('(Teacher) Pretrained weights found at {} and loaded with msg: {}'.format('Pre-training', msg_t))
    msg_s = student.load_state_dict(state_dict_s, strict=False)
    print('(Student) Pretrained weights found at {} and loaded with msg: {}'.format('Pre-training', msg_s))

    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu], find_unused_parameters=True)
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu], find_unused_parameters=True)
    # # teacher and student start with the same weights
    # teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()
    bce_loss = nn.BCEWithLogitsLoss()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 16.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready for unlabeled batch.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting DISTL training !")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... ============
        print("============ Training with pseudolabel ... ============")
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, bce_loss,
            data_loader, l_data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, args.ssl_epoch, fp16_scaler, args)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, bce_loss, data_loader, l_data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule, epoch, ssl_epoch,
                    fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, (images, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output, teacher_cls = teacher(images[:2])  # only the 2 global views pass through the teacher
            student_output, student_cls = student(images)
            loss_dino = dino_loss(student_output, teacher_output, epoch)
            loss_ce = bce_loss(student_cls[args.batch_size_per_gpu:].view(-1), torch.sigmoid(teacher_cls[:args.batch_size_per_gpu].view(-1)))

            if args.alpha:
                loss_ce = utils.alpha_weight(epoch, args.epochs) * loss_ce
            else:
                loss_ce = loss_ce

            # loss_ce_2 = bce_loss(student_cls[:args.batch_size_per_gpu].view(-1), torch.sigmoid(teacher_cls[:args.batch_size_per_gpu].view(-1)))
            # loss_ce = (loss_ce_1 + loss_ce_2) / 2.

            if epoch < ssl_epoch:
                loss = args.lam * loss_dino + (1 - args.lam) * loss_ce
            else:
                loss = loss_ce

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)


        if it % args.correct == 0:
            print("============ Correction with labeled data ... ============")
            # ====================== Correction with labeled data ===============================
            for _, (images, labels) in enumerate(metric_logger.log_every(l_data_loader, 10, header)):
                # move images to gpu
                images = [im.cuda(non_blocking=True) for im in images]
                labels = labels.float().cuda()

                # teacher and student forward passes + compute dino loss
                with torch.cuda.amp.autocast(fp16_scaler is not None):
                    teacher_output, teacher_cls = teacher(images[:2])
                    student_output, student_cls = student(images)

                    l_loss_dino = dino_loss(student_output, teacher_output, epoch)
                    l_loss_ce = bce_loss(student_cls[args.batch_size_per_gpu:].view(-1), labels.view(-1))

                    if epoch < ssl_epoch:
                        loss = args.lam * l_loss_dino + (1 - args.lam) * l_loss_ce
                    else:
                        loss = l_loss_ce

                if not math.isfinite(loss.item()):
                    print("Loss is {}, stopping training".format(loss.item()), force=True)
                    sys.exit(1)

                # student update
                optimizer.zero_grad()
                param_norms = None
                if fp16_scaler is None:
                    loss.backward()
                    if args.clip_grad:
                        param_norms = utils.clip_gradients(student, args.clip_grad)
                    utils.cancel_gradients_last_layer(epoch, student,
                                                      args.freeze_last_layer)
                    optimizer.step()
                else:
                    fp16_scaler.scale(loss).backward()
                    if args.clip_grad:
                        fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                        param_norms = utils.clip_gradients(student, args.clip_grad)
                    utils.cancel_gradients_last_layer(epoch, student,
                                                      args.freeze_last_layer)
                    fp16_scaler.step(optimizer)
                    fp16_scaler.update()

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5)]
        )

        normalize = transforms.Compose([
            transforms.ToTensor(),
        ])

        # first global crop (Very little augmentation)
        self.global_transfo1 = transforms.Compose([
            normalize,
        ])

        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(256, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            transforms.RandomRotation(degrees=(-15, 15)),
            transforms.RandomAutocontrast(p=0.3),
            transforms.RandomEqualize(p=0.3),
            utils.GaussianBlur(0.3),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(128, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            transforms.RandomRotation(degrees=(-15, 15)),
            transforms.RandomAutocontrast(p=0.5),
            transforms.RandomEqualize(p=0.5),
            utils.GaussianBlur(0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    args.option_dir = './'
    os.makedirs(args.option_dir, exist_ok=True)
    with open(os.path.join(args.option_dir, args.name + '_argv.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(vars(args).items())
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    train_dino(args)
