import os
import math
import argparse
import time
import datetime
from datetime import timedelta
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from utils.util_train import save_model, get_logger, AverageMeter
from utils.util_test import calc_similarity, Recall_metrics, IoU_metrics, Composite_metrics
from modules.metrics import sim_matrix_music_pooling, sim_matrix_video_pooling
from utils.scheduler import *
from dataloaders.data_dataloaders import DATALOADER_DICT
from model.model_Uni import Uni_model
from modules.loss import *
import torch.nn.functional as F
from music_detr.span_utils import span_cw_to_se, detr_iou
import json

torch.distributed.init_process_group(backend="nccl", timeout=timedelta(seconds=7200))

def parse_option():
    parser = argparse.ArgumentParser('train-Uni', add_help=False)
    # base
    parser.add_argument('--name', required=True, type=str, help='create model name')
    parser.add_argument('--output_dir', default='./logs', type=str)
    parser.add_argument('--load_uni_model_path', type=str, default="")
    parser.add_argument('--resume_path', type=str, default=None)
    # data
    parser.add_argument('--data', type=str, default='kuai50k')
    parser.add_argument('--train_data', type=str, default='kuai50k')
    parser.add_argument('--val_data', type=str, default='kuai50k')
    parser.add_argument('--test_data', type=str, default='kuai50k')
    parser.add_argument('--train_csv', type=str, default="dataset/MGSV-EC/train_data.csv")
    parser.add_argument('--val_csv', type=str, default="dataset/MGSV-EC/val_data.csv")
    parser.add_argument('--test_csv', type=str, default="dataset/MGSV-EC/test_data.csv")
    parser.add_argument('--image_resolution', type=int, default=224)
    parser.add_argument('--max_v_frames', type=int, default=30)
    parser.add_argument('--max_m_duration', type=int, default=240)
    parser.add_argument('--stride', type=float, default=2.5)
    parser.add_argument('--filter', type=float, default=4)
    parser.add_argument('--padding', type=int, default=0)
    parser.add_argument('--toph_moment', type=int, default=1)
    parser.add_argument('--gt_moment_num', type=int, default=1)
    # model
    parser.add_argument('--backbone_type', type=str, default="transf+detr", help="backbone type", choices=["baseline", "transf+detr"])
    parser.add_argument('--dim_input', type=int, default=256)
    parser.add_argument('--frozen_feature_path', type=str, default="features/Kuai_feature")
    parser.add_argument('--video_encoder_type', type=str, default="ViT", help="video encoder type", 
                        choices=["ViT", "ViViT"])
    parser.add_argument('--audio_encoder_type', type=str, default="AST", help="audio encoder type",
                        choices=["MERT", "AST", "DeepSim"])
    parser.add_argument('--temperature_init_value', type=float, default=0.07)
    # Temporal
    parser.add_argument('--video_attention_seqlen', type=int, default=250)
    parser.add_argument('--video_transformer_depth', type=int, default=1)
    parser.add_argument('--audio_transformer_depth', type=int, default=1)
    parser.add_argument('--with_cls_token', type=int, default=0)
    parser.add_argument('--with_last_token', type=int, default=0)
    parser.add_argument('--with_act_after_proj', type=int, default=0)
    parser.add_argument('--transformer_is_share', type=int, default=0)
    parser.add_argument('--projection_is_share', type=int, default=0)
    parser.add_argument('--SA_temporal_heads', type=int, default=8)
    parser.add_argument('--agg_module', type=str, default="transf", help="agg module type", choices=["None", "transf", "mlp"])
    parser.add_argument('--downup_is_share', type=int, default=0)
    parser.add_argument('--downup_dim', type=int, default=64)
    # VMR Pooling
    parser.add_argument('--vmr_fusion', type=str, default="XA-music", help="fusion type, XA is cross-attn, XA_video_music is cross-attn with video and music",
                        choices=["NO", "XA", "XA-video", "XA-music", "XA-video-music", "XA-music-video"])  # last two options are same
    parser.add_argument('--vmr_loss', type=str, default="dual_single_loss_fuse", help="dual / single tower loss", 
                        choices=["dual", "single", "dual_single", "dual_single_oneloss", "dual_single_sim_fuse", "dual_single_loss_fuse", "dual_single_feature_fuse"])
    parser.add_argument('--dual_single_loss_weight', type=float, default=1.0)
    parser.add_argument('--fusion_mask', type=int, default=1)
    # MML Fusion
    parser.add_argument('--mml_fusion', type=str, default="CA", choices=["CA", "concat", "add"])
    # Music-DETR
    parser.add_argument('--mml_localization', type=str, default="detr", help="mml localization type", choices=["detr", "regression"])
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--moment_query_type', type=str, default="video", choices=["video", "xpool", "music", "random", "zero"])
    parser.add_argument('--span_loss_type', type=str, default="l1", help="span loss type", choices=["l1", "ce"])
    parser.add_argument('--fb_label', type=str, default="01", choices=["01", "10"])
    # detr
    parser.add_argument('--detr_hidden_dim', type=int, default=256)
    parser.add_argument('--detr_dropout', type=float, default=0.1)
    parser.add_argument('--detr_nheads', type=int, default=8)
    parser.add_argument('--detr_dim_feedforward', type=int, default=1024)
    parser.add_argument('--detr_enc_layers', type=int, default=0)
    parser.add_argument('--detr_dec_layers', type=int, default=6)
    parser.add_argument('--detr_pre_norm', type=bool, default=False)
    parser.add_argument('--num_moment_queries', type=int, default=1)
    parser.add_argument('--decoder_SA', type=int, default=0)
    parser.add_argument('--predict_center', type=int, default=0)
    parser.add_argument('--reg_mlp_num_layers', type=int, default=3)
    # loss
    parser.add_argument('--l1_loss', type=int, default=1)
    parser.add_argument('--aux_loss', type=int, default=1)
    parser.add_argument('--contrastive_align_loss', type=int, default=1)
    parser.add_argument('--moment_loss', type=int, default=0)
    parser.add_argument('--audio_short_cut', type=int, default=1)
    parser.add_argument('--contrastive_dim', type=int, default=256)
    parser.add_argument('--position_embedding', default='sine', type=str, choices=['sine', 'learned'], help='detr pos embedding for music_detr/position_encoding.py')
    parser.add_argument('--input_dropout', type=float, default=0.5)
    parser.add_argument('--ret_loss_weight', type=float, default=3.0)
    parser.add_argument('--loc_loss_weight', type=float, default=0.2)

    # train
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size_train', type=int, default=512)
    parser.add_argument('--batch_size_val', type=int, default=128)
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--ignore_same_music', type=int, default=1, help="ignore same music in VMR dual loss, aka. more negative pairs")
    # distributed training
    parser.add_argument("--world_size", default=0, type=int, help="distribted training")
    parser.add_argument("--rank", default=0, type=int, help="distribted training")
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
    # optimization
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--matching_lr", default=1e-4, type=float, help="learning rate for Video-to-Music Matching")
    parser.add_argument("--detection_lr", default=1e-4, type=float, help="learning rate for Music Moment Detection")
    parser.add_argument('--decay_rate', default=0.9, type=float)
    parser.add_argument('--max_grad_norm', default=1.0, type=float)
    parser.add_argument('--scheduler', type=str, default='warmupcosine', help='scheduler type', 
                        choices=["warmupcosine", "warmuplinear", "warmupconstant", "constant", "exponential"])
    parser.add_argument('--lr_update_rate', type=int, default=50)
    parser.add_argument('--warmup_rate', type=float, default=0.1)
    parser.add_argument('--distance_type', type=str, default="COS")
    # display
    parser.add_argument('--num_display', type=int, default=15, help='number of steps to display in one epoch')
    parser.add_argument('--tb_writer', type=int, default=1, help='whether to use tensorboard')
    parser.add_argument('--save_model', type=int, default=1, help='whether to save model')
    parser.add_argument('--save_json', type=int, default=0, help='whether to save json')
    args = parser.parse_args()
    # data
    args.train_data = args.train_data + "_uni"
    args.val_data = args.val_data + "_uni"
    args.max_snippet_num = int(args.max_m_duration / args.stride)
    # model
    if "transf" not in args.agg_module:
        args.video_transformer_depth = 0
        args.audio_transformer_depth = 0
    assert (args.moment_loss >= args.audio_short_cut) or (args.contrastive_align_loss >= args.audio_short_cut), "moment loss must be 1 when audio_short_cut is 1"
    # dim
    args.hidden_dim = args.dim_input
    args.detr_hidden_dim = args.dim_input
    # fusion
    if "XA" in args.vmr_fusion and "single" not in args.vmr_loss:
        raise ValueError("XA fusion must support single tower loss in VMR")
    # detr
    if args.decoder_SA == 0 and args.num_moment_queries > 1:
        raise ValueError("decoder_SA must be 1 when num_moment_queries > 1")
    # feature path
    music_feature_dir_list = {
        2.5: "ast_feature2p5",
        5.0: "ast_feature5",
        7.5: "ast_feature7p5",
        10.0: "ast_feature10",
    }
    args.music_frozen_feature_path = os.path.join(args.frozen_feature_path, music_feature_dir_list[args.stride])
    args.frame_frozen_feature_path = os.path.join(args.frozen_feature_path, "vit_feature1")
    if args.local_rank == 0:
        print(f"music frozen feature path: {args.music_frozen_feature_path}, stride: {args.stride}")
        print(f"frame frozen feature path: {args.frame_frozen_feature_path}")
    return args

def set_seed_logger(args):
    global logger
    # set random seed
    random.seed(args.seed)  # for python
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)  # for numpy
    torch.manual_seed(args.seed)  # for pytorch on CPU
    torch.cuda.manual_seed(args.seed)  # for pytorch on GPU
    torch.cuda.manual_seed_all(args.seed)  # for pytorch on multi-GPUs
    torch.backends.cudnn.benchmark = False  # Disable CuDNN auto-tuning for reproducible results
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior in CuDNN for the same data and network structure
    # set distributed training
    args.world_size = torch.distributed.get_world_size()  # Total number of processes
    args.rank = torch.distributed.get_rank()  # Total number of processes
    # get logger
    creat_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())  # Get the training creation time
    creat_date = time.strftime("%m%d", time.localtime())  # Get the training creation date
    args.path_log = os.path.join(args.output_dir, f'{args.train_data}', f'{creat_date}+{args.name}')  # make sure Train log path
    os.makedirs(args.path_log, exist_ok=True)
    logger = get_logger(os.path.join(args.path_log, '%s-%s-%s_train.log' % (creat_time, args.name, args.train_data)))
    return args

def init_device(args):
    global logger
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", args.local_rank)
    args.gpu_num = torch.cuda.device_count()
    args.device_name = torch.cuda.get_device_name(0)
    logger.info("device {}".format(device))
    # check batch_size of train/val for multi-gpu
    if args.batch_size_train % args.gpu_num != 0 or args.batch_size_val % args.gpu_num != 0:
        raise ValueError("Invalid batch_size_train/batch_size_val and gpu_num parameter: {}%{} and {}%{}, should be == 0".format(
            args.batch_size_train, args.gpu_num, args.batch_size_val, args.gpu_num))
    return device, args

def count_parameters(model):
    global logger, tb_writer
    """Count number of parameters in PyTorch model,
    References: https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7.
    """
    n_all = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # transform to (M)
    n_all_m = n_all / 1e6
    n_trainable_m = n_trainable / 1e6
    logger.info(f"Parameter Count: all {n_all_m:.3f}M; trainable {n_trainable_m:.3f}M")

    # calculate parameters of each submodule
    vit_model_n_all = sum(p.numel() for p in model.vit_model.parameters())
    vit_model_n_trainable = sum(p.numel() for p in model.vit_model.parameters() if p.requires_grad)
    ast_model_n_all = sum(p.numel() for p in model.ast_model.parameters())
    ast_model_n_trainable = sum(p.numel() for p in model.ast_model.parameters() if p.requires_grad)
    logger.info(f"Vit Model: all {vit_model_n_all / 1e6:.3f}M; trainable {vit_model_n_trainable / 1e6:.3f}M")  # 151.28M, 0.0M
    logger.info(f"Ast Model: all {ast_model_n_all / 1e6:.3f}M; trainable {ast_model_n_trainable / 1e6:.3f}M")  # 88.132M, 0.0M
    if hasattr(model, 'detr_transformer'):
        detr_transformer_n_trainable = sum(p.numel() for p in model.detr_transformer.parameters() if p.requires_grad)
        detr_transformer_n_all = sum(p.numel() for p in model.detr_transformer.parameters())
        logger.info(f"DETR Transformer: all {detr_transformer_n_all / 1e6:.3f}M; trainable {detr_transformer_n_trainable / 1e6:.3f}M")
    return n_all_m, n_trainable_m

def init_model(args, device):
    global logger, tb_writer
    model = Uni_model(args, device, logger).to(device)
    model = model.float()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
    
    # freeze vit & ast
    model = model.module if hasattr(model, 'module') else model
    for name, param in model.named_parameters():
        if "mert_model" in name or "ast_model" in name or "vit_model" in name or "vivit_model" in name or "cnclip_model" in name:
            param.requires_grad = False
    # get SummaryWriter
    if args.tb_writer and args.local_rank == 0:
        tb_writer = SummaryWriter(log_dir=args.path_log)
    # count parameters
    count_parameters(model)
    return model

def prep_optimizer(args, model, warmup_steps, total_step):
    global logger
    model = model.module if hasattr(model, 'module') else model
    for name, param in model.named_parameters():
        if "mert_model" in name or "ast_model" in name or "vit_model" in name or "vivit_model" in name or "cnclip_model" in name:
            param.requires_grad = False

    if args.local_rank == 0:
        logger.info('Defining optimizer and scheduler')
    # optimizer
    optimizer_parameters = []
    optimizer_parameters.append({"params": model.get_temporal_parameter(), "lr": args.matching_lr})
    optimizer_parameters.append({"params": model.get_matching_parameter(), "lr": args.matching_lr})
    optimizer_parameters.append({"params": model.get_detection_parameter(), "lr": args.detection_lr})
    optimizer = optim.Adam(optimizer_parameters)
    # scheduler
    if args.local_rank == 0:
        logger.info('Using {} LR Scheduler'.format(args.scheduler))
        if args.scheduler == "exponential":
            logger.info(f"decay_rate = {args.decay_rate}")
        elif "warmup" in args.scheduler:
            logger.info(f"total_step = {total_step}, warmup_steps = {warmup_steps}")
    if args.scheduler != "exponential":
        args.lr_update_rate = 1
    if args.scheduler == "warmupcosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=warmup_steps, t_total=total_step)
    elif args.scheduler == "warmuplinear":
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=total_step)
    elif args.scheduler == "warmupconstant":
        scheduler = WarmupConstantSchedule(optimizer, warmup_steps=warmup_steps, t_total=total_step)
    elif args.scheduler == "constant":
        scheduler = ConstantLRSchedule(optimizer)
    elif args.scheduler == "exponential":
        scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=args.decay_rate)
    return optimizer, scheduler

def show_model_architecture(args, model):
    model = model.module if hasattr(model, 'module') else model
    print_params = [
        "detr_transformer.encoder.layers.0.self_attn",
        "detr_transformer.decoder.layers.0.self_attn",
        "ast_model.module.v.blocks.0.attn",
        "vit_model.visual.transformer.resblocks.3.attn",
        "vit_proj",
        "share_transformer.layers.0.0",
        "video_audio_fusion_cross_transformer.layers.0.0",
        "span_embed.layers.0",
    ]
    for name, param in model.named_parameters():
        if args.local_rank == 0:
            for nn in print_params:
                if nn in name:
                    logger.info(f"{name}, param.requires_grad={param.requires_grad}, param={param[0]}")



def train_one_epoch(epoch, args, model, train_loader, optimizer, scheduler, device, is_train=True):
    global logger, tb_writer
    torch.cuda.empty_cache()
    if args.local_rank == 0:
        logger.info('------Train epoch %d------', epoch)
    model.train()
    model = model.module if hasattr(model, 'module') else model

    num_steps = len(train_loader)
    num_log_steps = math.ceil(num_steps / args.num_display)
    batch_time, loss_meter, ret_loss_meter, loc_loss_meter = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    span_loss_meter, label_loss_meter, giou_loss_meter, class_error_meter, contrastive_loss_meter = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    mr_results_list = []
    start = time.time()
    for step, batch in enumerate(train_loader):
        end = time.time()
        data_map, meta_map, spans_target = batch
        frame_feats = data_map["frame_feats"].to(device)  # [bs, max_v_frames, 256]
        frame_masks = data_map["frame_mask"].to(device)  # [bs, max_v_frames]
        segment_feats = data_map["segment_feats"].to(device)  # [bs, max_snippet_num, 256]
        segment_masks = data_map["segment_mask"].to(device)  # [bs, max_snippet_num]
        video_ids = meta_map["video_id"]
        music_ids = meta_map["music_id"]
        gt_moment = meta_map["gt_moment"].to(device)  # [bs, 3, 2]
        m_duration = meta_map["m_duration"].to(device)  # [bs]
        v_duration = meta_map["v_duration"].to(device)  # [bs]
        spans_target = spans_target.to(device)  # [bs, 1, 2]

        # Forward
        output_map, loss_map, feat_map, mask_map, id_map = model(frame_feats, segment_feats, frame_masks, segment_masks, spans_target, v_duration=v_duration, video_ids=video_ids, music_ids=music_ids, is_train=True)
        retrieval_loss = loss_map["retrieval_loss"] * args.ret_loss_weight
        loss_dict = loss_map["localization_loss_dict"]
        localization_loss = loss_map["localization_loss"] * args.loc_loss_weight
        # output
        if "detr" in args.mml_localization:
            out_prob = F.softmax(output_map["pred_logits"], dim=-1)  # [bs, #queries, #classes=2]
            scores = out_prob[:, :, model.criterion.foreground_label]  # [bs, #queries]  foreground is 0
            pred_spans = output_map["pred_spans"]  # [bs, #queries, 2]
            for idx, (spans, score) in enumerate(zip(pred_spans.cpu(), scores.cpu())):
                if args.span_loss_type == "l1":
                    spans = span_cw_to_se(spans) * args.max_m_duration  # [#queries, 2]
                ranked_preds = torch.cat((spans, score.unsqueeze(-1)), dim=-1)  # [3 (st, ed, score),]
                ranked_preds = sorted(ranked_preds, key=lambda x: x[2], reverse=True)  # True
                ranked_preds = ranked_preds[:args.toph_moment]
                pred_dict = dict(
                    gt_moment = gt_moment[idx].detach().cpu(),
                    m_duration = m_duration[idx].detach().cpu(),
                    ranked_preds = ranked_preds
                )
                mr_results_list.append(pred_dict)
        elif "regression" in args.mml_localization:
            pred_spans = output_map["pred_spans"]  # [bs, 1, 2]
            for idx, spans in enumerate(pred_spans.cpu()):
                spans = span_cw_to_se(spans) * args.max_m_duration  # [1, 2]
                pred_dict = dict(
                    gt_moment = gt_moment[idx].detach().cpu(),
                    m_duration = m_duration[idx].detach().cpu(),
                    ranked_preds = spans  # [2,]
                )
                mr_results_list.append(pred_dict)

        # Backward
        loss = retrieval_loss +  localization_loss
        loss = reduce_tensor(loss, args.world_size)
        loss.backward()

        # update params
        nn.utils.clip_grad_norm_(model.get_temporal_parameter(), args.max_grad_norm)
        nn.utils.clip_grad_norm_(model.get_matching_parameter(), args.max_grad_norm)
        nn.utils.clip_grad_norm_(model.get_detection_parameter(), args.max_grad_norm)
        optimizer.step()
        if (args.total_step % args.lr_update_rate) == 0:
            scheduler.step()
        optimizer.zero_grad()

        # loss init
        span_loss = loss_dict["loss_span"] if "loss_span" in loss_dict else 0
        args.total_step += 1
        # AverageMeter record
        batch = frame_feats.size(0)
        batch_time.update(time.time() - end)
        ret_loss_meter.update(retrieval_loss, batch)
        loc_loss_meter.update(localization_loss, batch)
        loss_meter.update(loss, batch)
        span_loss_meter.update(span_loss, batch)
        label_loss_meter.update(loss_dict["loss_label"], batch)
        giou_loss_meter.update(loss_dict["loss_giou"], batch)
        class_error_meter.update(loss_dict["class_error"], batch)
        if "loss_contrastive_align" in loss_dict:
            contrastive_loss_meter.update(loss_dict["loss_contrastive_align"], batch)
        lr = scheduler.get_last_lr()[0]
        # Tensorboard record
        if args.tb_writer and args.local_rank == 0:
            tb_writer.add_scalar(f'train/loss', loss, (epoch-1) * num_steps + step)
            tb_writer.add_scalar(f'train/ret_loss', retrieval_loss, (epoch-1) * num_steps + step)
            tb_writer.add_scalar(f'train/loc_loss', localization_loss, (epoch-1) * num_steps + step)
            tb_writer.add_scalar(f'train/lr', lr, (epoch-1) * num_steps + step)
            tb_writer.add_scalar(f'train/loss_span', span_loss, (epoch-1) * num_steps + step)
            tb_writer.add_scalar(f'train/loss_label', loss_dict["loss_label"], (epoch-1) * num_steps + step)
            tb_writer.add_scalar(f'train/loss_giou', loss_dict["loss_giou"], (epoch-1) * num_steps + step)
            tb_writer.add_scalar(f'train/class_error', loss_dict["class_error"], (epoch-1) * num_steps + step)
            if "loss_contrastive_align" in loss_dict:
                tb_writer.add_scalar(f'train/loss_contrastive_align', loss_dict["loss_contrastive_align"], (epoch-1) * num_steps + step)
        # Logger record
        if step % num_log_steps == 0 and args.local_rank == 0:
            logger.info(
                f'Train [{epoch}/{args.epochs}, {step+1}/{num_steps}]\t'
                f'loss: {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'ret_loss: {ret_loss_meter.val:.4f} ({ret_loss_meter.avg:.4f})\t'
                f'loc_loss: {loc_loss_meter.val:.4f} ({loc_loss_meter.avg:.4f})\t'
                f'span {span_loss:.4f} label {loss_dict["loss_label"]:.4f} giou {loss_dict["loss_giou"]:.4f} class_error {loss_dict["class_error"]:.2f}({class_error_meter.avg:.2f})    '
                f'time: {batch_time.val:.1f}s ({batch_time.avg:.1f})\t'
                f'lr: {lr:.6f}')
    if args.local_rank == 0:
        logger.info(f"Epoch {epoch}/{args.epochs} Finished, Train Loss: {loss_meter.avg:.4f}")
    
    # get IoU matrix
    IoU_list = detr_iou(args, mr_results_list)
    loc_metrics = IoU_metrics(IoU_list)
    if args.local_rank == 0:
        logger.info(f"Music Moment Localization Train >>> mIoU: {loc_metrics['mIoU']:.4f} - IoU@0.5: {loc_metrics['IoU@0.5']:.2f} - IoU@0.7: {loc_metrics['IoU@0.7']:.2f}")
    
    # Sum time record
    epoch_time = time.time() - start
    if args.local_rank == 0:
        logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    return loss_meter.avg, loc_metrics




@torch.no_grad()
def eval_epoch(epoch, args, model, val_loader, device):
    if args.do_eval == False:
        return 0, {"mIoU": 0, "IoU@0.3": 0, "IoU@0.5": 0, "IoU@0.7": 0}
    global logger, tb_writer
    if args.local_rank == 0:
        logger.info('------Eval epoch %d------', epoch)
    model.eval()
    model = model.module if hasattr(model, 'module') else model

    num_steps = len(val_loader)
    num_log_steps = math.ceil(num_steps / args.num_display)
    batch_time, loss_meter = AverageMeter(), AverageMeter()
    span_loss_meter, label_loss_meter, giou_loss_meter, class_error_meter, contrastive_loss_meter = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    ret_loss_meter, loc_loss_meter = AverageMeter(), AverageMeter()
    mr_results_list, loc_results_list = [], []
    start = time.time()
    video_feat_list, audio_feat_list = [], []
    all_video_ids_list, all_music_ids_list = [], []
    all_video_embed_arr, all_segment_embed_arr = [], []
    all_music_embed_arr, all_frame_embed_arr = [], []
    all_frame_mask_arr, all_segment_mask_arr = [], []
    for step, batch in enumerate(val_loader):
        end = time.time()
        data_map, meta_map, spans_target = batch
        frame_feats = data_map["frame_feats"].to(device)  # [bs, max_v_frames, 256]
        frame_masks = data_map["frame_mask"].to(device)  # [bs, max_v_frames]
        segment_feats = data_map["segment_feats"].to(device)  # [bs, max_snippet_num, 256]
        segment_masks = data_map["segment_mask"].to(device)  # [bs, max_snippet_num]
        video_ids = meta_map['video_id']
        music_ids = meta_map['music_id']
        gt_moment = meta_map["gt_moment"].to(device)  # [bs, 1, 2]
        m_duration = meta_map["m_duration"].to(device)  # [bs]
        v_duration = meta_map["v_duration"].to(device)  # [bs]
        spans_target = spans_target.to(device)  # [bs, gt_moment_num, 2]
        
        # Forward
        output_map, loss_map, feat_map, mask_map, id_map = model(frame_feats, segment_feats, frame_masks, segment_masks, spans_target, v_duration=v_duration, video_ids=video_ids, music_ids=music_ids, is_train=False)
        retrieval_loss = loss_map["retrieval_loss"] * args.ret_loss_weight
        loss_dict = loss_map["localization_loss_dict"]
        localization_loss = loss_map["localization_loss"] * args.loc_loss_weight
        video_feats = feat_map["video_feats"]
        audio_feats = feat_map["music_feats"]
        frame_feats = feat_map["frame_feats"]
        segment_feats = feat_map["segment_feats"]
        frame_masks = mask_map["frame_masks"]
        segment_masks = mask_map["segment_masks"]
        video_ids = id_map["video_ids"]
        music_ids = id_map["music_ids"]
        # List for similarity matrix & Recall
        video_feat_list.append(video_feats.cpu().detach().numpy())  # [bs, 256]
        audio_feat_list.append(audio_feats.cpu().detach().numpy())  # [bs, 256]
        all_music_ids_list.extend(music_ids)
        all_video_ids_list.extend(video_ids)
        # For vmr_fusion == "XA"
        all_video_embed_arr.append(video_feats.cpu())  # From https://github.com/layer6ai-labs/xpool/blob/main/trainer/trainer.py#L127
        all_segment_embed_arr.append(segment_feats.cpu())
        all_music_embed_arr.append(audio_feats.cpu())
        all_frame_embed_arr.append(frame_feats.cpu())
        all_frame_mask_arr.append(frame_masks.cpu())
        all_segment_mask_arr.append(segment_masks.cpu())
        # output
        if "detr" in args.mml_localization:
            out_prob = F.softmax(output_map["pred_logits"], dim=-1)  # [bs, #queries, #classes]
            scores = out_prob[:, :, model.criterion.foreground_label]  # [bs, #queries]  foreground is 0
            pred_spans = output_map["pred_spans"]  # [bs, #queries, 2]
            for idx, (spans, score) in enumerate(zip(pred_spans.cpu(), scores.cpu())):
                if args.span_loss_type == "l1":
                    spans = span_cw_to_se(spans) * args.max_m_duration
                ranked_preds = torch.cat((spans, score.unsqueeze(-1)), dim=-1)  # [3 (st, ed, score),]
                # 按照score降序排列
                ranked_preds = sorted(ranked_preds, key=lambda x: x[2], reverse=True)  # list of #queries list, each list is [st, ed, score]
                ranked_preds = ranked_preds[:args.toph_moment] 
                pred_dict = dict(
                    gt_moment = gt_moment[idx].detach().cpu(),
                    m_duration = m_duration[idx].detach().cpu(),
                    ranked_preds = ranked_preds
                )
                pred_dict_np = dict(
                    gt_moment = gt_moment[idx].detach().cpu().tolist(),
                    video_id = video_ids[idx],
                    music_id = music_ids[idx],
                    m_duration = round(m_duration[idx].detach().cpu().tolist(), 3),
                    pred_st = round(ranked_preds[0][0].detach().cpu().tolist(), 3),
                    pred_ed = round(ranked_preds[0][1].detach().cpu().tolist(), 3),
                )
                mr_results_list.append(pred_dict)
                loc_results_list.append(pred_dict_np)
        elif "regression" in args.mml_localization:
            pred_spans = output_map["pred_spans"]  # [bs, 1, 2]
            for idx, spans in enumerate(pred_spans.cpu()):
                spans = span_cw_to_se(spans) * args.max_m_duration  # [1, 2]
                pred_dict = dict(
                    gt_moment = gt_moment[idx].detach().cpu(),
                    m_duration = m_duration[idx].detach().cpu(),
                    ranked_preds = spans  # [2,]
                )
                mr_results_list.append(pred_dict)

        # loss init
        loss = retrieval_loss +  localization_loss
        span_loss = loss_dict["loss_span"] if "loss_span" in loss_dict else 0
        # AverageMeter record
        batch = frame_feats.size(0)
        batch_time.update(time.time() - end)
        ret_loss_meter.update(retrieval_loss, batch)
        loc_loss_meter.update(localization_loss, batch)
        loss_meter.update(loss, batch)
        span_loss_meter.update(span_loss, batch)
        label_loss_meter.update(loss_dict["loss_label"], batch)
        giou_loss_meter.update(loss_dict["loss_giou"], batch)
        class_error_meter.update(loss_dict["class_error"], batch)
        if "loss_contrastive_align" in loss_dict:
            contrastive_loss_meter.update(loss_dict["loss_contrastive_align"], batch)
        # Tensorboard record
        if args.tb_writer and args.local_rank == 0:
            tb_writer.add_scalar(f'eval/loss', loss, (epoch-1) * num_steps + step)
            tb_writer.add_scalar(f'train/ret_loss', retrieval_loss, (epoch-1) * num_steps + step)
            tb_writer.add_scalar(f'train/loc_loss', localization_loss, (epoch-1) * num_steps + step)
            tb_writer.add_scalar(f'eval/loss_span', span_loss, (epoch-1) * num_steps + step)
            tb_writer.add_scalar(f'eval/loss_label', loss_dict["loss_label"], (epoch-1) * num_steps + step)
            tb_writer.add_scalar(f'eval/loss_giou', loss_dict["loss_giou"], (epoch-1) * num_steps + step)
            tb_writer.add_scalar(f'eval/class_error', loss_dict["class_error"], (epoch-1) * num_steps + step)
            if "loss_contrastive_align" in loss_dict:
                tb_writer.add_scalar(f'eval/loss_contrastive_align', loss_dict["loss_contrastive_align"], (epoch-1) * num_steps + step)
        # Logger record
        if step % num_log_steps == 0 and args.local_rank == 0:
            logger.info(
                f'Eval [{epoch}/{args.epochs}, {step+1}/{num_steps}]\t'
                f'loss: {loss:.4f} ({loss_meter.avg:.4f})\t'
                f'ret_loss: {ret_loss_meter.val:.4f} ({ret_loss_meter.avg:.4f})\t'
                f'loc_loss_meter: {loc_loss_meter.val:.4f} ({loc_loss_meter.avg:.4f})\t'
                f'span {span_loss:.4f} label {loss_dict["loss_label"]:.4f} giou {loss_dict["loss_giou"]:.4f} class_error {loss_dict["class_error"]:.2f}({class_error_meter.avg:.2f})    '
                f'time: {batch_time.val:.1f}s ({batch_time.avg:.1f})')
    if args.local_rank == 0:
        logger.info(f"Epoch {epoch}/{args.epochs} Result, Eval Loss: {loss_meter.avg:.4f}")
    
    # get Similarity matrix
    if "XA" not in args.vmr_fusion:
        if args.vmr_loss == "dual":
            sim_matrix = calc_similarity(video_feat_list, audio_feat_list, distance_type=args.distance_type)  # [val_len, val_len]
    else: # "XA" in args.vmr_fusion
        # From https://github.com/layer6ai-labs/xpool/blob/main/trainer/trainer.py#L149
        video_embeds = torch.cat(all_video_embed_arr, dim=0)  # [val_len, 256]
        frame_embeds = torch.cat(all_frame_embed_arr, dim=0)  # [val_len, max_v_frames, 256]
        frame_masks = torch.cat(all_frame_mask_arr, dim=0)  # [val_len, max_v_frames]
        music_embeds = torch.cat(all_music_embed_arr, dim=0)  # [val_len, 256]
        segment_embeds = torch.cat(all_segment_embed_arr, dim=0)  # [val_len, max_snippet_num, 256]
        segment_masks = torch.cat(all_segment_mask_arr, dim=0)  # [val_len, max_snippet_num]
        model.video_guided_to_music_pooling_cross_transformer.cpu()
        # [val_len_m, val_len_v, dim]
        music_embeds_pooled = model.video_guided_to_music_pooling_cross_transformer(video_embeds, segment_embeds, segment_masks if args.fusion_mask==1 else None)
        model.video_guided_to_music_pooling_cross_transformer.to(device)
        if args.vmr_loss == "single":
            single_sim_matrix = sim_matrix_music_pooling(video_embeds, music_embeds_pooled)
            sim_matrix = single_sim_matrix.cpu().detach().numpy()
        elif (args.vmr_loss == "dual_single_sim_fuse" or args.vmr_loss == "dual_single_loss_fuse"):
            single_sim_matrix = sim_matrix_music_pooling(video_embeds, music_embeds_pooled)
            single_sim_matrix = single_sim_matrix.cpu().detach().numpy()
            dual_sim_matrix = calc_similarity(video_feat_list, audio_feat_list, distance_type=args.distance_type)  # [val_len, val_len]
            sim_matrix = single_sim_matrix * 1.0 + dual_sim_matrix * 1.0
        elif args.vmr_loss == "dual_single_feature_fuse":
            _, bs_v, _ = music_embeds_pooled.shape
            music_embeds_pooled_fused = music_embeds_pooled + music_embeds.unsqueeze(1).expand(-1, bs_v, -1)  # [val_len_m, val_len_v, dim]
            single_sim_matrix = sim_matrix_music_pooling(video_embeds, music_embeds_pooled_fused)
            sim_matrix = single_sim_matrix.cpu().detach().numpy()
        else:
            raise ValueError(f"Invalid vmr_loss: {args.vmr_loss}")
    # get Recall
    ret_metrics, ret_rank_list, _ = Recall_metrics(sim_matrix, dedup=True, all_music_ids_list=all_music_ids_list)

    # get IoU metrics
    IoU_list = detr_iou(args, mr_results_list)  # [val_len]
    loc_metrics = IoU_metrics(IoU_list)

    # get Composite metrics
    com_metrics = Composite_metrics(ret_rank_list, IoU_list, mr_results_list, all_video_ids_list, all_music_ids_list)

    if args.local_rank == 0:
        logger.info(
            f"Video-to-Music Retrieval  Eval "
            f">>> R@1: {ret_metrics['R1']:.2f} - R@5: {ret_metrics['R5']:.2f}"
            f" - R@10: {ret_metrics['R10']:.1f} - R@25: {ret_metrics['R25']:.1f}"
            f" - R@50: {ret_metrics['R50']:.1f} - R@100: {ret_metrics['R100']:.1f}"
            f" - Median R: {ret_metrics['MedianR']:.1f} - Mean R: {ret_metrics['MeanR']:.1f}"
            f" - MRR: {ret_metrics['MRR']:.4f}")
        logger.info(
            f"Music Moment Localization Eval "
            f">>> mIoU: {loc_metrics['mIoU']:.4f} - IoU0.5: {loc_metrics['IoU@0.5']:.2f} - IoU0.7: {loc_metrics['IoU@0.7']:.2f}")
        logger.info(f"Composite Eval ")
        logger.info(f">> IoU0.5 - R1: {com_metrics['R1_iou0.5']:.2f} - R10: {com_metrics['R10_iou0.5']:.2f} - R100: {com_metrics['R100_iou0.5']:.2f}")
        logger.info(f">> IoU0.7 - R1: {com_metrics['R1_iou0.7']:.2f} - R10: {com_metrics['R10_iou0.7']:.2f} - R100: {com_metrics['R100_iou0.7']:.2f}")

    # Sum time record
    epoch_time = time.time() - start
    torch.distributed.barrier()
    if args.local_rank == 0:
        logger.info(f"Eval takes {datetime.timedelta(seconds=int(epoch_time))}")
    return loss_meter.avg, ret_metrics, loc_metrics, com_metrics

def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= world_size
    return rt






def main():
    global logger, tb_writer
    args = parse_option()
    args = set_seed_logger(args)
    # get device
    device, args = init_device(args)

    # print args
    if args.local_rank == 0:
        for param in sorted(vars(args).keys()):
            logger.info('--{0} {1}'.format(param, vars(args)[param]))
        logger.info("\n")
    
    # get model
    model = init_model(args, device)
    # get datasets
    assert args.val_data in DATALOADER_DICT
    val_loader, val_length, val_sampler = DATALOADER_DICT[args.val_data]["val"](args)
    train_length = 0
    if args.do_train:
        assert args.train_data in DATALOADER_DICT
        train_loader, train_length, train_sampler = DATALOADER_DICT[args.train_data]["train"](args)
        num_train_optimization_steps = (int(len(train_loader) + args.gradient_accumulation_steps - 1) / args.gradient_accumulation_steps) * args.epochs
        # get optimizer
        total_step = len(train_loader) * args.epochs
        warmup_steps = int(total_step * args.warmup_rate)
        optimizer, scheduler = prep_optimizer(args, model, warmup_steps, total_step)
    if args.local_rank == 0:
        logger.info("***** Data Set *****")
        logger.info("train_length = %d, val_length = %d" % (train_length, val_length))
        logger.info("batch_size_train = %d, batch_size_val = %d" % (args.batch_size_train, args.batch_size_val))


    if args.do_train:
        args.start_epoch += 1
        args.total_step = 0
        best_val_r1 = {"R1": 0.0, "epoch": 0}
        best_val_r5 = {"R5": 0.0, "epoch": 0}
        best_val_miou = {"mIoU": 0.0, "epoch": 0}
        best_val_r1iou05 = {"R1_iou0.5": 0.0, "epoch": 0}
        best_val_r1iou07 = {"R1_iou0.7": 0.0, "epoch": 0}
        for epoch in range(args.start_epoch, args.epochs + 1):  # [1 or resume_epoch+1, args.epochs]
            train_sampler.set_epoch(epoch)
            # train epoch
            train_loss, train_loc_metrics = train_one_epoch(epoch, args, model, train_loader, optimizer, scheduler, device, is_train=True)
            # eval epoch
            val_loss, val_ret_metrics, val_loc_metrics, val_com_metrics = eval_epoch(epoch, args, model, val_loader, device)
            # record
            if args.tb_writer and args.local_rank == 0:
                tb_writer.add_scalar(f'train/loss_epoch', train_loss, epoch)
                tb_writer.add_scalar(f'train/mIoU_epoch', train_loc_metrics['mIoU'], epoch)
                if args.do_eval:
                    tb_writer.add_scalar(f'eval/loss_epoch', val_loss, epoch)
                    tb_writer.add_scalar(f'eval/R1_epoch', val_ret_metrics['R1'], epoch)
                    tb_writer.add_scalar(f'eval/R5_epoch', val_ret_metrics['R5'], epoch)
                    tb_writer.add_scalar(f'eval/MdR_epoch', val_ret_metrics['MedianR'], epoch)
                    tb_writer.add_scalar(f'eval/mIoU_epoch', val_loc_metrics['mIoU'], epoch)
            if args.local_rank == 0:
                # save_model(epoch, args, logger, model, optimizer=None, loss=val_loss)
                if args.do_eval and val_ret_metrics['R1'] >= best_val_r1["R1"]:
                    best_val_r1["R1"] = val_ret_metrics['R1']
                    best_val_r1["epoch"] = epoch
                    save_model(epoch, args, logger, model, optimizer=None, loss=val_loss, best_model=True, best_name="best_r1")
                if args.do_eval and val_ret_metrics['R5'] >= best_val_r5["R5"]:
                    best_val_r5["R5"] = val_ret_metrics['R5']
                    best_val_r5["epoch"] = epoch
                if args.do_eval and val_loc_metrics['mIoU'] >= best_val_miou["mIoU"]:
                    best_val_miou["mIoU"] = val_loc_metrics['mIoU']
                    best_val_miou["epoch"] = epoch
                    save_model(epoch, args, logger, model, optimizer=None, loss=val_loss, best_model=True, best_name="best_iou")
                if args.do_eval and val_com_metrics['R1_iou0.5'] > best_val_r1iou05["R1_iou0.5"]:
                    best_val_r1iou05["R1_iou0.5"] = val_com_metrics['R1_iou0.5']
                    best_val_r1iou05["epoch"] = epoch
                    save_model(epoch, args, logger, model, optimizer=None, loss=val_loss, best_model=True, best_name="best_r1iou05")
                if args.do_eval and val_com_metrics['R1_iou0.7'] >= best_val_r1iou07["R1_iou0.7"]:
                    best_val_r1iou07["R1_iou0.7"] = val_com_metrics['R1_iou0.7']
                    best_val_r1iou07["epoch"] = epoch
                    save_model(epoch, args, logger, model, optimizer=None, loss=val_loss, best_model=True, best_name="best_r1iou07")
                logger.info(f"Epoch {epoch}, "
                            f"Best mIoU: {best_val_miou['mIoU']:.4f} in epoch {best_val_miou['epoch']}, "
                            f"Best R1: {best_val_r1['R1']:.4f} in epoch {best_val_r1['epoch']}, "
                            f"Best R5: {best_val_r5['R5']:.4f} in epoch {best_val_r5['epoch']}")
                logger.info(f"Epoch {epoch}, "
                            f"Best R1IoU0.5: {best_val_r1iou05['R1_iou0.5']:.4f} in epoch {best_val_r1iou05['epoch']}, "
                            f"Best R1IoU0.7: {best_val_r1iou07['R1_iou0.7']:.4f} in epoch {best_val_r1iou07['epoch']}")
            # Early stop
            if epoch >= max(60, max(best_val_r1["epoch"], best_val_r5["epoch"], best_val_miou["epoch"], best_val_r1iou05['epoch'], best_val_r1iou07['epoch']) + 20):
                break

if __name__ == '__main__':
    main()