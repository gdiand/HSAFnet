import os
import argparse
import time
import math

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from builder import build_optimizer, build_logger
from models import MoCo, build_model
from losses import build_loss
from datasets import build_dataset, build_dataset_ccrop

from utils.util import AverageMeter, format_time, set_seed, adjust_learning_rate
from utils.config import Config, ConfigDict, DictAction


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--cfgname', help='specify log_file; for debug use')
    parser.add_argument('--resume', type=str, help='path to resume checkpoint (default: None)')
    parser.add_argument('--load', type=str, help='Load init weights for fine-tune (default: None)')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                        help='update the config; e.g., --cfg-options use_ema=True k1=a,b k2="[a,b]"'
                             'Note that the quotation marks are necessary and that no white space is allowed.')
    args = parser.parse_args()
    return args


def get_cfg(args):
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        dirname = os.path.dirname(args.config).replace('configs', 'checkpoints', 1)
        filename = os.path.splitext(os.path.basename(args.config))[0]
        cfg.work_dir = os.path.join(dirname, filename)
    os.makedirs(cfg.work_dir, exist_ok=True)

    # cfgname
    if args.cfgname is not None:
        cfg.cfgname = args.cfgname
    else:
        cfg.cfgname = os.path.splitext(os.path.basename(args.config))[0]
    assert cfg.cfgname is not None

    # seed
    if args.seed != 0:
        cfg.seed = args.seed
    elif not hasattr(cfg, 'seed'):
        cfg.seed = 42
    set_seed(cfg.seed)

    # resume or load init weights
    if args.resume:
        cfg.resume = args.resume
    if args.load:
        cfg.load = args.load
    assert not (cfg.resume and cfg.load)

    return cfg


def load_weights(ckpt_path, train_set, model, optimizer, resume=True):
    # load checkpoint
    print("==> Loading checkpoint '{}'".format(ckpt_path))
    assert os.path.isfile(ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location='cuda')

    if resume:
        # load model & optimizer
        model.load_state_dict(checkpoint['moco_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    else:
        raise ValueError

    start_epoch = checkpoint['epoch'] + 1
    print("Loaded. (epoch {})".format(checkpoint['epoch']))
    return start_epoch

def load_pretrained(ckpt_path, model, logger=None):
    # 1. 加载文件
    print("==> Loading pretrained weights from '{}'".format(ckpt_path))
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    # 2. 找到真正的权重字典
    if 'state_dict' in checkpoint:
        raw_state_dict = checkpoint['state_dict']
    elif 'simsiam_state' in checkpoint:
        raw_state_dict = checkpoint['simsiam_state']
    elif 'moco_state' in checkpoint: # 增加对 MoCo 保存格式的兼容
        raw_state_dict = checkpoint['moco_state']
    else:
        raw_state_dict = checkpoint

    # 3. 自动探测当前模型是否带有 module. 前缀 (DDP 包装)
    # 我们看模型 state_dict 的第一个键名就知道有没有前缀了
    sample_key = list(model.state_dict().keys())[0]
    has_module_prefix = sample_key.startswith('module.')
    prefix = "module." if has_module_prefix else ""

    # 4. 核心改动：权重映射 (Mapping)
    new_state_dict = {}
    for k, v in raw_state_dict.items():
        # A. 先清理掉权重文件里自带的 module. (无论有没有)
        clean_k = k.replace("module.", "")
        
        # B. 逻辑：如果权重来自 SimSiam (以 encoder. 开头)
        if clean_k.startswith("encoder."):
            suffix = clean_k.replace("encoder.", "") # 拿到 layer1.0...
            # 映射到 q 侧和 k 侧，并根据当前模型状态补上 prefix
            new_state_dict[f"{prefix}encoder_q.{suffix}"] = v
            new_state_dict[f"{prefix}encoder_k.{suffix}"] = v
        
        # C. 逻辑：如果权重已经是 MoCo 格式 (以 encoder_q. 开头)
        elif clean_k.startswith("encoder_q."):
            suffix = clean_k.replace("encoder_q.", "")
            new_state_dict[f"{prefix}encoder_q.{suffix}"] = v
            new_state_dict[f"{prefix}encoder_k.{suffix}"] = v
        
        # D. 其他不需要映射的参数 (如 queue)
        else:
            new_state_dict[f"{prefix}{clean_k}"] = v

    # 5. 执行加载
    # 用 strict=False 忽略没存进去的 queue 和 queue_ptr
    msg = model.load_state_dict(new_state_dict, strict=False)
    
    # 6. 打印报告
    if logger:
        logger.info(f"=> Load Status: Missing {len(msg.missing_keys)} keys, Unexpected {len(msg.unexpected_keys)} keys")
    else:
        print(f"Pretrained Loaded.\nMatched: {len(new_state_dict)} keys")
        # 如果 Missing 还是很多，打印前 5 个看看是什么名
        if len(msg.missing_keys) > 2:
            print(f"First 5 missing: {msg.missing_keys[:5]}")

    return model


def train(train_loader, model, criterion, optimizer, epoch, cfg, logger, writer):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    num_iter = len(train_loader)
    end = time.time()
    time1 = time.time()
    for idx, (images, _) in enumerate(train_loader):
        images[0] = images[0].cuda(non_blocking=True)
        images[1] = images[1].cuda(non_blocking=True)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        output_global, output_obj, target = model(im_q=images[0], im_k=images[1])
        loss_global = criterion(output_global, target)
        loss_obj = criterion(output_obj, target)
        loss = loss_global + 1.0 * loss_obj
        losses.update(loss.item(), images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # print info
        if (idx + 1) % cfg.log_interval == 0 and logger is not None:  # cfg.rank == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(f'Epoch [{epoch}][{idx+1}/{num_iter}] - '
                        f'data_time: {data_time.avg:.3f},     '
                        f'batch_time: {batch_time.avg:.3f},     '
                        f'lr: {lr:.5f},     '
                        f'loss_g: {loss_global:.3f},       '
                        f'loss_p: {loss_obj:.3f}')

    if logger is not None:  # cfg.rank == 0
        time2 = time.time()
        epoch_time = format_time(time2 - time1)
        logger.info(f'Epoch [{epoch}] - epoch_time: {epoch_time}, '
                    f'train_loss: {losses.avg:.3f}')
    if writer is not None:
        lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Pretrain/lr', lr, epoch)
        writer.add_scalar('Pretrain/loss', losses.avg, epoch)


def main():
    # args & cfg
    args = parse_args()
    cfg = get_cfg(args)

    world_size = torch.cuda.device_count()
    print('GPUs on this node:', world_size)
    cfg.world_size = world_size

    # write cfg
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(cfg.work_dir, f'{timestamp}.cfg')
    with open(log_file, 'a') as f:
        f.write(cfg.pretty_text)

    # spawn
    mp.spawn(main_worker, nprocs=world_size, args=(world_size, cfg))


def main_worker(rank, world_size, cfg):
    print('==> Start rank:', rank)

    local_rank = rank % 8
    cfg.local_rank = local_rank
    torch.cuda.set_device(local_rank)

    dist.init_process_group(backend='nccl', init_method=f'tcp://localhost:{cfg.port}',
                            world_size=world_size, rank=rank)

    # build logger, writer
    logger, writer = None, None
    if rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(cfg.work_dir, 'tensorboard'))
        logger = build_logger(cfg.work_dir, 'pretrain')

    # build data loader
    bsz_gpu = int(cfg.batch_size / cfg.world_size)
    print('batch_size per gpu:', bsz_gpu)

    train_set = build_dataset_ccrop(cfg.data.train)
    len_ds = len(train_set)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=bsz_gpu,
        num_workers=cfg.num_workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True
    )
    eval_train_set = build_dataset(cfg.data.eval_train)
    eval_train_sampler = torch.utils.data.distributed.DistributedSampler(eval_train_set, shuffle=False)
    eval_train_loader = torch.utils.data.DataLoader(
        eval_train_set,
        batch_size=bsz_gpu,
        num_workers=cfg.num_workers,
        pin_memory=True,
        sampler=eval_train_sampler,
        drop_last=False
    )

    # build model, criterion; optimizer
    encoder_q = build_model(cfg.model)
    encoder_k = build_model(cfg.model)
    model = MoCo(encoder_q, encoder_k, **cfg.moco)  # cfg.moco.dim, K, m, T, mlp
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.local_rank])
    criterion = build_loss(cfg.loss).cuda()
    optimizer = build_optimizer(cfg.optimizer, model.parameters())
    start_epoch = 1
    if cfg.resume:
        start_epoch = load_weights(cfg.resume, train_set, model, optimizer, resume=True)
    elif hasattr(cfg, 'load') and cfg.load:
        # 注意：如果是 DDP 模型，建议传入 model.module
        load_pretrained(cfg.load, model.module if hasattr(model, 'module') else model, logger)
    cudnn.benchmark = True

    # Start training
    print("==> Start training...")
    for epoch in range(start_epoch, cfg.epochs + 1):
        train_sampler.set_epoch(epoch)
        adjust_learning_rate(cfg.lr_cfg, optimizer, epoch)

        # train; all processes
        train(train_loader, model, criterion, optimizer, epoch, cfg, logger, writer)

        # save ckpt; master process
        if rank == 0 and epoch % cfg.save_interval == 0:
            model_path = os.path.join(cfg.work_dir, f'epoch_{epoch}.pth')
            state_dict = {
                'optimizer_state': optimizer.state_dict(),
                'moco_state': model.state_dict(),
                'epoch': epoch
            }
            torch.save(state_dict, model_path)

    # save the last model; master process
    if rank == 0:
        model_path = os.path.join(cfg.work_dir, 'last.pth')
        state_dict = {
            'optimizer_state': optimizer.state_dict(),
            'moco_state': model.state_dict(),
            'epoch': cfg.epochs
        }
        torch.save(state_dict, model_path)


if __name__ == '__main__':
    main()
