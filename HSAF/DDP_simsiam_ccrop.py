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
from models import SimSiam, build_model
from losses import build_loss
from datasets import build_dataset, build_dataset_ccrop

from utils.util import AverageMeter, format_time, set_seed, adjust_lr_simsiam
from utils.config import Config, ConfigDict, DictAction
import torchvision.utils as vutils

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
        model.load_state_dict(checkpoint['simsiam_state'])
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
    # 按照优先级匹配常见的 SimSiam 键名
    if 'simsiam_state' in checkpoint:
        raw_state_dict = checkpoint['simsiam_state']
    elif 'state_dict' in checkpoint:
        raw_state_dict = checkpoint['state_dict']
    elif 'simclr_state' in checkpoint: # 兼容你之前的逻辑
        raw_state_dict = checkpoint['simclr_state']
    else:
        raw_state_dict = checkpoint

    # 3. 处理 DDP 带来的 "module." 前缀问题
    # 如果权重里有 module. 而你的模型定义里没有（或反之），会导致加载失败
    new_state_dict = {}
    for k, v in raw_state_dict.items():
        # 如果权重有 module. 前缀，而当前模型没有，则去掉前缀
        name = k.replace("module.", "") 
        new_state_dict[name] = v

    # 4. 加载权重 (使用 strict=False 允许缺失新加的 27 层)
    msg = model.load_state_dict(new_state_dict, strict=False)
    
    # 5. 打印加载报告
    if logger:
        logger.info(f"=> Pretrained Loaded. Missing keys: {len(msg.missing_keys)}")
        logger.info(f"=> Unexpected keys: {len(msg.unexpected_keys)}")
    else:
        print(f"Pretrained Loaded. \nMissing: {len(msg.missing_keys)} layers\nUnexpected: {len(msg.unexpected_keys)} layers")
    
    # 如果你想确认缺失的是不是只有咱们新加的那几层，可以取消下面注释
    # print(f"Missing detail: {msg.missing_keys[:5]}") 

    return model
def computer_cossim(p,z):
    z = F.normalize(z, dim=-1).unsqueeze(1) 
    p = F.normalize(p, dim=-1) 
    cos_sim = (p * z).sum(dim=-1)

    return cos_sim

import matplotlib.pyplot as plt

def vision(act,path):
    save_dir = path
    os.makedirs(save_dir, exist_ok=True)

    plt.imshow(act, cmap='jet')
    plt.colorbar()
    plt.title("Activation Map")

    num_files = len(os.listdir(save_dir))
    save_path = os.path.join(save_dir, f"{num_files}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close() 

    print(f"结果已保存至: {save_path}")

import torch
import torch.nn.functional as F

def get_tiered_patch_weights(feature_map, cold_percentile=0.4, hot_percentile=0.85, smooth_kernel=3):

    B, C, H, W = feature_map.shape
    N = H * W
    

    activation = torch.norm(feature_map, p=2, dim=1)
    
    act_flat = activation.view(B, N)
    
    # act = activation[0].detach().cpu().numpy()
    # vision(act,"/home/data3t/zhaoyuxiang/CL/ContrastiveCrop-mainn/ContrastiveCrop/pic/1.jpg")
    # exit()

    q_cold = torch.quantile(act_flat, cold_percentile, dim=1, keepdim=True)
    q_hot = torch.quantile(act_flat, hot_percentile, dim=1, keepdim=True)

    weights = torch.zeros_like(act_flat)
    

    

    warm_mask = (act_flat >= q_cold) & (act_flat <= q_hot)
    weights[warm_mask] = 1.0
    

    hot_mask = act_flat > q_hot
    weights[hot_mask] = 0.6 


    weights = weights.view(B, 1, H, W)
    weights = F.avg_pool2d(weights, kernel_size=smooth_kernel, stride=1, padding=smooth_kernel//2)
    

    background_mask = (activation < q_cold.view(B, 1, 1))
    weights[background_mask.unsqueeze(1)] = 0.0

    weights = weights.view(B, -1)
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-7)

    return weights.view(B, H, W)

def weighted_dense_to_global_loss(p_dense, z_global, weights):

    B, N, D = p_dense.shape
    w_flat = weights.view(B, N)
    
    p_dense = F.normalize(p_dense, dim=-1)
    z_global = F.normalize(z_global, dim=-1).unsqueeze(1)
    
    cos_sim = (p_dense * z_global).sum(dim=-1) # (B, 144)
    
    loss = -(cos_sim * w_flat).sum(dim=1)
    
    return loss.mean()

def get_object_representation(feature_map, weights):

    B, C, H, W = feature_map.shape
    
    w = weights.unsqueeze(1) 
    

    z = (feature_map * w).sum(dim=[2,3]) / (w.sum(dim=[2,3]) + 1e-6)
    
    return z


def denormalize(x, mean = (0.4406, 0.4273, 0.3858),std = (0.2312, 0.2265, 0.2237)):
    mean = torch.tensor(mean).view(1,3,1,1).to(x.device)
    std = torch.tensor(std).view(1,3,1,1).to(x.device)
    return x * std + mean

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
        x1 = images[0].cuda(non_blocking=True)
        x2 = images[1].cuda(non_blocking=True)

        # x1_denorm = denormalize(x1)
        # x1_denorm = torch.clamp(x1_denorm, 0, 1)
        # vutils.save_image(x1_denorm[0], "/home/data3t/zhaoyuxiang/CL/ContrastiveCrop-mainn/ContrastiveCrop/pic/orig/1.jpg")
        # print("111")
        # patches1 = patches1.cuda(non_blocking=True)
        # patches2 = patches2.cuda(non_blocking=True)


        # measure data time
        data_time.update(time.time() - end)
        
        # compute output
        feature1,feature2,p1, p2, z1, z2= model(x1, x2)
        # feature1,feature2,feature3,feature4=model(x1,x2)
        with torch.no_grad():
            weight_fea1=get_tiered_patch_weights(feature1)
            weight_fea2=get_tiered_patch_weights(feature2)
        z1_obj_backbone = get_object_representation(feature1, weight_fea1)
        z2_obj_backbone = get_object_representation(feature2, weight_fea2)

        z1_obj = model.module.encoder.fc(z1_obj_backbone)
        z2_obj = model.module.encoder.fc(z2_obj_backbone)

        p1_obj = model.module.predictor(z1_obj)
        p2_obj = model.module.predictor(z2_obj)

        # loss_patch = 0.5 * (
        #     weighted_dense_to_global_loss(p1_patch_flat, z2, weight_fea1.view(128, -1)) + 
        #     weighted_dense_to_global_loss(p2_patch_flat, z1, weight_fea2.view(128, -1))
        # )

        loss_obj = -0.5 * (
            (F.cosine_similarity(p1_obj, z2_obj.detach(), dim=-1).mean()) +
            (F.cosine_similarity(p2_obj, z1_obj.detach(), dim=-1).mean())
        )
        
        # loss_patch = -0.5 * (criterion(p1_patch, z2_patch).mean() + criterion(p2_patch, z1_patch).mean())

        loss_global = -0.5 * (criterion(p1, z2).mean() + criterion(p2, z1).mean())
        # loss_patch = -0.5 * (criterion(patch1, z2).mean() + criterion(patch2, z1).mean())
        # loss_patch = (distribution_loss(dist_p_proto1, dist_z2) + distribution_loss(dist_p_proto2, dist_z1)) * 0.5
                                                                                                                                                                            
        loss = loss_global+loss_obj
        losses.update(loss.item(), x1.size(0))

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
                        f'loss: {loss:.3f}({losses.avg:.3f})     '
                        f'global:{loss_global:.3f}     '
                        f'patch:{loss_obj:.3f}')

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
    encoder = build_model(cfg.model)
    model = SimSiam(encoder)  # cfg.simsiam.dim, pred_dim
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.local_rank])
    criterion = build_loss(cfg.loss).cuda()
    if cfg.fix_pred_lr:
        optim_params = [{'params': model.module.encoder.parameters(), 'fix_lr': False},
                        {'params': model.module.predictor.parameters(), 'fix_lr': True}]
    else:
        optim_params = model.parameters()
    optimizer = build_optimizer(cfg.optimizer, optim_params)

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
        adjust_lr_simsiam(cfg.lr_cfg, optimizer, epoch)

        # train; all processes
        train(train_loader, model, criterion, optimizer, epoch, cfg, logger, writer)

        # save ckpt; master process
        if rank == 0 and epoch % 50 == 0:
            model_path = os.path.join(cfg.work_dir, f'epoch_{epoch}.pth')
            state_dict = {
                'optimizer_state': optimizer.state_dict(),
                'simsiam_state': model.state_dict(),
                'epoch': epoch
            }
            torch.save(state_dict, model_path)

    # save the last model; master process
    if rank == 0:
        model_path = os.path.join(cfg.work_dir, 'last.pth')
        state_dict = {
            'optimizer_state': optimizer.state_dict(),
            'simsiam_state': model.state_dict(),
            'epoch': cfg.epochs
        }
        torch.save(state_dict, model_path)


if __name__ == '__main__':
    main()
