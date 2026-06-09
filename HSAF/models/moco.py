# Adapted from https://github.com/facebookresearch/moco/blob/main/moco/builder.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_tiered_patch_weights(feature_map, cold_percentile=0.5, hot_percentile=0.95, smooth_kernel=3):
    B, C, H, W = feature_map.shape
    N = H * W
    

    xianzhu = torch.norm(feature_map, p=2, dim=1)
    flat = xianzhu.view(B, N)
    # pic = xianzhu[0].detach().cpu().numpy()
    # vision(pic,"/home/data3t/zhaoyuxiang/CL/ContrastiveCrop-mainn/ContrastiveCrop/pic/1.jpg")
    # exit()

    q_cold = torch.quantile(flat, cold_percentile, dim=1, keepdim=True)
    q_hot = torch.quantile(flat, hot_percentile, dim=1, keepdim=True)

    weights = torch.zeros_like(flat)
    

    

    warm_mask = (flat >= q_cold) & (flat <= q_hot)
    weights[warm_mask] = 1.0
    

    hot_mask = flat > q_hot
    weights[hot_mask] = 0.6 


    weights = weights.view(B, 1, H, W)
    weights = F.avg_pool2d(weights, kernel_size=smooth_kernel, stride=1, padding=smooth_kernel//2)
    

    background_mask = (xianzhu < q_cold.view(B, 1, 1))
    weights[background_mask.unsqueeze(1)] = 0.0

    weights = weights.view(B, -1)
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-7)

    return weights.view(B, H, W)

def weighted_dense_to_global_loss(p_dense, z_global, weights):
    """
    p_dense: (B, 144, D)
    z_global: (B, D)
    weights: (B, 12, 12)
    """
    B, N, D = p_dense.shape
    w_flat = weights.view(B, N)
    
    p_dense = F.normalize(p_dense, dim=-1)
    z_global = F.normalize(z_global, dim=-1).unsqueeze(1)
    
    cos_sim = (p_dense * z_global).sum(dim=-1) # (B, 144)
    
    loss = -(cos_sim * w_flat).sum(dim=1)
    
    return loss.mean()

def get_object_representation(feature_map, weights):
    """
    feature_map: (B, C, H, W)
    weights: (B, H, W)
    return: (B, C)
    """
    B, C, H, W = feature_map.shape
    
    w = weights.unsqueeze(1)  # (B,1,H,W)
    

    z = (feature_map * w).sum(dim=[2,3]) / (w.sum(dim=[2,3]) + 1e-6)
    
    return z


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, encoder_q, encoder_k, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        self._features = []

        def _hook_fn(module, input, output):
            self._features.append(output)

        
        # num_classes is the output fc dimension
        self.encoder_q = encoder_q
        self.encoder_k = encoder_k

        self.encoder_q.layer4.register_forward_hook(_hook_fn)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        self._features = []

        z_q_global = self.encoder_q(im_q)  # queries: NxC
        q_global = nn.functional.normalize(z_q_global, dim=1)

        feat_q = self._features[0]

        with torch.no_grad():
            weights_q = get_tiered_patch_weights(feat_q)

        z_q_obj_backbone = get_object_representation(feat_q, weights_q)
        z_q_obj = self.encoder_q.fc(z_q_obj_backbone)
        q_obj = nn.functional.normalize(z_q_obj, dim=1)



        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1

        l_pos_global = torch.einsum('nc,nc->n', [q_global, k]).unsqueeze(-1)
        l_neg_global = torch.einsum('nc,ck->nk', [q_global, self.queue.clone().detach()])
        logits_global = torch.cat([l_pos_global, l_neg_global], dim=1)

        l_pos_obj = torch.einsum('nc,nc->n', [q_obj, k]).unsqueeze(-1)
        l_neg_obj = torch.einsum('nc,ck->nk', [q_obj, self.queue.clone().detach()])
        logits_obj = torch.cat([l_pos_obj, l_neg_obj], dim=1)


        

        # labels: positive key indicators
    
        labels = torch.zeros(logits_global.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits_global / self.T, logits_obj / self.T, labels


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
