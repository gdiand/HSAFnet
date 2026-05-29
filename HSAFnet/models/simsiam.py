# Adapted from https://github.com/facebookresearch/simsiam/blob/main/simsiam/builder.py
import torch.nn as nn


import torch
import torch.nn as nn
import torch.nn.functional as F

class SimSiam(nn.Module):
    def __init__(self, encoder, dim=512, pred_dim=128):
        super(SimSiam, self).__init__()

        self.encoder = encoder
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),
            self.encoder.fc, 
            nn.BatchNorm1d(dim, affine=False)
        )
        self.encoder.fc[6].bias.requires_grad = False
 
        self._features = []

        def _hook_fn(module, input, output):
            self._features.append(output)

        # self.encoder.layer1.register_forward_hook(_hook_fn)    
        # self.encoder.layer2.register_forward_hook(_hook_fn)
        # self.encoder.layer3.register_forward_hook(_hook_fn)
        self.encoder.layer4.register_forward_hook(_hook_fn)

        self.predictor = nn.Sequential(
            nn.Linear(dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_dim, dim)
        )
        

    #     self.object_predictor = nn.Sequential(
    # # 1x1 卷积本质上就是对每一个 12x12 的点分别做 MLP
    #         nn.Linear(dim, pred_dim, bias=False),
    #         nn.BatchNorm2d(pred_dim),
    #         nn.ReLU(inplace=True),
    #         nn.Linear(pred_dim, dim)
    #     )

        # self.patch_dist_head = nn.Linear(dim, 2048, bias=False)
        
        # self.t_s = 0.1 
        # self.t_t = 0.05 



    def get_dist(self, x, head, temp):
        x = head(x)
        return F.softmax(x / temp, dim=-1)
    
    def get_tiered_patch_weights(self,feature_map, cold_percentile=0.4, hot_percentile=0.85, smooth_kernel=3):
        """
        针对 12x12 特征图优化的重分布机制
        feature_map: (B, C, 12, 12)
        hot_percentile: 0.85 表示前 15% 为核心区
        """
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
    
    def mix_feature(self,feature,weight):

        B, C, H, W = feature.shape
        w = weight.unsqueeze(1) 
        
        z_weighted = (feature * w).sum(dim=(2, 3))

        w_sum = w.sum(dim=(2, 3)) + 1e-7
        z_weighted = z_weighted / w_sum
        return F.normalize(z_weighted, p=2, dim=-1)


    def forward(self, x1, x2):

        self._features= []
    
        z1 = self.encoder(x1) # [B, dim]
        z2 = self.encoder(x2) # [B, dim]

        fea1=self._features[0]
        fea2=self._features[1]

        # fea_z1=self.patch_projector(fea1)
        # fea_z2=self.patch_projector(fea2)
        # fea3=self._features[2]
        # fea4=self._features[3]
        # B, P, C, h, w = patches1.shape

        # z1_p = self.encoder(patches1.view(-1, C, h, w)).view(B, P, -1)
        # z2_p = self.encoder(patches2.view(-1, C, h, w)).view(B, P, -1) 

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        # proto1 = self.attention_aggregate(z1_p, z2.detach())
        # proto2 = self.attention_aggregate(z2_p, z1.detach())

        # p_proto1 = self.patch_predictor(proto1)
        # p_proto2 = self.patch_predictor(proto2)

        # dist_z1 = self.get_dist(z1, self.dist_head, self.t_t)
        # dist_z2 = self.get_dist(z2, self.dist_head, self.t_t)
        
        # dist_p_proto1 = self.get_dist(p_proto1, self.dist_head, self.t_s)
        # dist_p_proto2 = self.get_dist(p_proto2, self.dist_head, self.t_s)

        # return fea1,fea2,fea3,fea4
        return fea1,fea2,p1, p2, z1.detach(), z2.detach()
        # return p1, p2, z1.detach(), z2.detach(),p_m1,p_m2,z1_patch.detach(),z2_patch.detach()
