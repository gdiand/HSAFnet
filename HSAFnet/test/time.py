import torch
from fvcore.nn import FlopCountAnalysis
from models import MoCo
# 假设 model 是你的模型，input 是输入 tensor
model = MoCo(dim=128, K=65536, m=0.999, T=0.20, mlp=True)
input_data = torch.randn(1, 3, 224, 224)

# 计算 FLOPs
flop_analyzer = FlopCountAnalysis(model, input_data)
print(f"FLOPs: {flop_analyzer.total() / 1e9:.2f} G")

# 计算参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters: {total_params / 1e6:.2f} M")