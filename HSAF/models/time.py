import torch
import time
import numpy as np
from moco import MoCo
from resnet import ResNet

# 1. 初始化模型并移至 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder_q = ResNet(depth=18, num_classes=10, maxpool=False, zero_init_residual=True).to(device)
encoder_k = ResNet(depth=18, num_classes=10, maxpool=False, zero_init_residual=True).to(device)
model = MoCo(encoder_q, encoder_k, dim=128, K=65536, m=0.999, T=0.20, mlp=True).to(device)
model.eval()

# 2. 准备输入 (使用真实推理场景的形状)
# 注意：我们只测 encoder_q，因为它是实际推理时的核心
dummy_input = torch.randn(1, 3, 224, 224).to(device)

# 3. 预热 (Warm-up)
# 非常重要！GPU 第一次运行时需要分配内存、加载库，必须先跑几十次预热
print("Warming up...")
with torch.no_grad():
    for _ in range(50):
        _ = model.encoder_q(dummy_input)

# 4. 正式测量
print("Measuring latency...")
times = []
with torch.no_grad():
    for _ in range(200): # 测量 200 次取平均
        start_time = time.time()
        _ = model.encoder_q(dummy_input)
        # 如果你用的是 GPU，必须同步才能准确计时
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()
        times.append((end_time - start_time) * 1000) # 转换为毫秒

# 5. 输出结果
avg_time = np.mean(times)
std_time = np.std(times)
print(f"Average Inference Latency: {avg_time:.2f} ± {std_time:.2f} ms")