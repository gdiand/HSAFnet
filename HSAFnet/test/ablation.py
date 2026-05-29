import subprocess

ckpts = [
    "/home/data3t/zhaoyuxiang/CL/ckpt/genggai_moco/tiny_200/epoch_100.pth",
    "/home/data3t/zhaoyuxiang/CL/ckpt/genggai_moco/tiny_200/epoch_200.pth",
    "/home/data3t/zhaoyuxiang/CL/ckpt/genggai_moco/tiny_yuzhi/2-6.5/epoch_300.pth",
    "/home/data3t/zhaoyuxiang/CL/ckpt/genggai_moco/tiny_yuzhi/epoch_300.pth",
    "/home/data3t/zhaoyuxiang/CL/ckpt/genggai_moco/tiny_yuzhi/5-9.5/epoch_300.pth",
]

for i, ckpt in enumerate(ckpts):

    print(f"\n========== Experiment {i+1} ==========")
    print(f"Loading: {ckpt}")

    cmd = [
        "python",
        "/home/data3t/zhaoyuxiang/CL/ContrastiveCrop-mainn/ContrastiveCrop/DDP_linear.py",
        "/home/data3t/zhaoyuxiang/CL/ContrastiveCrop-mainn/ContrastiveCrop/configs/linear/tiny200_res18.py",
        "--load",
        ckpt
    ]

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"Experiment {i+1} failed!")
    else:
        print(f"Experiment {i+1} finished!")