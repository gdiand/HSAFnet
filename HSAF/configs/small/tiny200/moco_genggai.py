# python DDP_moco_ccrop.py path/to/this/config

# model
dim = 128
model = dict(type='ResNet', depth=18, num_classes=dim, maxpool=False)
moco = dict(dim=dim, K=65536, m=0.999, T=0.20, mlp=True)
loss = dict(type='CrossEntropyLoss')

# data
root = '/home/data3t/zhaoyuxiang/CL/Data/tiny-imagenet-200/train'
mean = (0.4802, 0.4481, 0.3975)
std = (0.2302, 0.2265, 0.2262)
batch_size = 256
num_workers = 4
data = dict(
    train=dict(
        ds_dict=dict(
            type='Tiny200_boxes',
            root=root,
        ),
        rcrop_dict=dict(
            type='tiny200_train_rcrop',
            mean=mean, std=std
        ),
    ),
    eval_train=dict(
        ds_dict=dict(
            type='ImageFolder',
            root=root,
        ),
        trans_dict=dict(
            type='tiny200_test',
            mean=mean, std=std
        ),
    ),
)

# boxes
fix_pred_lr = True
# training optimizer & scheduler
epochs = 300

lr = 0.05 * batch_size / 256
optimizer = dict(type='SGD', lr=lr, momentum=0.9, weight_decay=1e-4)
lr_cfg = dict(  # passed to adjust_learning_rate(cfg=lr_cfg)
    type='Cosine',
    steps=epochs,
    lr=lr,
    decay_rate=0.1,
    # decay_steps=[100, 150]
    warmup_steps=0,
    # warmup_from=0.01
)


# log & save
log_interval = 20
save_interval = 50
work_dir = None  # rewritten by args
resume = None
load = None
port = 10001
