_base_ = '../mask2former_swin-t_8xb2-160k_ade20k-512x512.py'

# Image and crop settings
crop_size = (768, 768)

# Optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.0005, eps=1e-8, betas=(0.9, 0.999)),
    clip_grad=dict(max_norm=0.1, norm_type=2)
)

# Training schedule
max_epochs = 100000
train_dataloader = dict(batch_size=2)
