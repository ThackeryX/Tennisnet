_base_ = [
    "../_base_/datasets/fineaction/features_internvideo_resize_trunc.py",
]

# Dataset configuration
data_path = "data/fineaction/fineaction_mae_g/"
block_list = None

dataset = dict(
    train=dict(data_path=data_path, block_list=block_list),
    val=dict(data_path=data_path, block_list=block_list),
    test=dict(data_path=data_path, block_list=block_list),
)

# Model configuration
model = dict(
    type="MambaBMN",
    tscale=192,
    dscale=192,
    prop_boundary_ratio=0.5,
    projection=dict(
        type="ConvSingleProj",
        in_channels=1408,  # Feature dimension for VideoMAE-Giant
        out_channels=256,
        num_convs=2,
    ),
    rpn_head=dict(
        type="MambaBMNHead",
        in_channels=256,
        num_classes=3,
        loss=dict(
            pos_thresh=0.5,
            gt_type=["startness", "endness", "actionness"],
        ),
        mamba_cfg=dict(d_state=16, d_conv=4, expand=2),
    ),
    roi_head=dict(
        type="MambaBMNRoIHead",
        in_channels=256,
        feat_dim=128,
        mamba_cfg=dict(d_state=16, d_conv=4, expand=2),
        loss=dict(
            cls_loss=dict(type="BalancedBCELoss", pos_thresh=0.9),
            reg_loss=dict(
                type="BalancedL2Loss",
                high_thresh=0.7,
                low_thresh=0.3,
                weight=10.0,
            ),
        ),
    ),
)

# Solver settings
solver = dict(
    train=dict(batch_size=16, num_workers=4),
    val=dict(batch_size=16, num_workers=4),
    test=dict(batch_size=16, num_workers=4),
    clip_grad_norm=1,
    ema=True,
)

# Optimizer & Scheduler
optimizer = dict(type="Adam", lr=1e-4, weight_decay=1e-4, paramwise=True)
scheduler = dict(type="MultiStepLR", milestones=[10], gamma=0.1, max_epoch=50)

# Inference and post-processing
inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    nms=dict(
        use_soft_nms=True,
        sigma=0.3,
        max_seg_num=200,
        min_score=0.0001,
        multiclass=True,
        voting_thresh=0.95,
    ),
    external_cls=dict(
        type="StandardClassifier",
        path="data/fineaction/classifiers/new_swinB_1x1x256_views2x3_max_label_avg_prob.json",
        topk=2,
    ),
    save_dict=True,
)

# Runtime workflow
workflow = dict(
    logging_interval=200,
    checkpoint_interval=1,
    val_loss_interval=-1,
    val_eval_interval=1,
    val_start_epoch=0,
)

work_dir = "exps/fineaction/tennisnet"
