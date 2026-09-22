_base_ = [
    "../_base_/datasets/fineaction/features_internvideo_resize_trunc.py",  # dataset config
    # "../_base_/models/actionformer.py",  # model config
]

data_path = "data1/fineaction/fineaction_mae_g/"
block_list = None # data_path + "missing_files.txt"
dataset = dict(
    train=dict(data_path=data_path, block_list=block_list),
    val=dict(data_path=data_path, block_list=block_list),
    test=dict(data_path=data_path, block_list=block_list),
)

model = dict(
    type="TemporalMaxer",
    # ===================== 修改 1：特征维度适配 =====================
    projection=dict(
        type="TemporalMaxerProj",
        in_channels=1408,  # 【修改】：2048 -> 1408 (匹配 fineaction_mae_g)
        out_channels=512,
        arch=(2, 0, 5),  # 特征投影层与下采样层结构
        conv_cfg=dict(kernel_size=3),
        norm_cfg=dict(type="LN"),
    ),
    neck=dict(
        type="FPNIdentity",
        in_channels=512,
        out_channels=512,
        num_levels=6,  # 6 层时序特征金字塔
    ),
    rpn_head=dict(
        type="TemporalMaxerHead",
        # ===================== 修改 2：类别数适配 =====================
        num_classes=106,  # 【修改】：20 -> 106 (FineAction 共有 106 个分类)
        in_channels=512,
        feat_channels=512,
        num_convs=2,
        cls_prior_prob=0.01,
        prior_generator=dict(
            type="PointGenerator",
            strides=[1, 2, 4, 8, 16, 32],
            regression_range=[
                (0, 4),
                (4, 8),
                (8, 16),
                (16, 32),
                (32, 64),
                (64, 10000),
            ],
        ),
        loss_normalizer=100,
        loss_normalizer_momentum=0.9,
        loss=dict(
            cls_loss=dict(type="FocalLoss"),
            reg_loss=dict(type="DIOULoss"),
        ),
        assigner=dict(
            type="AnchorFreeSimOTAAssigner",
            iou_weight=2,
            cls_weight=1.0,
            center_radius=1.5,
            keep_percent=1.0,
            confuse_weight=0.0,
        ),
    ),
)
solver = dict(
    train=dict(batch_size=16, num_workers=4),
    val=dict(batch_size=16, num_workers=4),
    test=dict(batch_size=16, num_workers=4),
    clip_grad_norm=1,
    ema=True,
)

# optimizer = dict(type="AdamW", lr=1e-3, weight_decay=0.05, paramwise=True)
# scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=50)
#
# inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
# post_processing = dict(
#     nms=dict(
#         use_soft_nms=True,
#         sigma=0.75,
#         max_seg_num=100,
#         iou_threshold=0,  # does not matter when use soft nms
#         min_score=0.001,
#         multiclass=True,
#         voting_thresh=0.9,  #  set 0 to disable
#     ),
#     external_cls=dict(
#         type="StandardClassifier",
#         path="./data1/fineaction/classifiers/new_swinB_1x1x256_views2x3_max_label_avg_prob.json",
#         topk=2,
#     ),
#     save_dict=True,
# )

optimizer = dict(type="Adam", lr=1e-4, weight_decay=1e-4, paramwise=True)
scheduler = dict(type="MultiStepLR", milestones=[10], gamma=0.1, max_epoch=50)


inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    nms=dict(
        use_soft_nms=True,
        sigma=0.3,
        max_seg_num=200,
        min_score=0.0001,
        multiclass=True,# False
        voting_thresh=0.95,  #  set 0 to disable
    ),

    external_cls=dict(
        type="StandardClassifier",
        path="./data1/fineaction/classifiers/new_swinB_1x1x256_views2x3_max_label_avg_prob.json",
        topk=2,
    ),
    save_dict=True,
)



workflow = dict(
    logging_interval=200,
    checkpoint_interval=1,
    val_loss_interval=-1,
    val_eval_interval=1,
    val_start_epoch=0,
)

work_dir = "exps/fineaction/temporalmaxer_1_videomaev2_g"
