_base_ = [
    "../_base_/datasets/fineaction/features_internvideo_resize_trunc.py",  # dataset config
    # "../_base_/models/bmn.py",  # model config
]

data_path = "data1/fineaction/fineaction_mae_g/"
block_list = None # data_path + "missing_files.txt"
dataset = dict(
    train=dict(data_path=data_path, block_list=block_list),
    val=dict(data_path=data_path, block_list=block_list),
    test=dict(data_path=data_path, block_list=block_list),
)

model = dict(
    type="GTAD",
    # ===================== 修改 1：特征维度适配 =====================
    projection=dict(
        type="ConvSingleProj",
        num_convs=1,
        in_channels=1408,  # 【修改】：400 -> 1408 (匹配 fineaction_mae_g 特征维度)
        out_channels=256,
        conv_cfg=dict(groups=4),  # 1408 能被 4 整除 (352)，保留分组卷积加速
    ),
    neck=dict(
        type="GCNeXt",
        in_channels=256,
        out_channels=256,
        k=3,
        groups=32,
    ),
    rpn_head=dict(
        type="GCNextTemporalEvaluationHead",
        in_channels=256,
        num_classes=2,
        loss=dict(pos_thresh=0.5, gt_type=["startness", "endness"]),
    ),
    roi_head=dict(
        type="StandardProposalMapHead",
        # ===================== 修改 2：时序尺度与输入对齐 =====================
        # 必须与数据流水线中的 resize_length: 192 保持一致，防止 tensordot 维度报错
        proposal_generator=dict(
            type="DenseProposalMap",
            tscale=192,  # 【修改】：100 -> 192
            dscale=192,  # 【修改】：100 -> 192
        ),
        proposal_roi_extractor=dict(
            type="GTADExtractor",
            in_channels=256,
            out_channels=512,
            tscale=192,  # 【修改】：100 -> 192 (必须与 proposal_generator 一致)
            dscale=192,  # 【修改】：100 -> 192
        ),
        proposal_head=dict(
            type="PEMHead",  # FC_head
            in_channels=512,
            feat_channels=128,
            num_convs=3,
            num_classes=2,
            kernel_size=1,
            loss=dict(
                cls_loss=dict(type="BalancedBCELoss", pos_thresh=0.9),
                reg_loss=dict(
                    type="BalancedL2Loss",
                    high_thresh=0.7,
                    low_thresh=0.3,
                    weight=5.0,
                ),
            ),
        ),
    ),
)



solver = dict(
    train=dict(batch_size=4, num_workers=4),
    val=dict(batch_size=4, num_workers=4),
    test=dict(batch_size=4, num_workers=4),
    clip_grad_norm=1,
    ema=True,
)

optimizer = dict(type="Adam", lr=1e-4, weight_decay=1e-4, paramwise=True)
scheduler = dict(type="MultiStepLR", milestones=[20], gamma=0.1, max_epoch=50)

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

work_dir = "exps/fineaction/gtad_videomaev2_g"
