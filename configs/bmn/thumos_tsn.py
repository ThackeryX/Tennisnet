_base_ = [
    "../_base_/datasets/thumos-14/features_tsn_sw.py",  # dataset config
    # "../_base_/models/bmn.py",  # model config
]

window_size = 125
dataset = dict(
    train=dict(
        feature_stride=1,
        sample_stride=1,
        window_size=window_size,
        window_overlap_ratio=0.5,
        ioa_thresh=0.9,
    ),
    val=dict(
        feature_stride=1,
        sample_stride=1,
        window_size=window_size,
        window_overlap_ratio=0.5,
        ioa_thresh=0.9,
    ),
    test=dict(
        feature_stride=1,
        sample_stride=1,
        window_size=window_size,
        window_overlap_ratio=0.5,
    ),
)



model = dict(
    type="MambaBMN",
    use_global_mamba=True,
    global_mamba_cfg=dict(
        bidirectional=True,   # 双向扫描
        use_conv=True,
        shared_conv=True,     # 共享卷积
    ),
    tscale=125,
    dscale=125,
    prop_boundary_ratio=0.5,
    projection=dict(
        type="ConvSingleProj",
        in_channels=2048,
        out_channels=256,
        num_convs=2,
        conv_cfg=dict(groups=4),
    ),
    rpn_head=dict(
        type="MambaBMNHead",
        use_mamba=True,       # 修正为 True
        in_channels=256,
        num_classes=3,
        loss=dict(pos_thresh=0.5, gt_type=["startness", "endness", "actionness"]), # 包含 Actionness
        mamba_cfg=dict(d_state=16, d_conv=4, expand=2),
    ),
    roi_head=dict(
        type="MambaBMNRoIHead",
        use_mamba=True,       # 修正为 True
        in_channels=256,
        feat_dim=128,
        d_axis_mode="mamba",  # D 轴使用 Mamba 建模
        mamba_cfg=dict(d_state=16, d_conv=4, expand=2),
    ),
)





solver = dict(
    train=dict(batch_size=16, num_workers=4),
    val=dict(batch_size=16, num_workers=4),
    test=dict(batch_size=16, num_workers=4),
    clip_grad_norm=1,
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
        multiclass=False,# False
        voting_thresh=0.95,  #  set 0 to disable
    ),
    external_cls=dict(
        type="UntrimmedNetTHUMOSClassifier",
        path="/home/cipan/tennis/OpenTAD-main/data/custom_scores.npy",
        # path="/home/cipan/tennis/OpenTAD-main/data1/thumos/classifiers/uNet_test.npy",
        topk=1,
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

work_dir = "exps/xiaorong/1_0_0"






