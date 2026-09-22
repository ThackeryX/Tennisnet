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
    type="TadTR",
    # ===================== 修改 1：特征维度适配 =====================
    projection=dict(
        type="ConvSingleProj",
        in_channels=1408,  # 【修改】：2048 -> 1408 (匹配 fineaction_mae_g 维度)
        out_channels=256,
        num_convs=1,
        conv_cfg=dict(kernel_size=1, padding=0),
        norm_cfg=dict(type="GN", num_groups=32),
        act_cfg=None,
    ),
    transformer=dict(
        type="TadTRTransformer",
        # ===================== 修改 2：候选框数量扩容 =====================
        # FineAction 动作极其密集，单个视频包含大量动作，原先 THUMOS 的 40 个 query 严重不足
        num_proposals=100,  # 【修改】：40 -> 100 (或 128)
        # ===================== 修改 3：类别数适配 =====================
        num_classes=106,  # 【修改】：20 -> 106 (FineAction 共有 106 个动作类别)
        with_act_reg=True,
        roi_size=16,
        roi_extend_ratio=0.25,
        aux_loss=True,
        position_embedding=dict(
            type="PositionEmbeddingSine",
            num_pos_feats=256,
            temperature=10000,
            offset=-0.5,
            normalize=True,
        ),
        encoder=dict(
            type="DeformableDETREncoder",
            embed_dim=256,
            num_heads=8,
            num_points=4,
            attn_dropout=0.1,
            ffn_dim=1024,
            ffn_dropout=0.1,
            num_layers=4,
            num_feature_levels=1,
            post_norm=False,
        ),
        decoder=dict(
            type="DeformableDETRDecoder",
            embed_dim=256,
            num_heads=8,
            num_points=4,
            attn_dropout=0.1,
            ffn_dim=1024,
            ffn_dropout=0.1,
            num_layers=4,
            num_feature_levels=1,
            return_intermediate=True,
        ),
        loss=dict(
            type="TadTRSetCriterion",
            # ===================== 修改 4：损失函数类别数同步 =====================
            num_classes=106,  # 【修改】：20 -> 106 (必须与模型预测头完全一致)
            matcher=dict(
                type="HungarianMatcher",
                cost_class=6.0,
                cost_bbox=5.0,
                cost_giou=2.0,
                cost_class_type="focal_loss_cost",
                iou_type="iou",
                use_multi_class=True,
            ),
            loss_class_type="focal_loss",
            weight_dict=dict(
                loss_class=2.0,
                loss_bbox=5.0,
                loss_iou=2.0,
                loss_actionness=4.0,
            ),
            use_multi_class=True,  # FineAction 存在重叠动作，保持 True 是正确的
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

work_dir = "exps/fineaction/tadtr_videomaev2_g"
