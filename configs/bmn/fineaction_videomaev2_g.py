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


# model = dict(
#     type="BMN",
#     neck=None,
#     # 1. 投影层：将 1408 维特征压缩至 256
#     projection=dict(
#         type="ConvSingleProj",
#         in_channels=1408,
#         out_channels=256,
#         num_convs=2,
#     ),
#     # 2. 时序边界头：源码只接收 in_channels 和 loss，其余多余参数已全部清除
#     rpn_head=dict(
#         type="TemporalEvaluationHead",
#         in_channels=256,
#         loss=dict(
#             gt_type=["startness", "endness"],
#             pos_thresh=0.5,
#         ),
#     ),
#     # 3. 候选框评估头
#     roi_head=dict(
#         type="StandardProposalMapHead",
#         proposal_generator=dict(
#             type="DenseProposalMap",
#             tscale=192,  # <--- 125 改为 192
#             dscale=192,  # <--- 125 改为 192
#         ),
#         proposal_roi_extractor=dict(
#             type="BMNExtractor",
#             in_channels=256,
#             out_channels=128,
#             roi_channels=512,
#             tscale=192,  # <--- 125 改为 192
#             dscale=192,  # <--- 125 改为 192
#             prop_extend_ratio=0.5,
#         ),
#         proposal_head=dict(
#             type="PEMHead",
#             in_channels=128,
#             feat_channels=128,
#             num_convs=2,
#             loss=dict(
#                 cls_loss=dict(type="BalancedBCELoss", pos_thresh=0.9),
#                 reg_loss=dict(
#                     type="BalancedL2Loss",
#                     weight=5.0,
#                     low_thresh=0.3,
#                     high_thresh=0.7,
#                 ),
#             ),
#         ),
#     ),
# )

model = dict(
    type="MambaBMN",
    use_global_mamba=True,
    global_mamba_cfg=dict(
        bidirectional=True,   # 双向扫描
        use_conv=True,
        shared_conv=False,     # 共享卷积
    ),
    tscale=192,
    dscale=192,
    prop_boundary_ratio=0.5,
    projection=dict(
        type="ConvSingleProj",
        in_channels=1408,
        out_channels=256,
        num_convs=2,

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
        # 加大loss
        loss=dict(
            cls_loss=dict(type="BalancedBCELoss", pos_thresh=0.9),
            reg_loss=dict(
                type="BalancedL2Loss",
                high_thresh=0.7,
                low_thresh=0.3,
                weight=10.0,  # 【修改这里】：默认是 5.0，改成 10.0 或 15.0
            ),
        ),
    ),
)



# model = dict(
#     # --- Detector 配置 ---
#     type="MambaBMN",  # 指向你新注册的 MambaBMN 类
#     use_global_mamba=True,  # 启用 Global Mamba
#     global_mamba_cfg=dict(
#         bidirectional=True,
#         use_conv=True,
#         shared_conv=False,  # 禁用共享卷积，采用独立局部卷积
#     ),
#     # ===================== 核心修改 1：时序尺度对齐 =====================
#     # 必须与你数据流 pipeline 中的 resize_length=192 严格保持一致！
#     # 否则立即报你刚才遇到的 RuntimeError: contracted dimensions need to match (192 vs 125)
#     tscale=192,
#     dscale=192,  # FineAction 跨度大，建议覆盖到 192
#     prop_boundary_ratio=0.5,
#     # ===================== 核心修改 2：特征维度对齐 =====================
#     # --- Projection 配置 ---
#     projection=dict(
#         type="ConvSingleProj",
#         in_channels=1408,  # 【关键】：VideoMAE-Giant 的特征维度是 1408，原先的 2048 是 TSN 的
#         out_channels=256,
#         num_convs=2,
#     ),
#     # --- RPN Head (TEM) 配置 ---
#     rpn_head=dict(
#         type="MambaBMNHead",  # 指向你的 MambaTEM 类
#         use_mamba=False,
#         in_channels=256,
#         num_classes=3,
#         loss=dict(
#             pos_thresh=0.5,
#             gt_type=[
#                 "startness",
#                 "endness",
#                 "actionness",
#             ],  # 显式声明需要 Actionness
#         ),
#         # Mamba 专属参数
#         mamba_cfg=dict(d_state=16, d_conv=4, expand=2),
#     ),
#     # --- RoI Head (PEM) 配置 ---
#     roi_head=dict(
#         type="MambaBMNRoIHead",  # 指向你的 BiMambaPEM 类
#         use_mamba=False,
#         in_channels=256,
#         feat_dim=128,  # 内部处理维度
#         # Mamba 专属参数
#         d_axis_mode="mamba",
#         mamba_cfg=dict(d_state=16, d_conv=4, expand=2),
#     ),
# )




solver = dict(
    train=dict(batch_size=16, num_workers=4),
    val=dict(batch_size=16, num_workers=4),
    test=dict(batch_size=16, num_workers=4),
    clip_grad_norm=1,
    ema=True,
)

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

work_dir = "exps/fineaction/ours_3_videomaev2_g"
