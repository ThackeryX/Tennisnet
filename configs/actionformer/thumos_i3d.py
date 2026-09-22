_base_ = [
    "../_base_/datasets/thumos-14/features_i3d_pad.py",  # dataset config
    "../_base_/models/actionformer.py",  # model config
]

model = dict(
    projection=dict(in_channels=2048, input_pdrop=0.2),
    rpn_head=dict(
        num_classes=1,
        # num_classes=20,#公共数据集20类别
    )
)



solver = dict(
    train=dict(batch_size=16, num_workers=4),
    val=dict(batch_size=1, num_workers=1),
    test=dict(batch_size=1, num_workers=1),
    clip_grad_norm=1,
    ema=True,
)

# optimizer = dict(type="AdamW", lr=1e-4, weight_decay=0.05, paramwise=True)
# scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=20, max_epoch=50)
#
# inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
# post_processing = dict(
#     nms=dict(
#         use_soft_nms=True,
#         sigma=0.5,
#         max_seg_num=2000,
#         iou_threshold=0.1,  # does not matter when use soft nms
#         min_score=0.001,
#         multiclass=False,
#         voting_thresh=0.7,  #  set 0 to disable
#     ),
#     save_dict=True,
# )
#
# workflow = dict(
#     logging_interval=20,
#     checkpoint_interval=1,
#     val_loss_interval=1,
#     val_eval_interval=1,
#     val_start_epoch=0,
# )



optimizer = dict(type="Adam", lr=1e-4, weight_decay=1e-4, paramwise=True)
scheduler = dict(type="MultiStepLR", milestones=[20], gamma=0.1, max_epoch=50)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    nms=dict(
        use_soft_nms=True,
        sigma=0.3,
        max_seg_num=200,
        min_score=0.0001,
        multiclass=False,
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
work_dir = "exps/oursdata/actionformer_i3d_2"
