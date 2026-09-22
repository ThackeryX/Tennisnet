_base_ = [
    "../_base_/datasets/thumos-14/features_i3d_sw.py",
    "../_base_/models/tadtr.py",
]

# dataset = dict(
#     train=dict(window_size=125, window_overlap_ratio=0.25),
#     val=dict(window_size=125, window_overlap_ratio=0.25),
#     test=dict(window_size=125, window_overlap_ratio=0.75),
# )
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


solver = dict(
    train=dict(batch_size=16, num_workers=4),
    val=dict(batch_size=16, num_workers=4),
    test=dict(batch_size=16, num_workers=4),
    clip_grad_norm=0.1,
)

# optimizer = dict(type="AdamW", lr=2e-4, weight_decay=1e-4, paramwise=True)
# scheduler = dict(type="MultiStepLR", milestones=[20], gamma=0.1, max_epoch=50)
#
# inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
# post_processing = dict(
#     nms=dict(
#         use_soft_nms=True,
#         sigma=0.4,
#         max_seg_num=2000,
#         multiclass=False,
#         voting_thresh=0.95,  #  set 0 to disable
#     ),
#     save_dict=False,
# )
#
# workflow = dict(
#     logging_interval=300,
#     checkpoint_interval=1,
#     val_loss_interval=-1,
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
work_dir = "exps/oursdata/tadtr_i3d_2"
