dataset_type = "ThumosSlidingDataset"
annotation_path = "/home/cipan/tennis/OpenTAD-main/data/thumos_14_anno.json"
class_map = "/home/cipan/tennis/OpenTAD-main/data/category_idx.txt"
data_path = "/home/cipan/tennis/OpenTAD-main/data/features"
block_list = None

# # 跑公共数据集
# dataset_type = "ThumosSlidingDataset"
# annotation_path = "/home/cipan/tennis/OpenTAD-main/data1/thumos/annotations/thumos_14_anno.json"
# class_map = "/home/cipan/tennis/OpenTAD-main/data1/thumos/annotations/category_idx.txt"
# data_path = "/home/cipan/tennis/OpenTAD-main/data1/thumos/tsn_gtad_stride1_thumos"
# block_list = "/home/cipan/tennis/OpenTAD-main/data1/thumos/tsn_gtad_stride1_thumos/missing_files.txt"

window_size = 125# 256

dataset = dict(
    train=dict(
        type=dataset_type,
        ann_file=annotation_path,
        subset_name="training",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        # thumos dataloader setting
        feature_stride=1,
        sample_stride=1,  # 1x4=4
        window_size=window_size,
        window_overlap_ratio=0.5,# 0.25
        offset_frames=0,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels"]),
            dict(type="SlidingWindowTrunc", with_mask=True),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    val=dict(
        type=dataset_type,
        ann_file=annotation_path,
        subset_name="testing",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        # thumos dataloader setting
        feature_stride=1,
        sample_stride=1,  # 1x4=4
        window_size=window_size,
        window_overlap_ratio=0.5,# 0.25
        offset_frames=0,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels"]),
            dict(type="SlidingWindowTrunc", with_mask=True),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    test=dict(
        type=dataset_type,
        ann_file=annotation_path,
        subset_name="testing",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        test_mode=True,
        # thumos dataloader setting
        feature_stride=1,
        sample_stride=1,  # 1x4=4
        window_size=window_size,
        window_overlap_ratio=0.5,
        offset_frames=0,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="ConvertToTensor", keys=["feats"]),
            dict(type="SlidingWindowTrunc", with_mask=True),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks"]),
        ],
    ),
)


evaluation = dict(
    type="mAP",
    subset="testing",
    tiou_thresholds=[0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    ground_truth_filename=annotation_path,
)

# evaluation = dict(
#     # === 关键修改：指定使用自定义评估器 ===
#     type="CustomBMNEvaluator",
#
#     # 保持原有的 GT 路径设置
#     ground_truth_filename=annotation_path,
#
#     # Thumos14 的测试集通常叫 "validation" (OpenTAD 数据集定义的习惯)
#     # 或者是 "test"，请根据你的 dataset 字典中的 test 下的 subset_name 保持一致
#     subset="testing",
#
#     # === 标准 mAP 参数 ===
#     tiou_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8],
#     recall_topk=[1, 10, 15, 20, 50, 100],
#
#     # # === 新指标专用参数 ===
#     # sl_thresholds=[0.1, 0.2, 0.3],  # SL-mAP 的容忍度
#     # f1_thresholds=[0.1, 0.3, 0.5],  # F1-Score 的 IoU 阈值  [0.3, 0.5]
#     # f1_topk=20,
#     # mof_fps=1.0  # Edit Score/MOF 采样率 (1.0 fps 计算较快)
# )

