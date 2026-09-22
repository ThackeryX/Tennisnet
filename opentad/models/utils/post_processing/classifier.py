import json
import numpy as np
import torch
from mmengine.registry import Registry

CLASSIFIERS = Registry("models")


def build_classifier(cfg):
    """Build external classifier."""
    return CLASSIFIERS.build(cfg)


@CLASSIFIERS.register_module()
class CUHKANETClassifier:
    def __init__(self, path, topk=1):
        super().__init__()

        with open(path, "r") as f:
            cuhk_data = json.load(f)
        self.cuhk_data_score = cuhk_data["results"]
        self.cuhk_data_action = np.array(cuhk_data["class"])
        self.topk = topk

    def __call__(self, video_id, segments, scores):
        assert len(segments) == len(scores)

        # sort video classification
        cuhk_score = np.array(self.cuhk_data_score[video_id])
        cuhk_classes = self.cuhk_data_action[np.argsort(-cuhk_score)]
        cuhk_score = cuhk_score[np.argsort(-cuhk_score)]

        new_segments = []
        new_labels = []
        new_scores = []
        # for segment, score in zip(segments, scores):
        for k in range(self.topk):
            new_segments.append(segments)
            new_labels.extend([cuhk_classes[k]] * len(segments))
            new_scores.append(scores * cuhk_score[k])

        new_segments = torch.cat(new_segments)
        new_scores = torch.cat(new_scores)
        return new_segments, new_labels, new_scores


@CLASSIFIERS.register_module()
class UntrimmedNetTHUMOSClassifier:
    def __init__(self, path, topk=1):
        super().__init__()

        # =========================================================
        # 修改 1: 仅保留您的自定义类别
        # =========================================================
        self.classes = ['网球挥拍']

        # 加载外部使得分数文件 (.npy)
        # 如果您没有生成过这个文件，可以在下面的 __call__ 中做容错处理
        try:
            self.cls_data = np.load(path, allow_pickle=True).item()
        except:
            # 如果加载失败或路径不对，初始化为空字典，后面会用默认分兜底
            print(f"[Warning] Cannot load classifier scores from {path}. Using default score 1.0.")
            self.cls_data = {}

        self.topk = topk

    def __call__(self, video_id, segments, scores):
        assert len(segments) == len(scores)

        # =========================================================
        # 修改 2: 使用字典查询代替 int(video_id[-4:])
        # =========================================================
        if video_id in self.cls_data:
            # 获取该视频的分类分数
            video_cls_scores = np.array(self.cls_data[video_id])
        else:
            # 兜底策略：如果字典里没找到这个视频（或者没有分类文件）
            # 给所有类别 1.0 的分数，这样 BMN 预测出的分数不会被降低
            video_cls_scores = np.ones(len(self.classes))

        # =========================================================
        # 修改 3: 简化的 Top-K 逻辑
        # =========================================================
        # 如果是单类别，其实不需要排序，直接取第0个即可
        # 为了兼容性，这里还是写通用的逻辑

        # 确保 video_cls_scores 长度和 self.classes 一致
        if len(video_cls_scores) != len(self.classes):
            # 如果形状不匹配（比如npy是20类，你现在是1类），强制取第一个或重置
            video_cls_scores = np.ones(len(self.classes))

        sorted_indices = np.argsort(video_cls_scores)[::-1]

        new_segments = []
        new_labels = []
        new_scores = []

        # 遍历前 TopK 个类别 (通常为 1)
        for k in range(self.topk):
            if k >= len(self.classes):
                break

            class_idx = sorted_indices[k]
            class_name = self.classes[class_idx]
            class_score = video_cls_scores[class_idx]

            # 复制片段坐标
            new_segments.append(segments)
            # 赋予类别名称
            new_labels.extend([class_name] * len(segments))
            # 最终得分 = BMN定位得分 * 视频全局分类得分
            # 这里的计算方式取决于 OpenTAD 版本，有些是乘法，有些是开方
            # 原版代码里有 new_scores.append(scores * unet_scores[k])
            new_scores.append(scores * class_score)

        if len(new_segments) > 0:
            new_segments = torch.cat(new_segments)
            new_scores = torch.cat(new_scores)
        else:
            # 防止空列表报错
            new_segments = segments
            new_scores = scores
            new_labels = [self.classes[0]] * len(segments)

        return new_segments, new_labels, new_scores

# class UntrimmedNetTHUMOSClassifier:
#     def __init__(self, path, topk=1):
#         super().__init__()
#
#         self.thumos_class = {
#             7: "BaseballPitch",
#             9: "BasketballDunk",
#             12: "Billiards",
#             21: "CleanAndJerk",
#             22: "CliffDiving",
#             23: "CricketBowling",
#             24: "CricketShot",
#             26: "Diving",
#             31: "FrisbeeCatch",
#             33: "GolfSwing",
#             36: "HammerThrow",
#             40: "HighJump",
#             45: "JavelinThrow",
#             51: "LongJump",
#             68: "PoleVault",
#             79: "Shotput",
#             85: "SoccerPenalty",
#             92: "TennisSwing",
#             93: "ThrowDiscus",
#             97: "VolleyballSpiking",
#         }
#
#         # self.thumos_class = {"网球挥拍"}
#
#         # 必须加上 allow_pickle=True 才能读取字典
#         # 必须加上 .item() 才能把读取到的 0-d array 转回为 python 字典
#         self.cls_data = np.load(path, allow_pickle=True).item()
#
#         self.thu_label_id = np.array(list(self.thumos_class.keys())) - 1  # get thumos class id
#         self.topk = topk
#
#     def __call__(self, video_id, segments, scores):
#         assert len(segments) == len(scores)
#
#         # sort video classification
#         video_cls = self.cls_data[int(video_id[-4:]) - 1][self.thu_label_id]  # order by video list, output 20
#         # if video_id in self.cls_data:
#         #     video_cls = self.cls_data[video_id]
#         # else:
#         #     # 兜底策略：如果字典里没找到这个视频，给个全0或者是默认值
#         #     # 这里假设你的类别数是 self.cls_data 中任意一个值的长度
#         #     print(f"!!! 严重警告: 字典里找不到视频 {video_id} !!!")
#         #     # 打印字典里前3个key，看看长什么样，对比一下差异
#         #     print(f"    字典里的Key示例: {list(self.cls_data.keys())[:3]}")
#         #     first_key = list(self.cls_data.keys())[0]
#         #     num_classes = len(self.cls_data[first_key])
#         #     video_cls = np.zeros(num_classes)
#
#         video_cls_rank = sorted((e, i) for i, e in enumerate(video_cls))
#         unet_classes = [self.thu_label_id[video_cls_rank[-k - 1][1]] + 1 for k in range(self.topk)]
#         unet_scores = [video_cls_rank[-k - 1][0] for k in range(self.topk)]
#
#         new_segments = []
#         new_labels = []
#         new_scores = []
#         # for segment, score in zip(segments, scores):
#         for k in range(self.topk):
#             new_segments.append(segments)
#             new_labels.extend([self.thumos_class[int(unet_classes[k])]] * len(segments))
#             new_scores.append(scores * unet_scores[k])
#
#         new_segments = torch.cat(new_segments)
#         new_scores = torch.cat(new_scores)
#         return new_segments, new_labels, new_scores


@CLASSIFIERS.register_module()
class TCANetHACSClassifier:
    def __init__(self, path, topk=1):
        super().__init__()

        with open(path, "r") as f:
            cls_data = json.load(f)
        self.cls_data_score = cls_data["results"]
        self.cls_data_action = cls_data["class"]
        self.topk = topk

    def __call__(self, video_id, segments, scores):
        assert len(segments) == len(scores)

        # sort video classification
        cls_score = np.array(self.cls_data_score[video_id][0])
        cls_score = np.exp(cls_score) / np.sum(np.exp(cls_score)) * 2.0
        cls_data_action = np.array(self.cls_data_action)
        cls_classes = cls_data_action[np.argsort(-cls_score)]
        cls_score = cls_score[np.argsort(-cls_score)]

        new_segments = []
        new_labels = []
        new_scores = []

        for k in range(self.topk):
            new_segments.append(segments)
            new_labels.extend([cls_classes[k]] * len(segments))
            new_scores.append(scores * cls_score[k])

        new_segments = torch.cat(new_segments)
        new_scores = torch.cat(new_scores)
        return new_segments, new_labels, new_scores


@CLASSIFIERS.register_module()
class StandardClassifier:
    def __init__(self, path, topk=1, apply_softmax=False):
        super().__init__()

        with open(path, "r") as f:
            cls_data = json.load(f)
        self.cls_data_score = cls_data["results"]
        self.cls_data_label = np.array(cls_data["class"]) if "class" in cls_data else np.array(cls_data["classes"])
        self.apply_softmax = apply_softmax
        self.topk = topk

    def __call__(self, video_id, segments, scores):
        assert len(segments) == len(scores)
        cls_score = np.array(self.cls_data_score[video_id])

        if self.apply_softmax:  # do softmax
            cls_score = np.exp(cls_score) / np.sum(np.exp(cls_score))

        # sort video classification scores
        topk_cls_idx = np.argsort(cls_score)[::-1][: self.topk]
        topk_cls_score = cls_score[topk_cls_idx]
        topk_cls_label = self.cls_data_label[topk_cls_idx]

        new_segments = []
        new_labels = []
        new_scores = []

        for k in range(self.topk):
            new_segments.append(segments)
            new_labels.extend([topk_cls_label[k]] * len(segments))
            new_scores.append(np.sqrt(scores * topk_cls_score[k]))  # default is sqrt

        new_segments = torch.cat(new_segments)
        new_scores = torch.cat(new_scores)
        return new_segments, new_labels, new_scores


@CLASSIFIERS.register_module()
class PseudoClassifier:
    def __init__(self, pseudo_label=""):
        super().__init__()

        self.pseudo_label = pseudo_label

    def __call__(self, video_id, segments, scores):
        assert len(segments) == len(scores)

        labels = [self.pseudo_label for _ in range(len(segments))]

        return segments, labels, scores


