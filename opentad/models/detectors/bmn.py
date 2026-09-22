import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import DETECTORS
from .two_stage import TwoStageDetector
from ..utils.post_processing import boundary_choose, batched_nms, convert_to_seconds



@DETECTORS.register_module()
class BMN(TwoStageDetector):
    def __init__(
        self,
        projection,
        rpn_head,
        roi_head,
        neck=None,
        backbone=None,
    ):
        super(BMN, self).__init__(
            backbone=backbone,
            neck=neck,
            projection=projection,
            rpn_head=rpn_head,
            roi_head=roi_head,
        )

        self.reset_params()

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward_test(self, inputs, masks, metas=None, infer_cfg=None):
        if self.with_backbone:
            x = self.backbone(inputs, masks)
        else:
            x = inputs

        if self.with_projection:
            x, masks = self.projection(x, masks)

        if self.with_neck:
            x, masks = self.neck(x, masks)

        # tem score
        tem_score = self.rpn_head.forward_test(x, masks)  # [B,2,T]

        # iou score
        pred_iou_map = self.roi_head.forward_test(x)  # [D,T,2]

        # pack all
        predictions = tem_score, pred_iou_map
        return predictions

    def post_processing(self, predictions, metas, post_cfg, ext_cls, **kwargs):
        assert ext_cls != None  #  BMN is a proposal generation network

        score_type = getattr(post_cfg, "score_type", "iou")  # otherwise, "iou*s*e"
        proposal_post = getattr(post_cfg, "proposal", False)

        tem_score, pred_iou_map = predictions
        tem_score = tem_score.sigmoid()
        pred_iou_map = pred_iou_map.sigmoid()

        dscale = self.roi_head.proposal_generator.dscale
        tscale = self.roi_head.proposal_generator.tscale

        # here is a nice matrix implementation for post processing
        ds = torch.arange(0, dscale).to(pred_iou_map.device)
        ts = torch.arange(0, tscale).to(pred_iou_map.device)
        ds_mesh, ts_mesh = torch.meshgrid(ds, ts, indexing="ij")
        start_end_index = torch.stack((ts_mesh, ts_mesh + ds_mesh), dim=-1)
        valid_mask = start_end_index[:, :, 1] < tscale
        start_end_index = start_end_index.clamp(max=tscale - 1).float()

        start_mask = boundary_choose(tem_score[:, 0, :])
        start_mask[:, 0] = True
        end_mask = boundary_choose(tem_score[:, 1, :])
        end_mask[:, -1] = True
        start_end_map = start_mask.unsqueeze(2) * end_mask.unsqueeze(1)
        pred_iou_map = pred_iou_map[:, 0, :, :] * pred_iou_map[:, 1, :, :]

        results = {}
        for i in range(len(metas)):  # processing each video
            start_end_mask = start_end_map[i][
                start_end_index[:, :, 0].view(-1).long(),
                start_end_index[:, :, 1].view(-1).long(),
            ]
            start_end_mask = start_end_mask.reshape(dscale, tscale) * valid_mask

            segments_start = start_end_index[start_end_mask][:, 0]
            segments_end = start_end_index[start_end_mask][:, 1] + 1
            segments = torch.stack((segments_start, segments_end), dim=-1).detach().cpu()

            # score
            scores_iou = pred_iou_map[i][start_end_mask].detach().cpu()
            if score_type == "iou":
                scores = scores_iou
            elif score_type == "iou*s*e":
                score_start = tem_score[i, 0, start_end_index[start_end_mask][:, 0].long()].detach().cpu()
                score_end = tem_score[i, 1, start_end_index[start_end_mask][:, 1].long()].detach().cpu()
                scores = score_start * score_end * scores_iou
            else:
                raise f"score type should be iou or iou*s*e, but get {score_type}"

            # nms
            labels = torch.zeros_like(scores_iou)  # pseudo labels
            if proposal_post:
                segments = segments / tscale  # convert to 0~1, since recall need a different NMS
                segments, scores, labels = batched_nms(segments, scores, labels, **post_cfg.nms)
                segments = segments * tscale  # convert back
            else:
                segments, scores, labels = batched_nms(segments, scores, labels, **post_cfg.nms)

            video_id = metas[i]["video_name"]

            # convert segments to seconds
            segments = convert_to_seconds(segments, metas[i])

            # merge with external classifier
            segments, labels, scores = ext_cls(video_id, segments, scores)

            results_per_video = []
            for segment, label, score in zip(segments, labels, scores):
                # convert to python scalars
                results_per_video.append(
                    dict(
                        segment=[round(seg.item(), 2) for seg in segment],
                        label=label,
                        score=round(score.item(), 4),
                    )
                )

            if video_id in results.keys():
                results[video_id].extend(results_per_video)
            else:
                results[video_id] = results_per_video

        return results




    def get_optim_groups(self, cfg):
        BASE_weight = []
        TEM_weight = []
        PEM_weight = []

        for name, p in self.named_parameters():
            # exclude the backbone
            if name.startswith("backbone"):
                continue

            if "projection" in name:
                BASE_weight.append(p)
            elif "rpn_head" in name:
                TEM_weight.append(p)
            elif "roi_head" in name:
                PEM_weight.append(p)
            else:
                print(name, "is not in the optimizer")

        # create the pytorch optimizer object
        optim_groups = [
            {"params": BASE_weight, "weight_decay": cfg["weight_decay"] * 10},
            {"params": TEM_weight, "weight_decay": cfg["weight_decay"]},
            {"params": PEM_weight, "weight_decay": cfg["weight_decay"]},
        ]
        return optim_groups
#
#
# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# from ..builder import DETECTORS
# from .two_stage import TwoStageDetector
# from ..utils.post_processing import boundary_choose, batched_nms, convert_to_seconds
# from ..bricks import Mamba
#
#
# @DETECTORS.register_module()
# class BMN(TwoStageDetector):
#     def __init__(
#             self,
#             projection,
#             rpn_head,
#             roi_head,
#             neck=None,
#             backbone=None,
#             use_mamba=True,
#             mamba_cfg=None,
#     ):
#         super(BMN, self).__init__(
#             backbone=backbone,
#             neck=neck,
#             projection=projection,
#             rpn_head=rpn_head,
#             roi_head=roi_head,
#         )
#
#         # === 1. 初始化 Mamba ===
#         self.use_mamba = use_mamba
#         if self.use_mamba:
#             if mamba_cfg is None:
#                 mamba_cfg = dict(d_model=256, d_state=16, d_conv=4, expand=2)
#
#             self.mamba = Mamba(**mamba_cfg)
#             self.mamba_norm = nn.LayerNorm(mamba_cfg['d_model'])
#
#             # === 2. 零初始化 ===
#             if hasattr(self.mamba, 'out_proj'):
#                 nn.init.constant_(self.mamba.out_proj.weight, 0)
#                 if self.mamba.out_proj.bias is not None:
#                     nn.init.constant_(self.mamba.out_proj.bias, 0)
#
#         self.reset_params()
#
#     def reset_params(self):
#         for name, m in self.named_modules():
#             if "mamba" in name:
#                 continue
#             if isinstance(m, (nn.Conv1d, nn.Conv2d)):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#
#     def _forward_mamba(self, x):
#         if not self.use_mamba:
#             return x
#
#         x_in = x.permute(0, 2, 1)
#         x_norm = self.mamba_norm(x_in)
#         x_out = self.mamba(x_norm)
#         x = x + x_out.permute(0, 2, 1)
#         return x
#
#     def forward_train(self, inputs, masks, metas, gt_segments, gt_labels, **kwargs):
#         if self.with_backbone:
#             x = self.backbone(inputs, masks)
#         else:
#             x = inputs
#
#         if self.with_projection:
#             x, masks = self.projection(x, masks)
#
#         if self.with_neck:
#             x, masks = self.neck(x, masks)
#
#         # === 3. 插入 Mamba ===
#         x = self._forward_mamba(x)
#
#         losses = dict()
#
#         # === 4. RPN Head (TEM) ===
#         head_output = self.rpn_head.forward_train(x, masks, gt_segments)
#         if isinstance(head_output, tuple):
#             loss_tem = head_output[0]
#         else:
#             loss_tem = head_output
#         losses.update(loss_tem)
#
#         # === 5. RoI Head (PEM) ===
#         roi_output = self.roi_head.forward_train(x, masks, gt_segments, gt_labels)
#         if isinstance(roi_output, tuple):
#             loss_pem = roi_output[0]
#         else:
#             loss_pem = roi_output
#         losses.update(loss_pem)
#
#         # === [关键修复] 6. 计算总损失 'cost' ===
#         cost = torch.tensor(0.0, device=x.device)
#         for val in losses.values():
#             if isinstance(val, torch.Tensor):
#                 cost = cost + val
#         losses['cost'] = cost  # 必须包含这一项，train_engine 才能反向传播
#
#         return losses
#
#     def forward_test(self, inputs, masks, metas=None, infer_cfg=None):
#         if self.with_backbone:
#             x = self.backbone(inputs, masks)
#         else:
#             x = inputs
#
#         if self.with_projection:
#             x, masks = self.projection(x, masks)
#
#         if self.with_neck:
#             x, masks = self.neck(x, masks)
#
#         # === 插入 Mamba (Test) ===
#         x = self._forward_mamba(x)
#
#         tem_score = self.rpn_head.forward_test(x, masks)
#         pred_iou_map = self.roi_head.forward_test(x)
#
#         return tem_score, pred_iou_map
#
#     def post_processing(self, predictions, metas, post_cfg, ext_cls, **kwargs):
#         assert ext_cls != None
#
#         score_type = getattr(post_cfg, "score_type", "iou")
#         proposal_post = getattr(post_cfg, "proposal", False)
#
#         tem_score, pred_iou_map = predictions
#         tem_score = tem_score.sigmoid()
#         pred_iou_map = pred_iou_map.sigmoid()
#
#         dscale = self.roi_head.proposal_generator.dscale
#         tscale = self.roi_head.proposal_generator.tscale
#
#         ds = torch.arange(0, dscale).to(pred_iou_map.device)
#         ts = torch.arange(0, tscale).to(pred_iou_map.device)
#         ds_mesh, ts_mesh = torch.meshgrid(ds, ts, indexing="ij")
#         start_end_index = torch.stack((ts_mesh, ts_mesh + ds_mesh), dim=-1)
#         valid_mask = start_end_index[:, :, 1] < tscale
#         start_end_index = start_end_index.clamp(max=tscale - 1).float()
#
#         start_mask = boundary_choose(tem_score[:, 0, :])
#         start_mask[:, 0] = True
#         end_mask = boundary_choose(tem_score[:, 1, :])
#         end_mask[:, -1] = True
#         start_end_map = start_mask.unsqueeze(2) * end_mask.unsqueeze(1)
#         pred_iou_map = pred_iou_map[:, 0, :, :] * pred_iou_map[:, 1, :, :]
#
#         results = {}
#         for i in range(len(metas)):
#             start_end_mask = start_end_map[i][
#                 start_end_index[:, :, 0].view(-1).long(),
#                 start_end_index[:, :, 1].view(-1).long(),
#             ]
#             start_end_mask = start_end_mask.reshape(dscale, tscale) * valid_mask
#
#             segments_start = start_end_index[start_end_mask][:, 0]
#             segments_end = start_end_index[start_end_mask][:, 1] + 1
#             segments = torch.stack((segments_start, segments_end), dim=-1).detach().cpu()
#
#             scores_iou = pred_iou_map[i][start_end_mask].detach().cpu()
#             if score_type == "iou":
#                 scores = scores_iou
#             elif score_type == "iou*s*e":
#                 score_start = tem_score[i, 0, start_end_index[start_end_mask][:, 0].long()].detach().cpu()
#                 score_end = tem_score[i, 1, start_end_index[start_end_mask][:, 1].long()].detach().cpu()
#                 scores = score_start * score_end * scores_iou
#             else:
#                 raise f"score type should be iou or iou*s*e, but get {score_type}"
#
#             labels = torch.zeros_like(scores_iou)
#             if proposal_post:
#                 segments = segments / tscale
#                 segments, scores, labels = batched_nms(segments, scores, labels, **post_cfg.nms)
#                 segments = segments * tscale
#             else:
#                 segments, scores, labels = batched_nms(segments, scores, labels, **post_cfg.nms)
#
#             video_id = metas[i]["video_name"]
#             segments = convert_to_seconds(segments, metas[i])
#             segments, labels, scores = ext_cls(video_id, segments, scores)
#
#             results_per_video = []
#             for segment, label, score in zip(segments, labels, scores):
#                 results_per_video.append(
#                     dict(
#                         segment=[round(seg.item(), 2) for seg in segment],
#                         label=label,
#                         score=round(score.item(), 4),
#                     )
#                 )
#
#             if video_id in results.keys():
#                 results[video_id].extend(results_per_video)
#             else:
#                 results[video_id] = results_per_video
#
#         return results
#
#     def get_optim_groups(self, cfg):
#         BASE_weight = []
#         TEM_weight = []
#         PEM_weight = []
#         MAMBA_weight = []
#
#         for name, p in self.named_parameters():
#             if not p.requires_grad:
#                 continue
#             if name.startswith("backbone"):
#                 continue
#
#             if "projection" in name:
#                 BASE_weight.append(p)
#             elif "rpn_head" in name:
#                 TEM_weight.append(p)
#             elif "roi_head" in name:
#                 PEM_weight.append(p)
#             elif "mamba" in name:
#                 MAMBA_weight.append(p)
#             else:
#                 BASE_weight.append(p)
#
#         optim_groups = [
#             {"params": BASE_weight, "weight_decay": cfg["weight_decay"] * 10},
#             {"params": TEM_weight, "weight_decay": cfg["weight_decay"]},
#             {"params": PEM_weight, "weight_decay": cfg["weight_decay"]},
#             {"params": MAMBA_weight, "weight_decay": cfg["weight_decay"]},
#         ]
#         return optim_groups
