import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import DETECTORS
from .two_stage import TwoStageDetector
from ..bricks import Mamba
from ..utils.post_processing import boundary_choose, batched_nms, convert_to_seconds


@DETECTORS.register_module()
class MambaBMN(TwoStageDetector):
    def __init__(
        self,
        projection,
        rpn_head,
        roi_head,
        backbone=None,
        neck=None,
        global_mamba_cfg=None,
        tscale=128,
        dscale=128,
        prop_boundary_ratio=0.5,
        **kwargs
    ):
        super().__init__(
            backbone=backbone,
            projection=projection,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            **kwargs
        )

        self.tscale = tscale
        self.dscale = dscale
        self.prop_boundary_ratio = prop_boundary_ratio

        # Global sequence modeling via Mamba
        if global_mamba_cfg is None:
            global_mamba_cfg = dict(bidirectional=True)
        self.global_mamba = Mamba(d_model=256, **global_mamba_cfg)
        self.global_norm = nn.LayerNorm(256)

        # Precompute proposal boundary map
        self.proposal_map, self.valid_mask = self._get_proposal_map(tscale, dscale)
        self.reset_params()

    def reset_params(self):
        for name, m in self.named_modules():
            if "mamba" in name:
                continue
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _get_proposal_map(self, tscale, dscale):
        ds = torch.arange(0, dscale)
        ts = torch.arange(0, tscale)
        ds_mesh, ts_mesh = torch.meshgrid(ds, ts, indexing="ij")
        start_end_index = torch.stack((ts_mesh, ts_mesh + ds_mesh), dim=-1)
        valid_mask = start_end_index[:, :, 1] < tscale
        return start_end_index.float(), valid_mask

    def _apply_global_mamba(self, x):
        x_in = x.permute(0, 2, 1)
        x_out = self.global_norm(self.global_mamba(x_in))
        return x_out.permute(0, 2, 1) + x

    def forward_train(self, inputs, masks, metas, gt_segments, gt_labels, **kwargs):
        x = self.backbone(inputs, masks) if self.with_backbone else inputs

        if self.with_projection:
            x, masks = self.projection(x, masks)

        if self.with_neck:
            x, masks = self.neck(x, masks)

        x = F.interpolate(x, size=self.tscale, mode="linear", align_corners=False)
        x = self._apply_global_mamba(x)

        losses = dict()

        # TEM Loss
        loss_tem = self.rpn_head.forward_train(x, masks, gt_segments)
        losses.update(loss_tem)

        # PEM Loss
        x_2d = self._boundary_matching_layer(x)
        prop_map = self.proposal_map.to(x.device)
        v_mask = self.valid_mask.to(x.device)
        loss_pem = self.roi_head.forward_train(x_2d, prop_map, v_mask, gt_segments)
        losses.update(loss_pem)

        cost = torch.tensor(0.0, device=x.device)
        for val in losses.values():
            if isinstance(val, torch.Tensor):
                cost = cost + val
        losses["cost"] = cost
        return losses

    def forward_test(self, inputs, masks, metas, infer_cfg=None, **kwargs):
        x = self.backbone(inputs, masks) if self.with_backbone else inputs

        if self.with_projection:
            x, masks = self.projection(x, masks)

        if self.with_neck:
            x, masks = self.neck(x, masks)

        x = F.interpolate(x, size=self.tscale, mode="linear", align_corners=False)
        x = self._apply_global_mamba(x)

        tem_output = self.rpn_head.forward_test(x, masks)
        x_2d = self._boundary_matching_layer(x)
        pem_output = self.roi_head.forward_test(x_2d, masks)

        return tem_output, pem_output

    def post_processing(self, predictions, metas, post_cfg, ext_cls, **kwargs):
        assert ext_cls is not None, "External classifier must be provided."

        score_type = getattr(post_cfg, "score_type", "iou")
        proposal_post = getattr(post_cfg, "proposal", False)

        tem_score, pred_iou_map = predictions
        tem_score = tem_score.sigmoid()
        pred_iou_map = pred_iou_map.sigmoid()

        dscale = self.dscale
        tscale = self.tscale

        ds = torch.arange(0, dscale, device=pred_iou_map.device)
        ts = torch.arange(0, tscale, device=pred_iou_map.device)
        ds_mesh, ts_mesh = torch.meshgrid(ds, ts, indexing="ij")

        start_end_index = torch.stack((ts_mesh, ts_mesh + ds_mesh), dim=-1)
        valid_mask = start_end_index[:, :, 1] < tscale
        start_end_index = start_end_index.clamp(max=tscale - 1).float()

        # Generate boundary masks
        start_mask = boundary_choose(tem_score[:, 0, :])
        start_mask[:, 0] = True
        end_mask = boundary_choose(tem_score[:, 1, :])
        end_mask[:, -1] = True

        start_end_map = start_mask.unsqueeze(2) * end_mask.unsqueeze(1)
        pred_iou_map = pred_iou_map[:, 0, :, :] * pred_iou_map[:, 1, :, :]

        results = {}
        for i in range(len(metas)):
            start_end_mask = start_end_map[i][
                start_end_index[:, :, 0].view(-1).long(),
                start_end_index[:, :, 1].view(-1).long(),
            ]
            start_end_mask = start_end_mask.reshape(dscale, tscale) * valid_mask

            segments_start = start_end_index[start_end_mask][:, 0]
            segments_end = start_end_index[start_end_mask][:, 1] + 1
            segments = torch.stack((segments_start, segments_end), dim=-1).detach().cpu()

            scores_iou = pred_iou_map[i][start_end_mask].detach().cpu()

            if score_type == "iou":
                scores = scores_iou
            elif score_type == "iou*s*e":
                score_start = tem_score[i, 0, start_end_index[start_end_mask][:, 0].long()].detach().cpu()
                score_end = tem_score[i, 1, start_end_index[start_end_mask][:, 1].long()].detach().cpu()
                scores = score_start * score_end * scores_iou
            else:
                raise ValueError(f"Unknown score type: {score_type}")

            labels = torch.zeros_like(scores_iou)

            if proposal_post:
                segments = segments / tscale
                segments, scores, labels = batched_nms(segments, scores, labels, **post_cfg.nms)
                segments = segments * tscale
            else:
                segments, scores, labels = batched_nms(segments, scores, labels, **post_cfg.nms)

            video_id = metas[i]["video_name"]
            segments = convert_to_seconds(segments, metas[i])
            segments, labels, scores = ext_cls(video_id, segments, scores)

            results_per_video = []
            for segment, label, score in zip(segments, labels, scores):
                results_per_video.append(
                    dict(
                        segment=[round(seg.item(), 2) for seg in segment],
                        label=label,
                        score=round(score.item(), 4),
                    )
                )

            if video_id in results:
                results[video_id].extend(results_per_video)
            else:
                results[video_id] = results_per_video

        return results

    def _boundary_matching_layer(self, x):
        B, C, T = x.shape
        return x.unsqueeze(2).expand(B, C, self.dscale, T)

    def get_optim_groups(self, cfg):
        base_params = []
        tem_params = []
        pem_params = []
        mamba_params = []

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("backbone"):
                continue

            if "projection" in name:
                base_params.append(p)
            elif "rpn_head" in name:
                tem_params.append(p)
            elif "roi_head" in name:
                pem_params.append(p)
            elif "mamba" in name:
                mamba_params.append(p)
            else:
                base_params.append(p)

        optim_groups = [
            {"params": base_params, "weight_decay": cfg["weight_decay"] * 10},
            {"params": tem_params, "weight_decay": cfg["weight_decay"]},
            {"params": pem_params, "weight_decay": cfg["weight_decay"]},
            {"params": mamba_params, "weight_decay": cfg["weight_decay"]},
        ]
        return optim_groups
