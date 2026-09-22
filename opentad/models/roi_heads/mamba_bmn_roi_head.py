import torch
import torch.nn as nn
from ..bricks import Mamba
from ..builder import HEADS, build_loss
from ..utils.iou_tools import compute_iou_torch


@HEADS.register_module()
class MambaBMNRoIHead(nn.Module):
    """
    Proposal Evaluation Module (PEM) Head with 2D Dual-axis Mamba (Time & Duration).
    """
    def __init__(self, in_channels=256, feat_dim=128, mamba_cfg=None, loss=None):
        super().__init__()
        if mamba_cfg is None:
            mamba_cfg = dict(d_state=16, d_conv=4, expand=2)

        if loss is None:
            loss = dict(
                cls_loss=dict(type="BalancedBCELoss", pos_thresh=0.9),
                reg_loss=dict(type="BalancedL2Loss", high_thresh=0.7, low_thresh=0.3),
            )

        self.in_channels = in_channels
        self.reduce_conv = nn.Conv2d(in_channels, feat_dim, kernel_size=1)

        # Dual-branch Mamba modules
        self.mamba_t = Mamba(d_model=feat_dim, **mamba_cfg)
        self.mamba_d = Mamba(d_model=feat_dim, **mamba_cfg)

        self.fusion = nn.Conv2d(feat_dim * 2, feat_dim, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)

        # Output predictions: IoU confidence score and regression confidence score
        self.out_conv = nn.Conv2d(feat_dim, 2, kernel_size=1)

        self.cls_loss = build_loss(loss["cls_loss"])
        self.reg_loss = build_loss(loss["reg_loss"])

    def forward(self, x):
        # x: Boundary Matching Feature Map [B, C, D, T]
        B, C, D, T = x.shape
        x_feat = self.act(self.reduce_conv(x))  # [B, feat_dim, D, T]
        C_curr = x_feat.shape[1]

        # Branch 1: Temporal Axis Scanning
        x_t = x_feat.permute(0, 2, 3, 1).reshape(B * D, T, C_curr)
        x_t = self.mamba_t(x_t).reshape(B, D, T, C_curr).permute(0, 3, 1, 2)

        # Branch 2: Duration Axis Scanning
        x_d = x_feat.permute(0, 3, 2, 1).reshape(B * T, D, C_curr)
        x_d = self.mamba_d(x_d).reshape(B, T, D, C_curr).permute(0, 3, 2, 1)

        # Feature Fusion
        x_concat = torch.cat([x_t, x_d], dim=1)
        x_fused = self.act(self.fusion(x_concat))

        return self.out_conv(x_fused)  # [B, 2, D, T]

    def forward_train(self, x, proposal_map, valid_mask, gt_segments, **kwargs):
        pred = self.forward(x)
        loss = self.losses(pred, proposal_map, valid_mask, gt_segments)
        return loss

    def forward_test(self, x, masks=None):
        return self.forward(x)

    @torch.no_grad()
    def prepare_targets(self, proposals, gt_segments):
        proposals = proposals.to(gt_segments[0].device)
        gt_ious = []
        for gt_segment in gt_segments:
            gt_iou = compute_iou_torch(gt_segment, proposals)
            gt_iou = torch.max(gt_iou, dim=1)[0]
            gt_ious.append(gt_iou)
        return torch.stack(gt_ious)

    def losses(self, pred, proposal_map, valid_mask, gt_segments):
        pred_valid = pred[:, :, valid_mask].sigmoid()
        proposals_valid = proposal_map[valid_mask, :]
        gt_ious = self.prepare_targets(proposals_valid, gt_segments)

        loss_cls = self.cls_loss(pred_valid[:, 0, :], gt_ious)
        loss_reg = self.reg_loss(pred_valid[:, 1, :], gt_ious)
        return {"loss_pem_cls": loss_cls, "loss_pem_reg": loss_reg}
