import torch
import torch.nn as nn
from ..bricks import Mamba
from ..builder import HEADS
from ..losses import BalancedBCELoss
from ..utils.iou_tools import compute_ioa_torch


@HEADS.register_module()
class MambaBMNHead(nn.Module):
    """
    Temporal Evaluation Module (TEM) Head with Mamba Sequence Modeling.
    """
    def __init__(self, in_channels, num_classes=3, mamba_cfg=None, loss=None):
        super().__init__()
        if loss is None:
            loss = dict(pos_thresh=0.5, gt_type=["startness", "endness", "actionness"])

        self.gt_type = loss.get("gt_type", ["startness", "endness", "actionness"])
        self.num_classes = len(self.gt_type)

        self.conv_in = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.act = nn.ReLU(inplace=True)

        if mamba_cfg is None:
            mamba_cfg = dict(d_state=16, d_conv=4, expand=2)
        self.mamba = Mamba(d_model=in_channels, **mamba_cfg)

        self.conv_out = nn.Conv1d(in_channels, self.num_classes, kernel_size=1)
        self.tem_loss = BalancedBCELoss(pos_thresh=loss["pos_thresh"])

    def forward(self, x, masks=None):
        x_in = self.act(self.conv_in(x))
        x_mamba = self.mamba(x_in.permute(0, 2, 1)).permute(0, 2, 1)
        x = x_in + x_mamba
        score = self.conv_out(x)
        return score

    def forward_train(self, x, masks, gt_segments, **kwargs):
        score = self.forward(x, masks)
        losses = {"loss_tem": self.losses(score, gt_segments)}
        return losses

    def forward_test(self, x, masks=None):
        return self.forward(x, masks)

    @torch.no_grad()
    def prepare_targets(self, gt_segments, tscale):
        temporal_anchor = torch.stack((torch.arange(0, tscale), torch.arange(1, tscale + 1)), dim=1)
        temporal_anchor = temporal_anchor.to(gt_segments[0].device).float()

        calc_start = "startness" in self.gt_type
        calc_end = "endness" in self.gt_type
        calc_action = "actionness" in self.gt_type

        batch_starts, batch_ends, batch_actions = [], [], []

        for gt_segment in gt_segments:
            gt_xmins = gt_segment[:, 0]
            gt_xmaxs = gt_segment[:, 1]

            if calc_start:
                gt_start_bboxs = torch.stack((gt_xmins - 1.5, gt_xmins + 1.5), dim=1)
                gt_start = compute_ioa_torch(gt_start_bboxs, temporal_anchor)
                batch_starts.append(torch.max(gt_start, dim=1)[0])

            if calc_end:
                gt_end_bboxs = torch.stack((gt_xmaxs - 1.5, gt_xmaxs + 1.5), dim=1)
                gt_end = compute_ioa_torch(gt_end_bboxs, temporal_anchor)
                batch_ends.append(torch.max(gt_end, dim=1)[0])

            if calc_action:
                gt_action = compute_ioa_torch(gt_segment, temporal_anchor)
                batch_actions.append(torch.max(gt_action, dim=1)[0])

        final_gts = []
        if calc_start:
            final_gts.append(torch.stack(batch_starts))
        if calc_end:
            final_gts.append(torch.stack(batch_ends))
        if calc_action:
            final_gts.append(torch.stack(batch_actions))

        return final_gts

    def losses(self, pred, gt_segments):
        gts = self.prepare_targets(gt_segments, tscale=pred.shape[-1])
        assert len(gts) == pred.shape[1], (
            f"Prediction channels {pred.shape[1]} != Target types {len(gts)}."
        )
        pred = pred.sigmoid()
        loss = 0
        for i, gt in enumerate(gts):
            loss = loss + self.tem_loss(pred[:, i, :], gt)
        return loss
