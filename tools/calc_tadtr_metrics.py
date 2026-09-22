import argparse
import torch
import time
import sys
import os
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, parameter_count

# -------------------------------------------------------------------------
# 1. 动态导入环境
# -------------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from opentad.models import build_detector
    from mmengine.config import ConfigDict
except ImportError:
    print("Error: 未找到 OpenTAD 或 mmengine。请确保将此脚本放在 OpenTAD 项目根目录下运行。")
    exit(1)


# -------------------------------------------------------------------------
# 2. TadTR 配置
# -------------------------------------------------------------------------
def get_tadtr_config():
    WINDOW_SIZE = 125
    model_cfg = dict(
        type="TadTR",
        projection=dict(
            type="ConvSingleProj",
            in_channels=2048,
            out_channels=256,
            num_convs=1,
            conv_cfg=dict(kernel_size=1, padding=0),
            norm_cfg=dict(type="GN", num_groups=32),
            act_cfg=None,
        ),
        transformer=dict(
            type="TadTRTransformer",
            num_proposals=40,
            num_classes=1,
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
                num_classes=1,
                matcher=dict(
                    type="HungarianMatcher",
                    cost_class=6.0,
                    cost_bbox=5.0,
                    cost_giou=2.0,
                    cost_class_type="focal_loss_cost",
                    iou_type="iou",
                    use_multi_class=False,
                ),
                loss_class_type="focal_loss",
                weight_dict=dict(loss_class=2.0, loss_bbox=5.0, loss_iou=2.0, loss_actionness=4.0),
                use_multi_class=False,
            ),
        ),
    )
    return ConfigDict(model_cfg), WINDOW_SIZE


# -------------------------------------------------------------------------
# 3. 修复工具类 (Defense Logic Added)
# -------------------------------------------------------------------------

class ProjAdapter(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x, mask):
        # 1. 修复特征维度 [B, 1, C, T] -> [B, C, T]
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.squeeze(1)

        # 2. [关键修复] 修复 Mask 维度 [B, 1, T] -> [B, T]
        # DeformableDETRTransformer 里的 get_valid_ratio 需要 2D Mask
        if mask.dim() == 3 and mask.shape[1] == 1:
            mask = mask.squeeze(1)

        x = self.layer(x)
        return x, mask


class DummyPosEmbed(nn.Module):
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.num_pos_feats = num_pos_feats

    def forward(self, mask):
        # 此时 mask 已经被 ProjAdapter 修正为 [B, T]
        B = mask.size(0)
        T = mask.size(-1)
        return torch.zeros((B, T, self.num_pos_feats), device=mask.device)


class TadTRWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, mask, metas):
        return self.model.forward_test(x, mask, metas)


# -------------------------------------------------------------------------
# 4. FLOPs 补偿
# -------------------------------------------------------------------------
def estimate_deform_attn_flops(cfg, T):
    dim = cfg.transformer.encoder.embed_dim
    n_heads = cfg.transformer.encoder.num_heads
    n_points = cfg.transformer.encoder.num_points
    n_levels = cfg.transformer.encoder.num_feature_levels
    enc_layers = cfg.transformer.encoder.num_layers
    dec_layers = cfg.transformer.decoder.num_layers
    num_proposals = cfg.transformer.num_proposals
    ops_per_point = 5

    flops = 0.0
    flops += T * n_heads * n_levels * n_points * dim * ops_per_point * enc_layers
    flops += num_proposals * n_heads * n_levels * n_points * dim * ops_per_point * dec_layers
    flops += (num_proposals ** 2) * dim * 2 * dec_layers
    return flops


# -------------------------------------------------------------------------
# 5. 主程序
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Calc TadTR Metrics Final v3')
    args = parser.parse_args()

    cfg, window_size = get_tadtr_config()
    device = torch.device('cuda:0')
    print(f"Building TadTR Model (Window Size T={window_size})...")

    try:
        model = build_detector(cfg)
    except Exception as e:
        print(f"[Error] 构建模型失败: {e}")
        return

    model.to(device)
    model.eval()

    if hasattr(model, 'backbone'): model.backbone = None

    # [Surgery 1] Projection
    if hasattr(model, 'projection'):
        print(">> [Surgery] Replacing Projection...")
        vanilla_proj = nn.Sequential(
            nn.Conv1d(2048, 256, kernel_size=1, padding=0),
            nn.GroupNorm(32, 256)
        ).to(device)
        model.projection = ProjAdapter(vanilla_proj)

    # [Surgery 2] Position Embedding
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'position_embedding'):
        print(">> [Surgery] Replacing PositionEmbeddingSine with DummyPosEmbed...")
        model.transformer.position_embedding = DummyPosEmbed(256)

    # 构造输入
    in_channels = 2048
    T = window_size
    print(f"Input Shape: [1, {in_channels}, {T}]")

    input_feat = torch.randn(1, in_channels, T).to(device)

    # [关键修复] 直接构造 2D Mask [1, T]，避免后续解包错误
    # 如果 ProjAdapter 工作正常，即使这里传了 [1, 1, T] 也没事，但这样更保险
    input_mask = torch.ones(1, T).to(device).bool()

    metas = [dict(
        video_name="test_video", duration=100.0, fps=25.0,
        feature_frame=T, batch_input_shape=(1, in_channels, T)
    )]

    wrapper = TadTRWrapper(model)

    print("\n" + "=" * 40)
    print(" 🚀 Starting Calculation (TadTR)")
    print("=" * 40)

    # --- FLOPs ---
    try:
        inputs = (input_feat, input_mask, metas)
        flops_obj = FlopCountAnalysis(wrapper, inputs)
        flops_obj.unsupported_ops_warnings(False)

        raw_flops = flops_obj.total()
        params_cnt = parameter_count(model)[""] / 1e6
        manual_flops = estimate_deform_attn_flops(cfg, T)
        total_flops = raw_flops + manual_flops

        print(f" [1] Complexity (Input T={T})")
        print(f"  - Params        : {params_cnt:.2f} M")
        print(f"  - FLOPs (Raw)   : {raw_flops / 1e9:.3f} G")
        print(f"  - FLOPs (Attn)  : {manual_flops / 1e9:.3f} G")
        print(f"  - FLOPs (Total) : {total_flops / 1e9:.3f} G")

    except Exception as e:
        print(f"  [Error] FLOPs calculation failed: {e}")
        import traceback;
        traceback.print_exc()

    # --- Speed ---
    print("\n [2] Speed Benchmark")
    try:
        # Warmup
        for _ in range(50):
            _ = wrapper(input_feat, input_mask, metas)
        torch.cuda.synchronize()

        iter_times = 1000
        start = time.time()
        for _ in range(iter_times):
            _ = wrapper(input_feat, input_mask, metas)
        torch.cuda.synchronize()
        end = time.time()

        avg_latency = (end - start) * 1000 / iter_times
        fps = 1000 / avg_latency

        print(f"  - Latency : {avg_latency:.2f} ms")
        print(f"  - FPS     : {fps:.1f}")

    except Exception as e:
        print(f"  [Error] Speed benchmark failed: {e}")
        import traceback;
        traceback.print_exc()

    print("=" * 40 + "\n")


if __name__ == '__main__':
    main()
    