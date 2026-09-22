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
# 2. TemporalMaxer 配置 (纯净版，不注入多余参数)
# -------------------------------------------------------------------------
def get_temporalmaxer_config():
    TRUNC_LEN = 1600

    # 按照你提供的配置进行合并
    model_cfg = dict(
        type="TemporalMaxer",
        projection=dict(
            type="TemporalMaxerProj",
            in_channels=2048,
            out_channels=512,
            arch=(2, 0, 5),
            conv_cfg=dict(kernel_size=3),
            norm_cfg=dict(type="LN"),
            drop_out=0.2,  # Override 合并
        ),
        neck=dict(
            type="FPNIdentity",
            in_channels=512,
            out_channels=512,
            num_levels=6,
            norm_cfg=dict(type="GN", num_groups=4, affine=False),  # Override 合并
        ),
        rpn_head=dict(
            type="TemporalMaxerHead",
            num_classes=1,
            in_channels=512,
            feat_channels=512,
            num_convs=2,
            cls_prior_prob=0.01,
            prior_generator=dict(
                type="PointGenerator",
                strides=[1, 2, 4, 8, 16, 32],
                regression_range=[(0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 10000)],
            ),
            loss_normalizer=100,
            loss_normalizer_momentum=0.9,
            loss=dict(
                cls_loss=dict(type="FocalLoss"),
                reg_loss=dict(type="DIOULoss"),
            ),
            assigner=dict(
                type="AnchorFreeSimOTAAssigner",
                iou_weight=2,
                cls_weight=1.0,
                center_radius=1.5,
                keep_percent=1.0,
                confuse_weight=0.0,
            ),
        ),
    )

    # [修正] 不再自动注入 max_seq_len，严格遵守原始配置
    return ConfigDict(model_cfg), TRUNC_LEN


# -------------------------------------------------------------------------
# 3. 包装器
# -------------------------------------------------------------------------
class TemporalMaxerWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, mask, metas):
        return self.model.forward_test(x, mask, metas)


# -------------------------------------------------------------------------
# 4. 主程序
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Calc TemporalMaxer Metrics v2')
    args = parser.parse_args()

    cfg, trunc_len = get_temporalmaxer_config()
    device = torch.device('cuda:0')
    print(f"Building TemporalMaxer Model (T={trunc_len})...")

    try:
        model = build_detector(cfg)
    except Exception as e:
        print(f"[Error] 构建模型失败: {e}")
        return

    model.to(device)
    model.eval()

    if hasattr(model, 'backbone'): model.backbone = None

    # 构造输入
    in_channels = 2048
    T = trunc_len
    print(f"Input Shape: [1, {in_channels}, {T}]")

    input_feat = torch.randn(1, in_channels, T).to(device)

    # Mask 策略: One-Stage 模型通常期望 [B, T] 的 Bool Mask
    input_mask = torch.ones(1, T).to(device).bool()

    metas = [dict(
        video_name="test_video", duration=100.0, fps=25.0,
        feature_frame=T, batch_input_shape=(1, in_channels, T)
    )]

    wrapper = TemporalMaxerWrapper(model)

    print("\n" + "=" * 40)
    print(" 🚀 Starting Calculation (TemporalMaxer)")
    print("=" * 40)

    # --- Params & FLOPs ---
    try:
        inputs = (input_feat, input_mask, metas)
        flops_obj = FlopCountAnalysis(wrapper, inputs)
        flops_obj.unsupported_ops_warnings(False)

        raw_flops = flops_obj.total()
        params_cnt = parameter_count(model)[""] / 1e6

        print(f" [1] Complexity (Input T={T})")
        print(f"  - Params        : {params_cnt:.2f} M")
        print(f"  - FLOPs         : {raw_flops / 1e9:.3f} G")

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