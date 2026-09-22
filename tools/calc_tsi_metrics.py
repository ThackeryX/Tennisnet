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
# 2. TSI 配置
# -------------------------------------------------------------------------
def get_tsi_config():
    IN_CHANNELS = 2048

    # [关键修复] 将 T 从 125 改为 128
    # 125 导致 U-Net 结构上下采样不对齐 (31 vs 30)
    # 128 是最接近且能被 8 整除的数字，计算量几乎一致
    TSCALE = 128
    DSCALE = 128

    model_cfg = dict(
        type="TSI",
        projection=dict(
            type="ConvSingleProj",
            in_channels=IN_CHANNELS,
            out_channels=256,
            num_convs=2,
            conv_cfg=dict(groups=4),
        ),
        rpn_head=dict(
            type="LocalGlobalTemporalEvaluationHead",
            in_channels=256,
            loss=dict(pos_thresh=0.5),
        ),
        roi_head=dict(
            type="StandardProposalMapHead",
            proposal_generator=dict(type="DenseProposalMap", tscale=TSCALE, dscale=DSCALE),
            proposal_roi_extractor=dict(
                type="BMNExtractor",
                in_channels=256,
                roi_channels=512,
                out_channels=128,
                tscale=TSCALE,
                dscale=DSCALE,
                prop_extend_ratio=0.5,
            ),
            proposal_head=dict(
                type="TSIHead",
                in_channels=128,
                feat_channels=128,
                num_convs=2,
                num_classes=2,
                loss=dict(
                    cls_loss=dict(type="ScaleInvariantLoss", pos_thresh=0.9),
                    reg_loss=dict(type="BalancedL2Loss", high_thresh=0.7, low_thresh=0.3, weight=5.0),
                ),
            ),
        ),
    )
    return ConfigDict(model_cfg), TSCALE


# -------------------------------------------------------------------------
# 3. 辅助类
# -------------------------------------------------------------------------
class ProjAdapter(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x, mask):
        # 强制维度修正 [B, 1, C, T] -> [B, C, T]
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.squeeze(1)
        x = self.layer(x)
        return x, mask


class TSIWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, mask, metas):
        return self.model.forward_test(x, mask, metas)


# -------------------------------------------------------------------------
# 4. 主程序
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Calc TSI Metrics v2')
    args = parser.parse_args()

    cfg, tscale = get_tsi_config()
    device = torch.device('cuda:0')
    print(f"Building TSI Model (T={tscale}, D={tscale})...")

    try:
        model = build_detector(cfg)
    except Exception as e:
        print(f"[Error] 构建模型失败: {e}")
        return

    model.to(device)
    model.eval()

    if hasattr(model, 'backbone'): model.backbone = None

    # ========================================================
    # 🔧 [Surgery] 替换 Projection
    # ========================================================
    if hasattr(model, 'projection'):
        print(">> [Surgery] Replacing Projection...")
        # TSI 配置: ConvSingleProj, num_convs=2, groups=4, 2048->256
        vanilla_proj = nn.Sequential(
            nn.Conv1d(2048, 256, kernel_size=1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=1, groups=4),
            nn.ReLU(inplace=True)
        ).to(device)
        model.projection = ProjAdapter(vanilla_proj)

    # 构造输入
    in_channels = 2048
    T = tscale
    print(f"Input Shape: [1, {in_channels}, {T}]")

    input_feat = torch.randn(1, in_channels, T).to(device)
    input_mask = torch.ones(1, 1, T).to(device)

    metas = [dict(
        video_name="test_video", duration=100.0, fps=25.0,
        feature_frame=T, batch_input_shape=(1, in_channels, T)
    )]

    wrapper = TSIWrapper(model)

    print("\n" + "=" * 40)
    print(" 🚀 Starting Calculation (TSI)")
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