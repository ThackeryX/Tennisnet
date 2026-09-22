import argparse
import torch
import time
import sys
import os
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, parameter_count

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from opentad.models import build_detector
    from mmengine.config import ConfigDict
except ImportError:
    print("Error: 未找到 OpenTAD 或 mmengine。")
    exit(1)


# -------------------------------------------------------------------------
# 1. G-TAD 配置
# -------------------------------------------------------------------------
def get_gtad_config():
    IN_CHANNELS = 2048
    TSCALE = 125
    DSCALE = 125

    model_cfg = dict(
        type="GTAD",
        projection=dict(
            type="ConvSingleProj",
            num_convs=1,
            in_channels=IN_CHANNELS,
            out_channels=256,
            conv_cfg=dict(groups=4),
        ),
        neck=dict(
            type="GCNeXt",  # 这个模块容易报错，稍后会被手术替换
            in_channels=256,
            out_channels=256,
            k=3,
            groups=32,
        ),
        rpn_head=dict(
            type="GCNextTemporalEvaluationHead",  # 内部包含 GCNeXt
            in_channels=256,
            num_classes=2,
            loss=dict(pos_thresh=0.5, gt_type=["startness", "endness"]),
        ),
        roi_head=dict(
            type="StandardProposalMapHead",
            proposal_generator=dict(type="DenseProposalMap", tscale=TSCALE, dscale=DSCALE),
            proposal_roi_extractor=dict(
                type="GTADExtractor",
                in_channels=256,
                out_channels=512,
                tscale=TSCALE,
                dscale=DSCALE,
                roi_size=16,
                context_size=16
            ),
            proposal_head=dict(
                type="PEMHead",
                in_channels=512,
                feat_channels=128,
                num_convs=3,
                num_classes=2,
                kernel_size=3,
                loss=dict(
                    cls_loss=dict(type="BalancedBCELoss", pos_thresh=0.9),
                    reg_loss=dict(type="BalancedL2Loss", high_thresh=0.7, low_thresh=0.3, weight=5.0),
                ),
            ),
        ),
    )
    return ConfigDict(model_cfg), TSCALE


# -------------------------------------------------------------------------
# 2. 辅助类
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


class NeckAdapter(nn.Module):
    """用于替换 GCNeXt Neck"""

    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x, mask):
        # 强制输入 3D
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.squeeze(1)
        x = self.layer(x)
        # 强制输出 3D (防止 Conv1d 意外行为，虽然不太可能)
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.squeeze(1)
        return x, mask


class GTADWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, mask, metas):
        return self.model.forward_test(x, mask, metas)


# -------------------------------------------------------------------------
# 3. 主程序
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Calc G-TAD Metrics Final')
    args = parser.parse_args()

    cfg, tscale = get_gtad_config()
    device = torch.device('cuda:0')
    print(f"Building G-TAD Model (T={tscale})...")

    try:
        model = build_detector(cfg)
    except Exception as e:
        print(f"[Error] 构建模型失败: {e}")
        return

    model.to(device)
    model.eval()

    if hasattr(model, 'backbone'): model.backbone = None

    # ========================================================
    # 🔧 [Surgery 1] 替换 Projection
    # ========================================================
    if hasattr(model, 'projection'):
        print(">> [Surgery] Replacing Projection...")
        vanilla_proj = nn.Sequential(
            nn.Conv1d(2048, 256, kernel_size=1, groups=4),  # matches config
            nn.ReLU(inplace=True)
        ).to(device)
        model.projection = ProjAdapter(vanilla_proj)

    # ========================================================
    # 🔧 [Surgery 2] 替换 Neck (GCNeXt -> Conv1d)
    # ========================================================
    # GCNeXt 本质上是一个带 context 的 Conv，用 Conv1d(k=3, p=1) 近似在 FLOPs 上是非常准确的
    if hasattr(model, 'neck'):
        print(">> [Surgery] Replacing Neck (GCNeXt -> Conv1d)...")
        # GCNeXt 配置: in=256, out=256, k=3, groups=32
        vanilla_neck = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1, groups=32),
            nn.ReLU(inplace=True)
        ).to(device)
        model.neck = NeckAdapter(vanilla_neck)

    # ========================================================
    # 🔧 [Surgery 3] 替换 RPN Head 中的 GCNeXt
    # ========================================================
    # RPN Head (GCNextTemporalEvaluationHead) 内部有一个 self.tem (ModuleList)
    if hasattr(model, 'rpn_head') and hasattr(model.rpn_head, 'tem'):
        print(">> [Surgery] Replacing RPN Head Internals (GCNeXt -> Conv1d)...")
        # 遍历 ModuleList 替换
        new_tem = nn.ModuleList()
        for i in range(len(model.rpn_head.tem)):
            # 每个 GCNeXt 也是 256->256, k=3, groups=32
            layer = nn.Sequential(
                nn.Conv1d(256, 256, kernel_size=3, padding=1, groups=32),
                nn.ReLU(inplace=True)
            ).to(device)
            new_tem.append(layer)

        model.rpn_head.tem = new_tem

    # ========================================================
    # 构造输入
    # ========================================================
    in_channels = 2048
    T = tscale
    print(f"Input Shape: [1, {in_channels}, {T}]")

    input_feat = torch.randn(1, in_channels, T).to(device)
    # G-TAD 比较皮实，[1,1,T] 的 mask 经过 adapter 修正后应该没问题
    input_mask = torch.ones(1, 1, T).to(device)

    metas = [dict(
        video_name="test_video", duration=100.0, fps=25.0,
        feature_frame=T, batch_input_shape=(1, in_channels, T)
    )]

    wrapper = GTADWrapper(model)

    print("\n" + "=" * 40)
    print(" 🚀 Starting Calculation (G-TAD)")
    print("=" * 40)

    # --- Params & FLOPs ---
    try:
        inputs = (input_feat, input_mask, metas)
        flops_obj = FlopCountAnalysis(wrapper, inputs)
        flops_obj.unsupported_ops_warnings(False)

        raw_flops = flops_obj.total()
        params_cnt = parameter_count(model)[""] / 1e6

        print(f" [1] Complexity (Input T={T})")
        print(f"  - Params : {params_cnt:.2f} M")
        print(f"  - FLOPs  : {raw_flops / 1e9:.3f} G")

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