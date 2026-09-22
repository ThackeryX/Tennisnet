# import argparse
# import sys
# import os
# import math
#
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from mmengine.config import Config
# from opentad.models import build_detector
#
#
# def parse_args():
#     parser = argparse.ArgumentParser(description='Calculate BMN Theoretical FLOPs')
#     parser.add_argument('config', help='train config file path')
#     # 默认 TSN 特征维度 2048
#     parser.add_argument('--in_channels', type=int, default=2048)
#     return parser.parse_args()
#
#
# def main():
#     args = parse_args()
#     cfg = Config.fromfile(args.config)
#
#     print(f"Building BMN model from: {args.config} ...")
#     model = build_detector(cfg.model)
#
#     # === 1. 获取关键参数 ===
#     # 尝试读取 tscale/dscale
#     tscale = 100  # 默认值
#     dscale = 100
#
#     if hasattr(cfg.model, 'tscale'):
#         tscale = cfg.model.tscale
#         dscale = cfg.model.dscale
#     elif hasattr(cfg.model.roi_head, 'proposal_generator'):
#         pg = cfg.model.roi_head.proposal_generator
#         tscale = pg.get('tscale', tscale)
#         dscale = pg.get('dscale', dscale)
#
#     print(f"Detected Configuration:")
#     print(f"  Tscale (T): {tscale}")
#     print(f"  Dscale (D): {dscale}")
#
#     # === 2. 计算 FLOPs (G) ===
#     # 公式: 2 * Cin * Cout * K * H * W (对于 Conv2d)
#     # 对于 Conv1d: 2 * Cin * Cout * K * T
#
#     flops = 0.0
#
#     # --- Part A: Projection (Backbone -> 256) ---
#     # 通常是 Conv1d: 2048 -> 256, k=1 or 3
#     # 这里我们简化估算，假设是两层 Conv1d
#     feat_dim = 256
#     # Layer 1: 2048 -> 256
#     flops += 2 * args.in_channels * feat_dim * 3 * tscale
#     # Layer 2: 256 -> 256
#     flops += 2 * feat_dim * feat_dim * 3 * tscale
#
#     print(f"  [Projection] FLOPs added.")
#
#     # --- Part B: TEM (Temporal Evaluation Module) ---
#     # 通常是 2-3 层 Conv1d on feature sequence
#     # 假设 2 层 Conv1d (256->256)
#     flops += 2 * 2 * feat_dim * feat_dim * 3 * tscale
#     print(f"  [TEM] FLOPs added.")
#
#     # --- Part C: PEM (Proposal Evaluation Module) ---
#     # 这是 BMN 的核心计算量来源
#     # 1. Map Generation (Matrix Multiplication)
#     # Output: [C, D, T]
#     # Sample Mask: [N, D] -> Tensordot
#     # Cost: C * T * D * N_sample_points (sample points usually 128 or 32)
#     # BMN 采样矩阵通常比较稀疏，或者用矩阵乘法实现
#     # 估算: C * T * D * 2 (乘加)
#     flops += 2 * feat_dim * tscale * dscale
#
#     # 2. PEM Layers (Conv2d / Conv3d on the Map)
#     # BMN 在 D*T 图上堆叠卷积
#     # 标准 BMN 有 4-5 层 Conv2d
#     # Input: [256, 125, 125] -> Conv2d -> [128, 125, 125] ...
#
#     # Layer 1: 256 -> 128, k=3
#     flops += 2 * 256 * 128 * 3 * 3 * dscale * tscale
#
#     # Layer 2: 128 -> 128, k=3
#     flops += 2 * 128 * 128 * 3 * 3 * dscale * tscale
#
#     # Layer 3: 128 -> 128, k=3
#     flops += 2 * 128 * 128 * 3 * 3 * dscale * tscale
#
#     # Layer 4 (Output): 128 -> 2 (Scores), k=1
#     flops += 2 * 128 * 2 * 1 * 1 * dscale * tscale
#
#     print(f"  [PEM] FLOPs added (Dominant part).")
#
#     # === 3. 计算 Params (M) ===
#     total_params = sum(p.numel() for p in model.parameters())
#
#     print("\n" + "=" * 50)
#     print(f" [Baseline] BMN Complexity Report")
#     print("-" * 50)
#     print(f" Total Params : {total_params / 1e6:.2f} M")
#     print(f" Total FLOPs  : {flops / 1e9:.2f} G")
#     print("-" * 50)
#     print(" Methodology: Theoretical summation of standard BMN layers")
#     print(f" (Proj + TEM + PEM) based on T={tscale}, D={dscale}.")
#     print("=" * 50 + "\n")
#
#
# if __name__ == '__main__':
#     main()

import argparse
import torch
import time
import sys
import os
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, parameter_count

# -------------------------------------------------------------------------
# 1. 动态导入环境 (假设你在 OpenTAD 环境下)
# -------------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from opentad.models import build_detector
    from mmengine.config import ConfigDict
except ImportError:
    print("Error: 未找到 OpenTAD 或 mmengine。请确保将此脚本放在 OpenTAD 项目根目录下运行。")
    exit(1)


# -------------------------------------------------------------------------
# 2. 你的 BMN 配置 (直接嵌入，确保无歧义)
# -------------------------------------------------------------------------
def get_bmn_config():
    # 显式定义 tscale 和 dscale，这是计算量的核心
    TSCALE = 125  # 标准 BMN 通常用 100，你的配置写了 125，这里我们用变量控制
    DSCALE = 125

    # 你的原始配置
    model_cfg = dict(
        type="BMN",
        projection=dict(
            type="ConvSingleProj",
            in_channels=2048,
            out_channels=256,
            num_convs=2,
            conv_cfg=dict(groups=4),  # 注意这里有 groups
        ),
        rpn_head=dict(
            type="TemporalEvaluationHead",
            in_channels=256,
            num_classes=2,
            conv_cfg=dict(groups=4),
            loss=dict(pos_thresh=0.5, gt_type=["startness", "endness"]),
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
                type="PEMHead",
                in_channels=128,
                feat_channels=128,
                num_convs=2,
                num_classes=2,
                loss=dict(
                    cls_loss=dict(type="BalancedBCELoss", pos_thresh=0.9),
                    reg_loss=dict(type="BalancedL2Loss", high_thresh=0.7, low_thresh=0.3, weight=5.0),
                ),
            ),
        ),
    )
    return ConfigDict(model_cfg), TSCALE


# -------------------------------------------------------------------------
# 3. 包装器：只暴露 forward_test 供 fvcore 分析
# -------------------------------------------------------------------------
class BMNWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, mask, metas):
        # 直接调用 forward_test，避开 post_processing (NMS)
        return self.model.forward_test(x, mask, metas)


# -------------------------------------------------------------------------
# 4. 主程序
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Calc BMN Metrics')
    parser.add_argument('--tscale', type=int, default=None, help='Override T scale')
    args = parser.parse_args()

    # 1. 准备配置
    cfg, default_tscale = get_bmn_config()
    if args.tscale:
        tscale = args.tscale
        # 更新配置中的 tscale (如果是动态类配置，这步很重要)
        cfg.roi_head.proposal_generator.tscale = tscale
        cfg.roi_head.proposal_roi_extractor.tscale = tscale
    else:
        tscale = default_tscale

    device = torch.device('cuda:0')
    print(f"Building BMN Model (T={tscale})...")

    # 2. 构建模型
    try:
        model = build_detector(cfg)
    except Exception as e:
        print(f"\n[Error] 构建模型失败。可能是你的 OpenTAD 版本没有注册 'BMN' 或组件。")
        print(f"错误详情: {e}")
        return

    model.to(device)
    model.eval()

    # --------------------------------------------------------
    # [手术台] 替换复杂的 Projection 层以确保 fvcore 准确计数
    # BMN 的 Projection 包含 Groups=4，fvcore 有时对自定义 Layer 识别不清
    # --------------------------------------------------------
    if hasattr(model, 'projection'):
        print(">> [Surgery] Replacing OpenTAD Projection with Vanilla PyTorch Conv1d...")
        # 你的配置: 2048->256, num_convs=2, groups=4
        vanilla_proj = nn.Sequential(
            nn.Conv1d(2048, 256, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True)
        ).to(device)

        # 适配器：保持接口一致 (x, mask) -> (x, mask)
        class ProjAdapter(nn.Module):
            def __init__(self, layer): super().__init__(); self.layer = layer

            def forward(self, x, mask): return self.layer(x), mask

        model.projection = ProjAdapter(vanilla_proj)
        print("   Replaced with: Conv1d(2048->256, g=4) x2")

    # 3. 构造输入
    # BMN 输入通常为 [B, C, T]
    in_channels = 2048
    input_feat = torch.randn(1, in_channels, tscale).to(device)
    # Mask 全 1
    input_mask = torch.ones(1, 1, tscale).to(device)
    # Metas (BMNExtractor 有时需要读 fps 或 duration，虽然 forward_test 通常不用)
    metas = [dict(
        video_name="test_video",
        duration=100.0,
        fps=25.0,
        feature_frame=tscale,
        batch_input_shape=(1, in_channels, tscale)
    )]

    wrapper = BMNWrapper(model)

    # 4. 计算指标
    print("\n" + "=" * 40)
    print(" 🚀 Starting Calculation")
    print("=" * 40)

    # --- Params & FLOPs ---
    try:
        inputs = (input_feat, input_mask, metas)
        flops_obj = FlopCountAnalysis(wrapper, inputs)
        # 忽略不支持的操作警告 (如一些 reshape 或 view)
        flops_obj.unsupported_ops_warnings(False)

        params_cnt = parameter_count(model)[""] / 1e6
        flops_cnt = flops_obj.total() / 1e9

        print(f" [1] Complexity (Input T={tscale})")
        print(f"  - Params : {params_cnt:.2f} M")
        print(f"  - FLOPs  : {flops_cnt:.3f} G")

        # 检查是否遗漏了 BMN 核心操作
        # 如果 PEMHead 没被统计到，FLOPs 会非常小
        if flops_cnt < 1.0:
            print("  ⚠️ 警告: FLOPs 过低，可能 fvcore 未追踪到 BMN Map 生成矩阵。")

    except Exception as e:
        print(f"  [Error] fvcore calculation failed: {e}")
        import traceback;
        traceback.print_exc()

    # --- Latency & FPS ---
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
        print("  (Note: Excluding NMS post-processing)")

    except Exception as e:
        print(f"  [Error] Speed benchmark failed: {e}")

    print("=" * 40 + "\n")


if __name__ == '__main__':
    main()

