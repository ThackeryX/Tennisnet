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
#     parser = argparse.ArgumentParser(description='Calculate ActionFormer Theoretical Complexity')
#     parser.add_argument('config', help='train config file path')
#     parser.add_argument('--tscale', type=int, default=125, help='input sequence length')
#     return parser.parse_args()
#
#
# def main():
#     args = parse_args()
#     cfg = Config.fromfile(args.config)
#
#     print(f"Building ActionFormer model from: {args.config} ...")
#     model = build_detector(cfg.model)
#
#     # 1. 精确计算参数量
#     total_params = sum(p.numel() for p in model.parameters())
#
#     # 2. 理论估算 FLOPs
#     # ActionFormer 结构: Linear Projection + Transformer Blocks + FPN + Head
#     # 绝大部分计算量是 Linear 层 (Fully Connected)
#     # 估算公式: FLOPs ≈ 2 * Params * T_effective
#
#     # ActionFormer 使用特征金字塔，特征长度随层数减半: T, T/2, T/4, T/8...
#     # 平均有效长度 T_effective 约为输入 T 的 1.5倍左右 (考虑到多尺度求和)
#     # 但考虑到 Transformer 的 Self-Attention 复杂度较低 (Local Attention)，
#     # 且 FFN 层参数量巨大但只作用于单个 Token。
#
#     # 我们采用一个保守的通用估算: FLOPs ≈ 2 * Params * T
#     # 这是一个下界 (Lower Bound)，实际 FLOPs 会比这个只高不低。
#     estimated_flops = 2.0 * total_params * args.tscale
#
#     print("\n" + "=" * 50)
#     print(f" [Comparison] ActionFormer Complexity Report")
#     print("-" * 50)
#     print(f" Total Params : {total_params / 1e6:.2f} M")
#     print(f" Est. FLOPs   : {estimated_flops / 1e9:.2f} G (Conservative Lower Bound)")
#     print("-" * 50)
#     print(" Methodology: Theoretical estimation based on T=125.")
#     print(" Note: ActionFormer is significantly heavier in parameters.")
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
# 2. ActionFormer 配置 (基于你提供的内容)
# -------------------------------------------------------------------------
def get_af_config():
    # 你的 ActionFormer 配置
    model_cfg = dict(
        type="ActionFormer",
        projection=dict(
            type="Conv1DTransformerProj",
            in_channels=2048,
            out_channels=512,
            arch=(2, 2, 5),  # layers in embed / stem / branch
            conv_cfg=dict(kernel_size=3, proj_pdrop=0.0),
            norm_cfg=dict(type="LN"),
            attn_cfg=dict(n_head=4, n_mha_win_size=19),
            path_pdrop=0.1,
            use_abs_pe=False,
            max_seq_len=2304,  # 关键参数：最大序列长度
        ),
        neck=dict(
            type="FPNIdentity",
            in_channels=512,
            out_channels=512,
            num_levels=6,
        ),
        rpn_head=dict(
            type="ActionFormerHead",
            num_classes=20,
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
            center_sample="radius",
            center_sample_radius=1.5,
            label_smoothing=0.0,
            loss=dict(
                cls_loss=dict(type="FocalLoss"),
                reg_loss=dict(type="DIOULoss"),
            ),
        ),
    )
    return ConfigDict(model_cfg)


# -------------------------------------------------------------------------
# 3. 包装器
# -------------------------------------------------------------------------
class ActionFormerWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, mask, metas):
        # ActionFormer 的 forward_test 会自动处理 padding
        # 返回 (rpn_proposals, rpn_scores)
        return self.model.forward_test(x, mask, metas)


# -------------------------------------------------------------------------
# 4. 主程序
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Calc ActionFormer Metrics')
    args = parser.parse_args()

    # 1. 准备配置
    cfg = get_af_config()

    # 提取 max_seq_len，这是 ActionFormer 计算量的基准
    max_seq_len = 1600
    if 'max_seq_len' in cfg.projection:
        max_seq_len = cfg.projection.max_seq_len

    device = torch.device('cuda:0')
    print(f"Building ActionFormer Model (Max Seq Len={max_seq_len})...")

    # 2. 构建模型
    try:
        model = build_detector(cfg)
    except Exception as e:
        print(f"\n[Error] 构建模型失败。请检查 OpenTAD 环境是否包含 ActionFormer 组件。")
        print(f"错误详情: {e}")
        return

    model.to(device)
    model.eval()

    # 切除 Backbone (如果 build_detector 意外加上了的话)
    if hasattr(model, 'backbone'):
        model.backbone = None

    # 3. 构造输入
    # 为了模拟最真实的计算负载，我们直接构造长度为 max_seq_len 的输入
    # 这样避免了 pad_data 内部的 tensor copy，测速更准
    in_channels = cfg.projection.in_channels
    T = max_seq_len

    print(f"Input Shape: [1, {in_channels}, {T}] (Matching max_seq_len)")

    input_feat = torch.randn(1, in_channels, T).to(device)
    input_mask = torch.ones(1, T).to(device).bool()  # ActionFormer mask 维度通常是 [B, T] 或 [B, 1, T]

    # 兼容性处理：检查 mask 维度需求
    # 有些 OpenTAD 版本 mask 需要 [B, 1, T]
    if hasattr(model, 'pad_data'):
        # 简单测试一下
        try:
            model.pad_data(input_feat, input_mask)
        except:
            input_mask = torch.ones(1, 1, T).to(device)

    metas = [dict(
        video_name="test_video",
        duration=100.0,
        fps=25.0,
        feature_frame=T,
        batch_input_shape=(1, in_channels, T)
    )]

    wrapper = ActionFormerWrapper(model)

    # 4. 计算指标
    print("\n" + "=" * 40)
    print(" 🚀 Starting Calculation (ActionFormer)")
    print("=" * 40)

    # --- Params & FLOPs ---
    try:
        inputs = (input_feat, input_mask, metas)
        flops_obj = FlopCountAnalysis(wrapper, inputs)
        flops_obj.unsupported_ops_warnings(False)

        # ActionFormer 的 Transformer 包含大量 Attention
        # fvcore 通常能处理，但如果用了特殊算子可能会警告

        params_cnt = parameter_count(model)[""] / 1e6
        flops_cnt = flops_obj.total() / 1e9

        print(f" [1] Complexity (Window Size T={T})")
        print(f"  - Params : {params_cnt:.2f} M")
        print(f"  - FLOPs  : {flops_cnt:.3f} G")

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

        iter_times = 500  # ActionFormer 比较重，跑500次够了
        start = time.time()
        for _ in range(iter_times):
            _ = wrapper(input_feat, input_mask, metas)
        torch.cuda.synchronize()
        end = time.time()

        avg_latency = (end - start) * 1000 / iter_times
        fps = 1000 / avg_latency

        print(f"  - Latency : {avg_latency:.2f} ms")
        print(f"  - FPS     : {fps:.1f}")
        print(f"  (Note: FPS based on sliding window of {T} frames)")

    except Exception as e:
        print(f"  [Error] Speed benchmark failed: {e}")

    print("=" * 40 + "\n")


if __name__ == '__main__':
    main()
