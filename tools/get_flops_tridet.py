# import argparse
# import torch
# import sys
# import os
#
# # 加入项目路径
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#
# from opentad.models import build_detector
# from mmengine.config import Config
# from thop import profile
# from thop.utils import clever_format
#
#
# def parse_args():
#     parser = argparse.ArgumentParser(description='Calculate TriDet FLOPs/Params')
#     # 你的 TriDet 配置文件路径
#     parser.add_argument('config', help='config file path (e.g., configs/tridet/thumos_tsn.py)')
#     # 默认输入长度设为 125 (和你的 Mamba/BMN 保持一致以确保公平)
#     parser.add_argument('--shape', type=int, nargs='+', default=[2048, 125], help='input feature size (Channel, Time)')
#     return parser.parse_args()
#
#
# def main():
#     args = parse_args()
#     cfg = Config.fromfile(args.config)
#
#     print(f"Building TriDet model: {cfg.model.type} ...")
#     model = build_detector(cfg.model)
#     model.eval()
#
#     # 1. 确定输入维度
#     c = args.shape[0]
#     t = args.shape[1]
#
#     print(f"Profiling with input shape: [1, {c}, {t}]")
#
#     # 2. 构造输入 (TriDet 需要标准的 mask 和 metas)
#     input_feat = torch.randn(1, c, t)
#     input_mask = torch.ones(1, 1, t)
#
#     # TriDet 的后处理通常需要 batch_input_shape
#     dummy_metas = [dict(
#         video_name="test_video",
#         duration=100.0,
#         fps=25.0,
#         feature_frame=t,
#         batch_input_shape=(1, c, t)
#     )]
#
#     # 3. 尝试计算
#     try:
#         # TriDet 基于 anchor-free 机制，结构相对清晰，有时 thop 可以跑通
#         flops, params = profile(model, inputs=(input_feat, input_mask, dummy_metas), verbose=False)
#
#         flops_str, params_str = clever_format([flops, params], "%.3f")
#
#         print("\n" + "=" * 50)
#         print(f" [Comparison] TriDet Complexity (Calculated via thop)")
#         print("-" * 50)
#         print(f" Total Params : {params_str}")
#         print(f" Total FLOPs  : {flops_str}")
#         print("-" * 50)
#         print(" Note: TriDet uses SG-Former which is efficient,")
#         print("       but usually has more heads/bins than Mamba.")
#         print("=" * 50 + "\n")
#
#     except Exception as e:
#         print(f"[Info] thop profiling failed ({e}). Switching to Theoretical Estimation.")
#
#         # === 兜底方案：理论估算 ===
#         # TriDet 也是线性复杂度的 (O(T))
#         # 它的计算量主要在 Projection 和 Head
#
#         total_params = sum(p.numel() for p in model.parameters())
#
#         # 估算公式: 2 * Params * T
#         # TriDet 使用 FPN，每一层 T 减半，所以实际系数比 2 要小一点，
#         # 但为了保守估计（Upper Bound），用 2 是安全的。
#         estimated_flops = 2.0 * total_params * t
#
#         print("\n" + "=" * 50)
#         print(f" [Comparison] TriDet Complexity (Theoretical)")
#         print("-" * 50)
#         print(f" Total Params : {total_params / 1e6:.2f} M")
#         print(f" Est. FLOPs   : {estimated_flops / 1e9:.2f} G (Conservative)")
#         print("-" * 50)
#         print(f" Methodology: Based on T={t} and linear complexity.")
#         print("=" * 50 + "\n")
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
# 2. TriDet 配置 (基于你提供的代码)
# -------------------------------------------------------------------------
def get_tridet_config(target_len=1600):
    # 你的原始配置
    model_cfg = dict(
        type="TriDet",
        projection=dict(
            type="TriDetProj",
            in_channels=2048,
            out_channels=512,
            sgp_mlp_dim=768,
            arch=(2, 2, 5),
            downsample_type="max",
            sgp_win_size=[1, 1, 1, 1, 1, 1],
            k=5,
            init_conv_vars=0,
            conv_cfg=dict(kernel_size=3),
            norm_cfg=dict(type="LN"),
            path_pdrop=0.1,
            use_abs_pe=False,
            max_seq_len=2304,  # 原始配置是 2304
            input_noise=0.0,
        ),
        neck=dict(
            type="FPNIdentity",
            in_channels=512,
            out_channels=512,
            num_levels=6,
        ),
        rpn_head=dict(
            type="TriDetHead",
            num_classes=1,  # 假设是单类 (网球挥拍)
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
            boundary_kernel_size=3,
            iou_weight_power=0.2,
            num_bins=16,
            loss=dict(
                cls_loss=dict(type="FocalLoss"),
                reg_loss=dict(type="DIOULoss"),
                iou_rate=dict(type="GIOULoss"),
            ),
        ),
    )

    cfg = ConfigDict(model_cfg)

    # [关键步骤] 强制修改 max_seq_len 为 1600
    # 这样 TriDet 里的 pad_data 就会以 1600 为基准，而不是填充到 2304
    if target_len:
        print(f">> [Config Override] Forcing max_seq_len to {target_len}")
        cfg.projection.max_seq_len = target_len

    return cfg


# -------------------------------------------------------------------------
# 3. 包装器
# -------------------------------------------------------------------------
class TriDetWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, mask, metas):
        # TriDet 的 forward_test 返回 (points, rpn_reg, rpn_scores)
        return self.model.forward_test(x, mask, metas)


# -------------------------------------------------------------------------
# 4. 主程序
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Calc TriDet Metrics')
    args = parser.parse_args()

    # 设定目标长度
    TARGET_T = 1600

    # 1. 准备配置
    cfg = get_tridet_config(target_len=TARGET_T)
    device = torch.device('cuda:0')

    print(f"Building TriDet Model (Configured Max Seq Len={cfg.projection.max_seq_len})...")

    # 2. 构建模型
    try:
        model = build_detector(cfg)
    except Exception as e:
        print(f"\n[Error] 构建模型失败。请检查 OpenTAD 环境是否包含 TriDet 组件。")
        print(f"错误详情: {e}")
        return

    model.to(device)
    model.eval()

    # 切除 Backbone
    if hasattr(model, 'backbone'):
        model.backbone = None

    # 3. 构造输入
    in_channels = cfg.projection.in_channels
    print(f"Input Shape: [1, {in_channels}, {TARGET_T}]")

    input_feat = torch.randn(1, in_channels, TARGET_T).to(device)
    # TriDet 的 mask 通常是 Bool 类型 [B, T]
    input_mask = torch.ones(1, TARGET_T).to(device).bool()

    metas = [dict(
        video_name="test_video",
        duration=61.0,  # 你的视频时长
        fps=25.0,
        feature_frame=TARGET_T,
        batch_input_shape=(1, in_channels, TARGET_T)
    )]

    wrapper = TriDetWrapper(model)

    # 4. 计算指标
    print("\n" + "=" * 40)
    print(" 🚀 Starting Calculation (TriDet)")
    print("=" * 40)

    # --- Params & FLOPs ---
    try:
        inputs = (input_feat, input_mask, metas)
        flops_obj = FlopCountAnalysis(wrapper, inputs)
        flops_obj.unsupported_ops_warnings(False)

        params_cnt = parameter_count(model)[""] / 1e6
        flops_cnt = flops_obj.total() / 1e9

        print(f" [1] Complexity (Sequence T={TARGET_T})")
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

        iter_times = 500
        start = time.time()
        for _ in range(iter_times):
            _ = wrapper(input_feat, input_mask, metas)
        torch.cuda.synchronize()
        end = time.time()

        avg_latency = (end - start) * 1000 / iter_times
        fps = 1000 / avg_latency

        print(f"  - Latency : {avg_latency:.2f} ms")
        print(f"  - FPS     : {fps:.1f}")
        print(f"  (Note: Processing full {TARGET_T} frames per pass)")

    except Exception as e:
        print(f"  [Error] Speed benchmark failed: {e}")

    print("=" * 40 + "\n")


if __name__ == '__main__':
    main()
    