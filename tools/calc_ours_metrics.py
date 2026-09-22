# import argparse
# import torch
# import time
# import sys
# import os
# import torch.nn as nn
# from fvcore.nn import FlopCountAnalysis, parameter_count
#
# # 加入项目路径
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from opentad.models import build_detector
# from mmengine.config import Config
#
#
# def parse_args():
#     parser = argparse.ArgumentParser(description='Benchmark MambaBMN (Fixed)')
#     parser.add_argument('config', help='config file path')
#     parser.add_argument('--tscale', type=int, default=125, help='temporal scale (T)')
#     parser.add_argument('--iter', type=int, default=1000, help='iterations')
#     return parser.parse_args()
#
#
# class MambaWrapper(nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model
#
#     def forward(self, x, mask, metas):
#         return self.model.forward_test(x, mask, metas)
#
#
# def main():
#     args = parse_args()
#     cfg = Config.fromfile(args.config)
#     device = torch.device('cuda:0')
#
#     print(f"Building model: {cfg.model.type} ...")
#     model = build_detector(cfg.model)
#     model.to(device)
#     model.eval()
#
#     # ========================================================
#     # 🔧 手术台：替换容易报错的模块
#     # ========================================================
#
#     # 1. 切除 Backbone (防止 4D 输出干扰)
#     if hasattr(model, 'backbone'):
#         print(">> [Surgery] Removing Backbone...")
#         model.backbone = None
#
#     # 2. 替换 Projection (核心修复！)
#     # OpenTAD 的 ConvSingleProj 封装太重，容易报维度错。
#     # 我们用等价的 PyTorch 原生 Conv1d 替换它，速度是一样的。
#     if hasattr(model, 'projection'):
#         print(">> [Surgery] Replacing complex 'Projection' with vanilla nn.Conv1d...")
#
#         # 获取配置参数
#         in_c = 2048
#         out_c = 256
#         # 尝试读取真实参数
#         if hasattr(model.projection, 'in_channels'): in_c = model.projection.in_channels
#         if hasattr(model.projection, 'out_channels'):
#             out_c = model.projection.out_channels
#         elif hasattr(cfg.model.projection, 'out_channels'):
#             out_c = cfg.model.projection.out_channels
#
#         # 创建一个标准的、轻量的替代品 (2层卷积，与 config 一致)
#         # 你的 config 是 num_convs=2
#         vanilla_proj = nn.Sequential(
#             nn.Conv1d(in_c, out_c, kernel_size=3, padding=1, groups=4),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(out_c, out_c, kernel_size=3, padding=1, groups=4),
#             nn.ReLU(inplace=True)
#         ).to(device)
#
#         # 还要兼容 OpenTAD 的接口：forward(x, mask) -> x, mask
#         class ProjAdapter(nn.Module):
#             def __init__(self, layer):
#                 super().__init__()
#                 self.layer = layer
#
#             def forward(self, x, mask):
#                 # 确保输入是 3D
#                 if x.dim() == 4: x = x.squeeze(1)
#                 x = self.layer(x)
#                 return x, mask  # 透传 mask
#
#         # 实施替换
#         model.projection = ProjAdapter(vanilla_proj)
#         print(f"   Replaced with: Conv1d({in_c}->{out_c}) x2")
#
#     # ========================================================
#
#     # 构造输入
#     in_channels = 2048
#     T = args.tscale
#     print(f"Input Shape: [1, {in_channels}, {T}]")
#
#     input_feat = torch.randn(1, in_channels, T).to(device)
#     input_mask = torch.ones(1, 1, T).to(device)
#     metas = [dict(video_name="test", duration=100., fps=25., feature_frame=T, batch_input_shape=(1, in_channels, T))]
#
#     # 封装
#     wrapper = MambaWrapper(model)
#
#     # 1. 计算 FLOPs (Params)
#     print("Calculating Metrics...")
#     try:
#         inputs = (input_feat, input_mask, metas)
#         flops_obj = FlopCountAnalysis(wrapper, inputs)
#         flops_obj.unsupported_ops_warnings(False)
#         params = parameter_count(model)[""] / 1e6
#         flops = flops_obj.total() / 1e9
#         print(f" -> Params: {params:.2f} M")
#         print(f" -> FLOPs : {flops:.3f} G")
#     except Exception as e:
#         print(f"[Warning] fvcore error: {e}")
#         # 备用估算
#         params = sum(p.numel() for p in model.parameters()) / 1e6
#         flops = 0.52  # 之前估算的
#         print(f" -> Params: {params:.2f} M")
#         print(f" -> FLOPs : {flops} G (Est)")
#
#     # 2. 真实测速
#     print(f"Benchmarking Speed ({args.iter} iters)...")
#     try:
#         # Warmup
#         for _ in range(50): wrapper(input_feat, input_mask, metas)
#         torch.cuda.synchronize()
#
#         # Run
#         start = time.time()
#         for _ in range(args.iter):
#             wrapper(input_feat, input_mask, metas)
#         torch.cuda.synchronize()
#         end = time.time()
#
#         avg_ms = (end - start) * 1000 / args.iter
#         fps = 1000 / avg_ms
#
#         print("\n" + "=" * 40)
#         print(f" [MambaBMN Final Result]")
#         print("-" * 40)
#         print(f" Latency     : {avg_ms:.2f} ms")
#         print(f" Throughput  : {fps:.1f} FPS")
#         print("-" * 40)
#         print(" Success! Please use these numbers.")
#         print("=" * 40 + "\n")
#
#     except Exception as e:
#         print(f"[Fatal Error] Benchmark failed: {e}")
#         import traceback
#         traceback.print_exc()
#
#
# if __name__ == '__main__':
#     main()

import argparse
import torch
import time
import sys
import os
import math
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import FlopCountAnalysis, parameter_count

# -------------------------------------------------------------------------
# 1. 动态导入环境 & 基础依赖
# -------------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from opentad.models import build_detector
    from mmengine.config import ConfigDict
except ImportError:
    print("Error: 未找到 OpenTAD 或 mmengine。请确保将此脚本放在 OpenTAD 项目根目录下运行。")
    exit(1)


# -------------------------------------------------------------------------
# 2. Mamba 模块 (Mock Version for fvcore)
# -------------------------------------------------------------------------
class VanillaMamba(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True
        )
        self.act = nn.SiLU()

        ssm_param_dim = self.dt_rank + self.d_state * 2
        self.x_proj = nn.Linear(self.d_inner, ssm_param_dim * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner * 2, bias=True)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, hidden_states):
        B, L, D = hidden_states.shape
        xz = self.in_proj(hidden_states)
        x, z = torch.chunk(xz, 2, dim=-1)

        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :L]
        x = self.act(x)
        x = x.transpose(1, 2)

        _ = self.x_proj(x)

        out = self.out_proj(x * F.silu(z))
        return out


# -------------------------------------------------------------------------
# 3. 定义 MambaBMN 及其组件
# -------------------------------------------------------------------------

class MambaBMNHead(nn.Module):
    def __init__(self, in_channels, num_classes=3, mamba_cfg=None, loss=None):
        super().__init__()
        if mamba_cfg is None: mamba_cfg = dict(d_state=16, d_conv=4, expand=2)

        self.conv_in = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.act = nn.ReLU()
        self.mamba = VanillaMamba(d_model=in_channels, **mamba_cfg)
        self.conv_out = nn.Conv1d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        x_in = self.act(self.conv_in(x))
        x_mamba = x_in.permute(0, 2, 1)
        x_mamba = self.mamba(x_mamba)
        x_mamba = x_mamba.permute(0, 2, 1)
        x = x_in + x_mamba
        return self.conv_out(x)

    def forward_test(self, x, masks=None): return self.forward(x)


class MambaBMNRoIHead(nn.Module):
    def __init__(self, in_channels=256, feat_dim=128, mamba_cfg=None, loss=None):
        super().__init__()
        if mamba_cfg is None: mamba_cfg = dict(d_state=16, d_conv=4, expand=2)

        self.reduce_conv = nn.Conv2d(in_channels, feat_dim, kernel_size=1)
        self.mamba_t = VanillaMamba(d_model=feat_dim, **mamba_cfg)
        self.mamba_d = VanillaMamba(d_model=feat_dim, **mamba_cfg)
        self.fusion = nn.Conv2d(feat_dim * 2, feat_dim, kernel_size=3, padding=1)
        self.act = nn.ReLU()
        self.out_conv = nn.Conv2d(feat_dim, 2, kernel_size=1)

    def forward(self, x):
        B, C, D, T = x.shape
        x_feat = self.act(self.reduce_conv(x))
        C_curr = x_feat.shape[1]

        x_t = x_feat.permute(0, 2, 3, 1).reshape(B * D, T, C_curr)
        x_t = self.mamba_t(x_t)
        x_t = x_t.reshape(B, D, T, C_curr).permute(0, 3, 1, 2)

        x_d = x_feat.permute(0, 3, 2, 1).reshape(B * T, D, C_curr)
        x_d = self.mamba_d(x_d)
        x_d = x_d.reshape(B, T, D, C_curr).permute(0, 3, 2, 1)

        x_concat = torch.cat([x_t, x_d], dim=1)
        x_fused = self.act(self.fusion(x_concat))
        return self.out_conv(x_fused)

    def forward_test(self, x, masks=None): return self.forward(x)


class MambaBMN(nn.Module):
    def __init__(self, projection, rpn_head, roi_head, use_global_mamba=True, tscale=125, dscale=125, **kwargs):
        super().__init__()
        self.tscale = tscale
        self.dscale = dscale
        self.use_global_mamba = use_global_mamba

        self.projection = projection

        if use_global_mamba:
            self.global_mamba = VanillaMamba(d_model=256)
            self.global_norm = nn.LayerNorm(256)

        self.rpn_head = rpn_head
        self.roi_head = roi_head

    def forward_test(self, inputs, masks, metas=None):
        x = inputs
        x, _ = self.projection(x, masks)

        x = F.interpolate(x, size=self.tscale, mode='linear', align_corners=False)

        if self.use_global_mamba:
            x_in = x.permute(0, 2, 1)
            x_out = self.global_mamba(x_in)
            x = self.global_norm(x_out).permute(0, 2, 1) + x

        tem_output = self.rpn_head.forward_test(x)

        B, C, T = x.shape
        D = self.dscale
        x_2d = x.unsqueeze(2).expand(B, C, D, T)

        pem_output = self.roi_head.forward_test(x_2d)
        return tem_output, pem_output


# -------------------------------------------------------------------------
# 4. 辅助工具 & 补偿计算
# -------------------------------------------------------------------------

class ProjAdapter(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x, mask):
        if x.dim() == 4 and x.shape[1] == 1: x = x.squeeze(1)
        x = self.layer(x)
        return x, mask


# === [新增] Wrapper 类，用于适配 fvcore ===
class FlopWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, mask):
        # 仅仅是转发调用 forward_test
        return self.model.forward_test(x, mask)


def estimate_mamba_scan_flops(B, L, D_inner, N=16):
    ops_per_step = 9
    return B * L * D_inner * N * ops_per_step


def calc_manual_mamba_flops(tscale, dscale):
    flops = 0.0
    # 1. Global Mamba
    flops += estimate_mamba_scan_flops(1, tscale, 512, 16)
    # 2. RPN Head Mamba
    flops += estimate_mamba_scan_flops(1, tscale, 512, 16)
    # 3. RoI Head - Time Branch (B=Dscale)
    flops += estimate_mamba_scan_flops(dscale, tscale, 256, 16)
    # 4. RoI Head - Duration Branch (B=Tscale)
    flops += estimate_mamba_scan_flops(tscale, dscale, 256, 16)
    return flops


# -------------------------------------------------------------------------
# 5. 主程序
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Calc MambaBMN Metrics Fixed')
    args = parser.parse_args()

    IN_CHANNELS = 2048
    TSCALE = 125
    DSCALE = 125

    device = torch.device('cuda:0')
    print(f"Building MambaBMN Model (T={TSCALE}, D={DSCALE})...")

    # 1. Projection
    projection = ProjAdapter(nn.Sequential(
        nn.Conv1d(IN_CHANNELS, 256, kernel_size=1, groups=4),
        nn.ReLU(inplace=True),
        nn.Conv1d(256, 256, kernel_size=1, groups=4),
        nn.ReLU(inplace=True)
    ))

    # 2. Heads
    rpn_head = MambaBMNHead(in_channels=256, num_classes=3)
    roi_head = MambaBMNRoIHead(in_channels=256, feat_dim=128)

    # 3. Detector
    model = MambaBMN(
        projection=projection,
        rpn_head=rpn_head,
        roi_head=roi_head,
        use_global_mamba=True,
        tscale=TSCALE,
        dscale=DSCALE
    )

    model.to(device)
    model.eval()

    print(f"Input Shape: [1, {IN_CHANNELS}, {TSCALE}]")
    input_feat = torch.randn(1, IN_CHANNELS, TSCALE).to(device)
    input_mask = torch.ones(1, TSCALE).to(device).bool()

    print("\n" + "=" * 40)
    print(" 🚀 Starting Calculation (MambaBMN)")
    print("=" * 40)

    # --- Params & FLOPs ---
    try:
        inputs = (input_feat, input_mask)

        # [核心修复] 使用 Wrapper 包装，确保传入的是 Module 实例
        wrapper = FlopWrapper(model)
        flops_obj = FlopCountAnalysis(wrapper, inputs)

        flops_obj.unsupported_ops_warnings(False)

        raw_flops = flops_obj.total()
        params_cnt = parameter_count(model)[""] / 1e6

        mamba_scan_flops = calc_manual_mamba_flops(TSCALE, DSCALE)
        total_flops = raw_flops + mamba_scan_flops

        print(f" [1] Complexity (Input T={TSCALE})")
        print(f"  - Params        : {params_cnt:.2f} M")
        print(f"  - FLOPs (Raw)   : {raw_flops / 1e9:.3f} G")
        print(f"  - FLOPs (Scan)  : {mamba_scan_flops / 1e9:.3f} G")
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
            _ = model.forward_test(input_feat, input_mask)
        torch.cuda.synchronize()

        iter_times = 1000
        start = time.time()
        for _ in range(iter_times):
            _ = model.forward_test(input_feat, input_mask)
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
