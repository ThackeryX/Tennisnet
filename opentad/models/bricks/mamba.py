import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    selective_scan_fn = None


class Mamba(nn.Module):
    """
    Bidirectional State Space Model (Mamba) Block with shared causal 1D convolution.
    """
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        bidirectional=True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.bidirectional = bidirectional

        # 1. In-projection
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)

        # 2. Shared depthwise causal 1D convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=conv_bias,
        )
        self.act = nn.SiLU()

        # 3. SSM parameter projections
        ssm_param_dim = self.dt_rank + self.d_state * 2
        num_directions = 2 if self.bidirectional else 1

        self.x_proj = nn.Linear(self.d_inner, ssm_param_dim * num_directions, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner * num_directions, bias=True)

        # Initialize A and D parameters
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).unsqueeze(0)
        A = A.expand(self.d_inner, -1).clone()
        self.A_log = nn.Parameter(torch.log(A).repeat(num_directions, 1, 1))

        self.D = nn.Parameter(torch.ones(num_directions, self.d_inner))
        self.D._no_weight_decay = True

        # Initialize dt projection weights and biases
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)

        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            if self.bidirectional:
                self.dt_proj.bias.copy_(torch.cat([inv_dt, inv_dt]))
            else:
                self.dt_proj.bias.copy_(inv_dt)

        # 4. Out-projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

    def forward(self, hidden_states):
        batch, seqlen, dim = hidden_states.shape

        # 1. Project & Split
        xz = self.in_proj(hidden_states)
        x, z = torch.chunk(xz, 2, dim=-1)
        x = x.transpose(1, 2)  # [B, D, L]

        # 2. Local 1D Convolution
        x_conv = self.conv1d(x)[:, :, :seqlen]
        x_local = self.act(x_conv)

        # 3. SSM Projection
        d_split = [self.dt_rank, self.d_state, self.d_state]

        if self.bidirectional:
            x_dbl = self.x_proj(x_local.transpose(1, 2))
            x_dbl_f, x_dbl_b = torch.chunk(x_dbl, 2, dim=-1)

            dt_f, B_f, C_f = torch.split(x_dbl_f, d_split, dim=-1)
            dt_b, B_b, C_b = torch.split(x_dbl_b, d_split, dim=-1)

            dt_f_out = F.linear(dt_f, self.dt_proj.weight[:self.d_inner], self.dt_proj.bias[:self.d_inner])
            dt_b_out = F.linear(dt_b, self.dt_proj.weight[self.d_inner:], self.dt_proj.bias[self.d_inner:])

            # Forward Scan
            A_f = -torch.exp(self.A_log[0])
            y_f = selective_scan_fn(
                x_local, dt_f_out.transpose(1, 2), A_f, B_f.transpose(1, 2), C_f.transpose(1, 2),
                self.D[0].float(), z=None, delta_bias=None, delta_softplus=True, return_last_state=False
            )

            # Backward Scan
            x_rev = torch.flip(x_local, dims=[-1])
            A_b = -torch.exp(self.A_log[1])
            y_b = selective_scan_fn(
                x_rev, dt_b_out.transpose(1, 2), A_b, B_b.transpose(1, 2), C_f.transpose(1, 2),
                self.D[1].float(), z=None, delta_bias=None, delta_softplus=True, return_last_state=False
            )
            y_b = torch.flip(y_b, dims=[-1])

            y = (y_f + y_b) / 2
        else:
            x_dbl = self.x_proj(x_local.transpose(1, 2))
            dt_f, B_f, C_f = torch.split(x_dbl, d_split, dim=-1)
            dt_f_out = self.dt_proj(dt_f)

            A_f = -torch.exp(self.A_log[0])
            y = selective_scan_fn(
                x_local, dt_f_out.transpose(1, 2), A_f, B_f.transpose(1, 2), C_f.transpose(1, 2),
                self.D[0].float(), z=None, delta_bias=None, delta_softplus=True, return_last_state=False
            )

        y = y.transpose(1, 2)
        out = self.out_proj(y * F.silu(z))
        return out
