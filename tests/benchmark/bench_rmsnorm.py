# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import argparse

import torch
import torch.nn.functional as F
import triton

import cuda.tile as ct
import tilegym
from tilegym.backend import is_backend_available
from tilegym.backend import register_impl
from tilegym.ops.cutile.rms_norm import rms_norm_kernel_gather_delay_upcast
from tilegym.ops.cutile.utils import next_power_of_2

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def reference_rms_norm(
    input: torch.Tensor,
    normalized_shape: tuple,
    weight: torch.Tensor,
    eps: float,
    bias: torch.Tensor = None,  # Unused - kept for interface compatibility
    static_persistent: bool = False,  # Unused - kept for interface compatibility
):
    """Fused PyTorch RMSNorm baseline using F.rms_norm.

    This is the correct comparison target since it dispatches to a single
    fused kernel (the equivalent of what users would actually replace with a
    custom kernel), rather than the unfused .pow(2).mean().rsqrt() sequence
    that would artificially inflate the performance gap.
    """
    if bias is not None:
        raise NotImplementedError("Bias is not supported in standard CuTile RMSNorm")
    return F.rms_norm(input, normalized_shape, weight=weight, eps=eps)


register_impl("rms_norm", "torch")(reference_rms_norm)


def rms_norm_delay_upcast(
    input: torch.Tensor,
    normalized_shape: tuple,
    weight: torch.Tensor,
    eps: float,
    bias: torch.Tensor = None,
    static_persistent: bool = False,
    offset: float = 0.0,
    **kwargs,
):
    """Launch rms_norm_kernel_gather_delay_upcast directly."""
    x = input.contiguous()
    weight = weight.contiguous()
    x_arg = x.reshape(-1, x.shape[-1])
    M, N = x_arg.shape
    y = torch.empty_like(x_arg)
    rstd = torch.empty(M, dtype=torch.float32, device=x.device)
    EPL = next_power_of_2(N)
    ct.launch(
        torch.cuda.current_stream(),
        (M,),
        rms_norm_kernel_gather_delay_upcast,
        (x_arg, weight, y, rstd, N, eps, offset, EPL),
    )
    return y.view(*x.shape)


register_impl("rms_norm", "cutile_delay_upcast")(rms_norm_delay_upcast)


# Available backends with their display names and plot styles
# Each entry: (key, display_name, style, launch_kwargs)
ALL_BACKENDS = [
    ("cutile_persistent", "CuTile (persistent)", ("blue", "-"), {"backend": "cutile", "static_persistent": True}),
    ("cutile_gather", "CuTile (gather)", ("cyan", "-"), {"backend": "cutile", "static_persistent": False}),
    ("cutile_delay_upcast", "CuTile (delay upcast)", ("red", "-"), {"backend": "cutile_delay_upcast", "static_persistent": False}),
    ("torch", "PyTorch", ("green", "-"), {"backend": "torch", "static_persistent": False}),
]


def get_supported_backends(selected=None):
    """Filter backends based on availability and user selection."""
    available = ALL_BACKENDS if is_backend_available("cutile") else [b for b in ALL_BACKENDS if b[0] == "torch"]
    if selected is not None:
        available = [b for b in available if b[0] in selected]
    return available


DTYPE_MAP = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}

ALL_KERNEL_KEYS = [b[0] for b in ALL_BACKENDS]


def parse_args():
    parser = argparse.ArgumentParser(description="RMSNorm benchmark")
    parser.add_argument(
        "--dtype",
        nargs="+",
        default=["float16", "bfloat16"],
        choices=list(DTYPE_MAP.keys()),
        help="Data types to benchmark (default: float16 bfloat16)",
    )
    parser.add_argument(
        "--kernel",
        nargs="+",
        default=None,
        choices=ALL_KERNEL_KEYS,
        help="Kernels to benchmark (default: all available)",
    )
    return parser.parse_args()


def create_benchmark_config(dtype, backends):
    """Create a benchmark configuration for given parameters"""
    if not backends:
        return None

    keys, names, styles, _ = zip(*backends)
    dtype_name = str(dtype).split(".")[-1]

    return triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[2**i for i in range(10, 15)],  # Hidden size from 1024 to 16384
        line_arg="kernel",
        line_vals=list(keys),
        line_names=list(names),
        styles=list(styles),
        ylabel="GB/s",
        plot_name=f"rmsnorm-performance-{dtype_name}-GBps",
        args={
            "dtype": dtype,
            "M": 4096,
        },  # Fixed batch*seq_len
    )


# Build lookup from kernel key -> launch kwargs
_KERNEL_KWARGS = {b[0]: b[3] for b in ALL_BACKENDS}


def make_bench(configs):
    @triton.testing.perf_report(configs)
    def bench_rmsnorm(N, kernel, dtype, M, device=DEVICE):
        eps = 1e-5

        x_shape = (M, N)
        w_shape = (N,)

        x = torch.rand(x_shape, dtype=dtype, device=device, requires_grad=False).mul_(0.5).add_(-2.3)
        weight = torch.randn(w_shape, dtype=dtype, device=device, requires_grad=False)

        kw = _KERNEL_KWARGS[kernel]
        fn = lambda: tilegym.ops.rms_norm(x, w_shape, weight, eps, **kw)
        ref = lambda: reference_rms_norm(x, w_shape, weight, eps)
        torch.testing.assert_close(fn(), ref(), atol=5e-2, rtol=0.0)

        ms = triton.testing.do_bench_cudagraph(fn)

        bytes_per_element = x.element_size()
        input_bytes = x.numel() * bytes_per_element
        weight_bytes = weight.numel() * bytes_per_element
        output_bytes = x.numel() * bytes_per_element
        total_bytes = input_bytes + weight_bytes + output_bytes

        gb_per_s = total_bytes * 1e-9 / (ms * 1e-3)
        return gb_per_s

    return bench_rmsnorm


if __name__ == "__main__":
    args = parse_args()
    dtypes = list(dict.fromkeys(DTYPE_MAP[d] for d in args.dtype))
    backends = get_supported_backends(selected=args.kernel)
    configs = [create_benchmark_config(dt, backends) for dt in dtypes]
    configs = [c for c in configs if c is not None]
    if not configs:
        print("No valid benchmark configurations. Check --dtype and --kernel args.")
    else:
        make_bench(configs).run(print_data=True)
