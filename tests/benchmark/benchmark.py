"""Benchmark gputools vs CPU (numpy/scipy) for common volume processing tasks.

Outputs a markdown table suitable for pasting into the README.

Usage:
    python tests/benchmark/benchmark.py
"""

import platform
import subprocess
from time import time

import numpy as np
import scipy.ndimage as spf
from scipy import ndimage
from skimage.restoration import denoise_nl_means
from skimage.transform import integral_image as sk_integral_image

from gputools import OCLArray, fft, get_device
from gputools.convolve import gaussian_filter, median_filter, uniform_filter
from gputools.convolve.generic_separable_filters import _gauss_filter
from gputools.denoise import nlm3
from gputools.transforms import (
    affine,
    geometric_transform,
    integral_image,
    scale,
)

_TYPE_NAMES = {
    np.uint8: "uint8",
    np.uint16: "uint16",
    np.float32: "float32",
    np.complex64: "complex64",
}


def _get_cpu_name() -> str:
    """Return a short CPU model name."""
    try:
        out = subprocess.check_output(
            ["lscpu"], text=True, stderr=subprocess.DEVNULL
        )
        for line in out.splitlines():
            if "Model name" in line:
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def _get_gpu_name() -> str:
    """Return the OpenCL device name used by gputools."""
    try:
        return get_device().device.name.strip()
    except Exception:
        return "unknown"


def bench(description, dshape, dtype, func_cpu, func_gpu,
          func_gpu_notransfer=None, niter=2):
    """Run a single benchmark and return (t_cpu, t_gpu, t_gpu_notransfer)."""
    x = np.random.randint(0, 100, dshape).astype(dtype)

    # --- CPU ---
    func_cpu(x)
    t = time()
    for _ in range(niter):
        func_cpu(x)
    t_cpu = (time() - t) / niter

    # --- GPU (with transfer) ---
    func_gpu(x)
    t = time()
    for _ in range(niter):
        func_gpu(x)
    t_gpu = (time() - t) / niter

    # --- GPU (no transfer) ---
    if func_gpu_notransfer is not None:
        x_g = OCLArray.from_array(x)
        tmp_g = OCLArray.empty_like(x)
        func_gpu_notransfer(x_g, tmp_g)
        get_device().queue.finish()
        t = time()
        for _ in range(niter):
            func_gpu_notransfer(x_g, tmp_g)
        get_device().queue.finish()
        t_gpu_notransfer = (time() - t) / niter
    else:
        t_gpu_notransfer = None

    return description, dshape, dtype, t_cpu, t_gpu, t_gpu_notransfer


def _fmt_ms(t):
    return f"{1000 * t:.0f} ms" if t is not None else "-"


def print_table(rows, cpu_name, gpu_name):
    """Print results as a markdown table."""
    print()
    print(
        f"Task | Image Size/type | CPU[1] | GPU[2] | GPU (w/o transfer)[3]\n"
        f"----|----| ----| ---- | ----"
    )
    for desc, dshape, dtype, t_cpu, t_gpu, t_gpu_nt in rows:
        size_type = f"{dshape} {_TYPE_NAMES[dtype]}"
        print(
            f"{desc}| {size_type} | {_fmt_ms(t_cpu)} | "
            f"{_fmt_ms(t_gpu)} | {_fmt_ms(t_gpu_nt)}"
        )
    print()
    print(f"\t[1] {cpu_name} using numpy/scipy functions")
    print(f"\t[2] {gpu_name} using gputools")
    print(f"\t[3] as [2] but without CPU->GPU->CPU transfer")
    print()


if __name__ == "__main__":
    cpu_name = _get_cpu_name()
    gpu_name = _get_gpu_name()
    print(f"CPU: {cpu_name}")
    print(f"GPU: {gpu_name}")

    dshape = (128, 1024, 1024)

    # ---- benchmarks shown in README ----
    readme_rows = []

    readme_rows.append(bench(
        "Mean filter 7x7x7", dshape, np.uint8,
        lambda x: spf.uniform_filter(x, 7),
        lambda x: uniform_filter(x, 7),
        lambda x_g, res_g: uniform_filter(x_g, 7, res_g=res_g),
    ))

    readme_rows.append(bench(
        "Median filter 3x3x3", dshape, np.uint8,
        lambda x: spf.median_filter(x, size=3),
        lambda x: median_filter(x, size=3),
        lambda x_g, res_g: median_filter(x_g, size=3, res_g=res_g),
    ))

    readme_rows.append(bench(
        "Gaussian filter 5x5x5", dshape, np.float32,
        lambda x: spf.gaussian_filter(x, 5),
        lambda x: gaussian_filter(x, 5),
        lambda x_g, res_g: gaussian_filter(x_g, 5, res_g=res_g),
    ))

    readme_rows.append(bench(
        "Zoom/Scale 2x2x2", dshape, np.uint8,
        lambda x: ndimage.zoom(x, (2,) * 3, order=1, prefilter=False),
        lambda x: scale(x, (2,) * 3, interpolation="linear"),
    ))

    readme_rows.append(bench(
        "NLM denoising", (64, 256, 256), np.float32,
        lambda x: denoise_nl_means(x, 5, 5, channel_axis=None),
        lambda x: nlm3(x, 0.1, 2, 5),
    ))

    readme_rows.append(bench(
        "FFT (pow2)", dshape, np.complex64,
        lambda x: np.fft.fftn(x),
        lambda x: fft(x),
        lambda x_g, res_g: fft(x_g, inplace=True),
    ))

    print_table(readme_rows, cpu_name, gpu_name)

    # ---- extra benchmarks (not in README) ----
    extra_rows = []

    M = np.random.randn(4, 4)
    M[0] = [0, 0, 0, 1]
    extra_rows.append(bench(
        "Affine", dshape, np.float32,
        lambda x: ndimage.affine_transform(x, M[:3, :3], offset=M[:3, 3]),
        lambda x: affine(x, M),
    ))

    extra_rows.append(bench(
        "Gaussian (separable) 5x5x5", dshape, np.uint16,
        lambda x: _gauss_filter(x, (5, 5, 5)),
        lambda x: _gauss_filter(x, (5, 5, 5)),
        lambda x_g, res_g: _gauss_filter(x_g, (5, 5, 5), res_g=res_g),
    ))

    extra_rows.append(bench(
        "geometric_transform", dshape, np.uint8,
        lambda x: ndimage.geometric_transform(
            x, lambda c: c, output_shape=x.shape, order=1, prefilter=False
        ),
        lambda x: geometric_transform(x, "c0,c1,c2", output_shape=x.shape),
    ))

    extra_rows.append(bench(
        "Integral Image", (512, 1024, 1024), np.float32,
        lambda x: sk_integral_image(x),
        lambda x: integral_image(x),
        lambda x_g, res_g: integral_image(x_g, res_g=res_g.astype(np.float32)),
    ))

    if extra_rows:
        print("Extra benchmarks:")
        print_table(extra_rows, cpu_name, gpu_name)
