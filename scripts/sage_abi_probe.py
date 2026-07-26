#!/usr/bin/env python3
"""
SageAttention wheel viability probe — run ON the Vast.ai instance.

Answers one question: can we skip the ~6 minute source build and just install a
prebuilt wheel? The wheel path was abandoned as "unreliable" without a recorded
failure mode, so this checks all three ways a compiled CUDA extension can fail:

  1. ABI break      -> import raises `undefined symbol: _ZN3c10...`
                       (wheel linked against a different libtorch than this image ships)
  2. Arch mismatch  -> call raises `no kernel image is available for execution`
                       (wheel compiled for sm_120, host is sm_89, or vice versa)
  3. Silent garbage -> import and call both succeed, output is NaN/wrong
                       (the dangerous one — you only notice it in the video)

It also dumps the exact ABI triple (torch / cuda / cp tag / sm arch) so a wheel
that DOES work can be committed under an ABI-keyed name and matched by filename
on the next instance instead of guessed at.

Usage on the instance:

    source /venv/main/bin/activate
    python3 sage_abi_probe.py                    # probe whatever is installed
    python3 sage_abi_probe.py --install <wheel>  # install that wheel first, then probe
    python3 sage_abi_probe.py --json             # machine-readable, for pasting back

Exit code is 0 only if a working SageAttention is present. Anything else means
fall back to the source build.
"""
import argparse
import json
import os
import subprocess
import sys
import traceback

# Sage is lossy by design (int8/fp8 quantized QK^T). It will NOT match the fp16
# reference closely, so we assert on direction, not magnitude. A correct sage
# kernel lands ~0.99+ cosine similarity vs SDPA; a broken one lands near 0 or NaN.
MIN_COSINE_SIMILARITY = 0.98


def _run(cmd):
    """Run a command, return (rc, stdout+stderr). Never raises."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True)
        return p.returncode, (p.stdout + p.stderr).strip()
    except Exception as e:  # noqa: BLE001 - probe must never die on a subprocess
        return -1, str(e)


def collect_environment():
    """The ABI triple. This is what a wheel must match to be loadable here."""
    env = {}

    env["python"] = ".".join(str(v) for v in sys.version_info[:3])
    env["cp_tag"] = f"cp{sys.version_info.major}{sys.version_info.minor}"

    try:
        import torch

        env["torch"] = torch.__version__
        env["torch_cuda"] = torch.version.cuda
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["gpu_name"] = torch.cuda.get_device_name(0)
            major, minor = torch.cuda.get_device_capability(0)
            env["sm_arch"] = f"{major}.{minor}"
            env["sm_tag"] = f"sm_{major}{minor}"
    except Exception as e:  # noqa: BLE001
        env["torch_import_error"] = repr(e)

    rc, out = _run(["nvcc", "--version"])
    env["nvcc"] = out.splitlines()[-1] if rc == 0 and out else "not found"

    return env


def check_memory_vs_sdpa(mod, entrypoint, sm_tag):
    """
    Failure mode 4: the wheel imports, runs, and returns CORRECT results - but is
    pathologically memory-hungry on this GPU, which only shows up as an OOM deep
    inside a real workflow.

    Do NOT try to detect this by looking for an arch-named extension. An earlier
    version of this probe required a _qattn_sm120*.so and reported MISSING for a
    wheel built from source on the very GPU in question: SageAttention 2.2.0 has
    no per-arch module, it compiles the sm80/sm89 kernel SOURCES for whatever
    TORCH_CUDA_ARCH_LIST is set to. The filenames say nothing about the target.

    Measure behaviour instead. At WanVideo's real attention shape a healthy sage
    build is FASTER than SDPA and within ~1.5x its peak memory (measured on a
    5090: sdpa 106.9ms/1.57GiB vs sageattn 39.1ms/2.34GiB). A build that falls
    back materialises far more than that.
    """
    import torch
    import torch.nn.functional as F

    result = {"sm_tag": sm_tag}
    fn = getattr(mod, entrypoint, None)
    if fn is None:
        result["error"] = f"{entrypoint} not found"
        return result

    # Big enough to expose a fallback, small enough to be safe beside other work.
    heads, seq, dim = 16, 8192, 128
    q, k, v = (
        torch.randn(1, heads, seq, dim, device="cuda", dtype=torch.float16)
        for _ in range(3)
    )

    def peak_of(call):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            call()
            torch.cuda.synchronize()
            return torch.cuda.max_memory_allocated() / 2**30, None
        except Exception as e:  # noqa: BLE001
            return None, repr(e)

    sdpa_peak, sdpa_err = peak_of(lambda: F.scaled_dot_product_attention(q, k, v))
    try:
        sage_peak, sage_err = peak_of(lambda: fn(q, k, v, tensor_layout="HND"))
    except TypeError:
        sage_peak, sage_err = peak_of(lambda: fn(q, k, v))

    result.update({"sdpa_peak_gib": sdpa_peak, "sage_peak_gib": sage_peak,
                   "sdpa_error": sdpa_err, "sage_error": sage_err})

    if sdpa_peak and sage_peak:
        ratio = sage_peak / sdpa_peak
        result["peak_ratio"] = ratio
        # 1.5x is normal for sage's quantisation buffers; 3x means it fell back.
        result["memory_sane"] = ratio <= 3.0
        if not result["memory_sane"]:
            result["warning"] = (
                f"SageAttention peaks at {ratio:.1f}x SDPA on this GPU - it is "
                f"falling back rather than using a native kernel. Expect OOM in "
                f"real workflows even though output is numerically correct."
            )
    else:
        result["memory_sane"] = True  # could not measure; do not fail the wheel
    return result


def check_import(module_name):
    """
    Failure mode 1: ABI break. An `undefined symbol` here means the wheel was
    linked against a different libtorch than this base image ships.
    """
    result = {"module": module_name, "imported": False}
    try:
        mod = __import__(module_name)
        result["imported"] = True
        result["version"] = getattr(mod, "__version__", "unknown")
        result["callables"] = sorted(
            n for n in dir(mod) if callable(getattr(mod, n, None)) and not n.startswith("_")
        )
        return result, mod
    except Exception as e:  # noqa: BLE001
        result["error"] = repr(e)
        result["is_abi_break"] = "undefined symbol" in str(e)
        result["traceback"] = traceback.format_exc()
        return result, None


def check_kernel(mod, entrypoint):
    """
    Failure modes 2 and 3: run a real attention call on the GPU and compare it
    against PyTorch SDPA. Catches both `no kernel image` and silent NaN/garbage.
    """
    import torch
    import torch.nn.functional as F

    result = {"entrypoint": entrypoint, "ran": False}

    fn = getattr(mod, entrypoint, None)
    if fn is None:
        result["error"] = f"{entrypoint} not found on module"
        return result

    # Shapes chosen to look like a real WanVideo attention block, not a toy.
    batch, heads, seq, dim = 1, 12, 1024, 128
    torch.manual_seed(0)
    q, k, v = (
        torch.randn(batch, heads, seq, dim, device="cuda", dtype=torch.float16)
        for _ in range(3)
    )

    try:
        out = fn(q, k, v, tensor_layout="HND", is_causal=False)
    except TypeError:
        # Older/newer signatures drop the kwargs. Retry positionally.
        try:
            out = fn(q, k, v)
        except Exception as e:  # noqa: BLE001
            result["error"] = repr(e)
            result["is_arch_mismatch"] = "no kernel image" in str(e)
            return result
    except Exception as e:  # noqa: BLE001
        result["error"] = repr(e)
        result["is_arch_mismatch"] = "no kernel image" in str(e)
        return result

    torch.cuda.synchronize()
    result["ran"] = True
    result["output_shape"] = list(out.shape)
    result["has_nan"] = bool(torch.isnan(out).any().item())
    result["has_inf"] = bool(torch.isinf(out).any().item())
    result["all_zero"] = bool((out == 0).all().item())

    reference = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    cos = F.cosine_similarity(
        out.float().flatten(), reference.float().flatten(), dim=0
    ).item()
    result["cosine_vs_sdpa"] = cos
    result["max_abs_err"] = (out.float() - reference.float()).abs().max().item()

    result["usable"] = (
        not result["has_nan"]
        and not result["has_inf"]
        and not result["all_zero"]
        and cos >= MIN_COSINE_SIMILARITY
    )
    if not result["usable"] and cos < MIN_COSINE_SIMILARITY:
        # This is the failure that never shows up in a log — record it loudly.
        result["is_silent_garbage"] = True

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", metavar="WHEEL", help="pip install this wheel before probing")
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    args = parser.parse_args()

    report = {"environment": collect_environment()}

    if args.install:
        rc, out = _run([sys.executable, "-m", "pip", "install", "--no-cache-dir", args.install])
        report["install"] = {"wheel": args.install, "returncode": rc, "output": out[-2000:]}

    # sageattention == 2.x (the one both provisioning scripts build from source).
    # sageattn3 is the separate Blackwell-only fp4 package; probe it too if present.
    report["modules"] = {}
    ok = False
    arch_ok = True
    for module_name, entrypoint in (("sageattention", "sageattn"), ("sageattn3", "sageattn3")):
        import_result, mod = check_import(module_name)
        entry = {"import": import_result}
        if module_name == "sageattention" and mod is not None and report[
                "environment"].get("cuda_available"):
            entry["memory"] = check_memory_vs_sdpa(
                mod, "sageattn", report["environment"].get("sm_tag", ""))
            arch_ok = entry["memory"].get("memory_sane", True)
        if mod is not None and report["environment"].get("cuda_available"):
            # Prefer the documented entrypoint; fall back to whatever the module exposes.
            candidates = [entrypoint] + [
                c for c in import_result.get("callables", []) if c.startswith("sageattn")
            ]
            for candidate in dict.fromkeys(candidates):
                kernel_result = check_kernel(mod, candidate)
                entry.setdefault("kernel", []).append(kernel_result)
                if kernel_result.get("usable"):
                    ok = True
                    break
        report["modules"][module_name] = entry

    # A wheel without a kernel for this GPU is worse than no wheel: it passes
    # every correctness check and then quietly blows the VRAM budget. Treat a
    # missing arch kernel as a reason to build from source.
    if ok and not arch_ok:
        report["verdict"] = "FALL_BACK_TO_SOURCE_BUILD"
        report["reason"] = "correct output but pathological VRAM vs SDPA"
        ok = False
    else:
        report["verdict"] = "WHEEL_USABLE" if ok else "FALL_BACK_TO_SOURCE_BUILD"

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(json.dumps(report, indent=2, default=str))
        env = report["environment"]
        print("\n" + "=" * 60)
        print(f"VERDICT: {report['verdict']}")
        if report.get("reason"):
            print(f"REASON:  {report['reason']}")
        memck = (report["modules"].get("sageattention") or {}).get("memory") or {}
        if memck.get("peak_ratio"):
            print(f"Peak VRAM vs SDPA: {memck['peak_ratio']:.2f}x "
                  f"(sage {memck['sage_peak_gib']:.2f} GiB / "
                  f"sdpa {memck['sdpa_peak_gib']:.2f} GiB)")
        print(
            "ABI triple: torch={t} cuda={c} {cp} {sm}".format(
                t=env.get("torch", "?"),
                c=env.get("torch_cuda", "?"),
                cp=env.get("cp_tag", "?"),
                sm=env.get("sm_tag", "?"),
            )
        )
        # The ABI goes in the PATH, not the filename: PEP 427 requires a build tag
        # to start with a digit, so "...-torch2.10.0-..." in the name is rejected
        # by pip. Keep the wheel's own name intact and key the directory.
        print("If usable, commit the wheel as:")
        print(
            "  python/sage/torch{t}-cu{c}-{sm}/<original-wheel-name>.whl".format(
                t=str(env.get("torch", "unknown")).split("+")[0],
                c=str(env.get("torch_cuda", "unknown")).replace(".", ""),
                sm=env.get("sm_tag", "unknown"),
            )
        )
        print("=" * 60)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
