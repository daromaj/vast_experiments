#!/usr/bin/env python3
"""Patch load_quantized_dit to build model on meta device to avoid OOM."""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/LongCat-Video/longcat_video/modules/quantization.py"

with open(path, 'r') as f:
    content = f.read()

# Patch: build model on meta device instead of real memory
old = """    # Instantiate model (empty weights)
    model = LongCatVideoAvatarTransformer3DModel(**config)

    # Replace Linear layers with QuantizedLinear (empty)
    skip_patterns = DEFAULT_SKIP_PATTERNS
    modules_to_replace = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            should_skip = any(pattern in name for pattern in skip_patterns)
            if not should_skip:
                ql = QuantizedLinear(module.in_features, module.out_features, bias=module.bias is not None)
                modules_to_replace[name] = ql

    for name, ql in modules_to_replace.items():
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], ql)"""

new = """    # Build model on meta device (zero memory) to avoid fp32 OOM
    with torch.device("meta"):
        model = LongCatVideoAvatarTransformer3DModel(**config)

    # Replace meta Linear layers with real INT8 QuantizedLinear (on CPU)
    skip_patterns = DEFAULT_SKIP_PATTERNS
    import gc
    modules_to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            should_skip = any(pattern in name for pattern in skip_patterns)
            if not should_skip:
                ql = QuantizedLinear(module.in_features, module.out_features, bias=module.bias is not None)
                modules_to_replace.append((name, ql))
            else:
                # Skipped Linear: materialize from meta to CPU, keep original dtype
                for pname, p in list(module.named_parameters(recurse=False)):
                    if p.device.type == "meta":
                        new_p = torch.nn.Parameter(
                            torch.empty(p.shape, device="cpu", dtype=p.dtype),
                            requires_grad=p.requires_grad
                        )
                        setattr(module, pname, new_p)

    for name, ql in modules_to_replace:
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        old_linear = getattr(parent, parts[-1])
        setattr(parent, parts[-1], ql)
        del old_linear
        if len(modules_to_replace) % 50 == 0:
            gc.collect()
    gc.collect()

    # Materialize remaining non-meta parameters to CPU (keep original dtype)
    for mod in model.modules():
        for pname, param in list(mod.named_parameters(recurse=False)):
            if param.device.type == "meta":
                new_p = torch.nn.Parameter(
                    torch.empty(param.shape, device="cpu", dtype=param.dtype),
                    requires_grad=param.requires_grad
                )
                setattr(mod, pname, new_p)
        for bname, buf in list(mod.named_buffers(recurse=False)):
            if buf.device.type == "meta":
                new_buf = torch.empty(buf.shape, device="cpu", dtype=buf.dtype)
                setattr(mod, bname, new_buf)"""

if old in content:
    content = content.replace(old, new)
    with open(path, 'w') as f:
        f.write(content)
    print(f"Patched {path}: meta-device model construction")
else:
    print("ERROR: old pattern not found in file")
    # Show what's actually there
    idx = content.find("Instantiate model")
    if idx > 0:
        print("Found at position", idx)
        print(repr(content[idx:idx+400]))
