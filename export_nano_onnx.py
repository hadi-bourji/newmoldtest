"""
ONNX + TRT export for Jetson Nano.

Exports with decode_in_inference=False so the ONNX graph contains
only convolutions + sigmoid (no grid/meshgrid/arange ops that older
TensorRT versions may mishandle). The grid decode is done in Python
by nano_visualize_predictions.py instead.

Usage:
    python nano_onnx_export.py
"""
import torch
import os
import numpy as np
import tensorrt as trt
from yolox.model import create_yolox_m
from yolox.handle_weights import load_pretrained_weights

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def onnx_export(model, output_path):
    dummy_input = torch.randn(1, 3, 640, 640)
    torch.onnx.export(
        model,
        dummy_input,
        f"{output_path}.onnx",
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
    )
    print(f"Wrote ONNX: {output_path}.onnx")


def build_engine(onnx_path: str, engine_path: str, precision=[]) -> trt.ICudaEngine:
    """Build a TensorRT engine from an ONNX model.
    Default precision is FP32 (empty list) for Nano compatibility.
    Always rebuilds — TRT engines are GPU-specific and stale files cause silent failures.
    """
    logger = TRT_LOGGER
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    success = parser.parse_from_file(onnx_path)
    for idx in range(parser.num_errors):
        print(parser.get_error(idx))
    if not success:
        raise RuntimeError("ONNX parsing failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)  # 256 MiB (Nano-friendly)
    if "fp16" in precision:
        print("Building with FP16")
        config.set_flag(trt.BuilderFlag.FP16)
    if "int8" in precision:
        print("Building with INT8")
        config.set_flag(trt.BuilderFlag.INT8)
    if not precision:
        print("Building with FP32 (default)")

    print("Building TensorRT engine …")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Engine build failed")

    os.makedirs(os.path.dirname(engine_path) or ".", exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    print(f"Wrote TRT engine: {engine_path}")

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    return engine


if __name__ == "__main__":
    num_classes = 8
    weights = "yolox_m_uaFalse_transformsTrue_dn(train.txt)_nc8_ep400_bs128_lr1e-04_wd5e-04_03-09_13.pth"
    onnx_path = "onnx/yolox_m_nano.onnx"
    trt_path  = "onnx/yolox_m_nano.trt"

    model = create_yolox_m(num_classes=num_classes)
    model = load_pretrained_weights(model, weights, remap=False)
    model.eval()

    # KEY DIFFERENCE: disable decode so ONNX/TRT graph is simple
    model.head.decode_in_inference = False

    # Always delete old files to prevent stale exports
    for path in [onnx_path, trt_path]:
        if os.path.exists(path):
            print(f"Deleting old: {path}")
            os.remove(path)

    onnx_export(model, "onnx/yolox_m_nano")

    # Verify ONNX matches PyTorch raw output
    print("\nVerifying ONNX export...")
    import onnxruntime as ort
    dummy = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        pt_raw = model(dummy).numpy()
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_raw = sess.run(None, {"input": dummy.numpy()})[0]
    max_diff = np.abs(pt_raw - onnx_raw).max()
    print(f"  Max |PyTorch - ONNX| = {max_diff:.6f}")
    print(f"  PyTorch raw xy: [{pt_raw[...,0:2].min():.3f}, {pt_raw[...,0:2].max():.3f}]")
    print(f"  ONNX    raw xy: [{onnx_raw[...,0:2].min():.3f}, {onnx_raw[...,0:2].max():.3f}]")
    if max_diff > 0.01:
        print("  ⚠️  Large diff — ONNX export may not match PyTorch!")
    else:
        print("  ✅ ONNX matches PyTorch closely.")

    # Build TRT engine — FP16 for Nano (Tegra X1 has native FP16 at ~2x throughput)
    # ⚠️  TRT engines are GPU-specific. Run this ON the target device!
    build_engine(onnx_path, trt_path, precision=["fp16"])
