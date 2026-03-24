import torch
import os
import tensorrt as trt
from yolox.model import create_yolox_s, create_yolox_m
from yolox.handle_weights import load_pretrained_weights

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def onnx_export(model, output_path):
    dummy_input = torch.randn(1, 3, 640, 640)
    torch.onnx.export(
        model,
        dummy_input,
        f"{output_path}.onnx",
        opset_version=18,
        do_constant_folding=False,
        input_names=['input'],
        output_names=['output'],
    )
    print(f"Wrote ONNX: {output_path}.onnx")


def build_engine(onnx_path: str, engine_path: str, precision=["fp16"]) -> trt.ICudaEngine:
    """Build (or load) a TensorRT engine from an ONNX model."""
    if os.path.exists(engine_path):
        print(f"Loading existing TRT engine: {engine_path}")
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as rt:
            return rt.deserialize_cuda_engine(f.read())

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
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GiB
    if "fp16" in precision:
        print("Building with FP16")
        config.set_flag(trt.BuilderFlag.FP16)
    if "int8" in precision:
        print("Building with INT8")
        config.set_flag(trt.BuilderFlag.INT8)

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
    weights = "model_checkpoints/yolox_m_uaFalse_transformsTrue_dn(train.txt)_nc8_ep400_bs128_lr1e-04_wd5e-04_03-09_13.pth"
    onnx_path = "onnx/yolox_m_best.onnx"
    trt_path  = "onnx/yolox_m_best.trt"

    model = create_yolox_m(num_classes=num_classes)
    model = load_pretrained_weights(model, weights, remap=False)
    model.eval()

    if not os.path.exists(onnx_path):
        onnx_export(model, "onnx/yolox_m_best")
    else:
        print(f"ONNX already exists: {onnx_path}")

    # Build TRT engine from ONNX
    if not os.path.exists(trt_path):
        build_engine(onnx_path, trt_path, precision=["fp16"])
    else:
        print(f"TRT engine already exists: {trt_path}")
