import onnx
import torch
from torch import nn


def export_to_onnx(model: nn.Module, input_size, save_path: str):
    model.eval()
    dummy_input = torch.randint(1, 10, size=input_size)
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        input_names=["input_seq"],
        output_names=["output_seq"],
        opset_version=13,
        dynamic_axes={
            "input_seq": {0: "batch_size"},
            "output_seq": {0: "batch_size"},
        },
    )
    print(f"[✓] Exported to ONNX: {save_path}")


def verify_onnx(save_path: str):
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    print(f"[✓] ONNX model is valid: {save_path}")
    return onnx_model


def export_to_trt(onnx_path: str, trt_path: str):
    try:
        import tensorrt as trt
    except ImportError:
        raise ImportError("TensorRT must be installed.")

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        flags=1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX model.")

    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1 GB

    engine = builder.build_engine(network, config)

    with open(trt_path, "wb") as f:
        f.write(engine.serialize())
    print(f"[✓] Exported to TensorRT: {trt_path}")
