import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from src.utils.logging import get_logger
from src.utils.config import load_config
from src.modeling.segmentation import LightResUNet_MSA

logger = get_logger("inference.export_onnx")

def export_to_onnx(model: nn.Module,
                   output_path: str,
                   input_shape: tuple = (1, 1, 512, 512),
                   dynamic_axes: Optional[Dict[str, Any]] = None,
                   opset_version: int = 17,
                   simplify: bool = True) -> str:
    model.eval()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dummy_input = torch.randn(input_shape)
    if dynamic_axes is None:
        dynamic_axes = {
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size", 2: "height", 3: "width"}
        }
    with torch.no_grad():
        original_training = model.training
        model.eval()
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            verbose=False
        )
        model.train(original_training)
    logger.info(f"Exported ONNX model to: {output_path}")
    if simplify:
        try:
            import onnx
            from onnxsim import simplify
            onnx_model = onnx.load(output_path)
            onnx_simplified, check = simplify(
                onnx_model,
                input_shapes={"input": input_shape},
                dynamic_input_shape=True
            )
            if check:
                onnx.save(onnx_simplified, output_path)
                logger.info(f"Simplified ONNX model saved")
            else:
                logger.warning("ONNX simplification check failed, using original model")
        except ImportError:
            logger.warning("onnxsim not available, skipping model simplification")
        except Exception as e:
            logger.warning(f"ONNX simplification failed: {e}")
    return output_path

def load_pytorch_model(checkpoint_path: str,
                       config_path: str = "configs/config.yaml",
                       device: str = "cpu") -> LightResUNet_MSA:
    config = load_config(config_path)
    model = LightResUNet_MSA(
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        base_channels=config["model"]["base_channels"],
        num_res_blocks=config["model"]["num_res_blocks"],
        attention_heads=config["model"]["attention_heads"],
        attention_channels=config["model"]["attention_channels"],
        dropout_rate=0.0,
        use_deep_supervision=False
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    new_state_dict = {}
    for k, v in state_dict.items():
        if "deep_supervision" in k:
            continue
        new_state_dict[k] = v
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    if missing_keys:
        logger.warning(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys: {unexpected_keys}")
    model.eval()
    logger.info(f"Loaded PyTorch model from: {checkpoint_path}")
    return model

def main():
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to PyTorch checkpoint")
    parser.add_argument("--output", type=str, default="models/model.onnx",
                       help="Output ONNX model path")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to config file")
    parser.add_argument("--input_size", type=int, nargs=2, default=[512, 512],
                       help="Input image size (H W)")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for dummy input")
    parser.add_argument("--opset", type=int, default=17,
                       help="ONNX opset version")
    parser.add_argument("--no_simplify", action="store_true",
                       help="Disable ONNX simplification")
    parser.add_argument("--verify", action="store_true",
                       help="Verify ONNX model against PyTorch")
    args = parser.parse_args()
    device = "cpu"
    model = load_pytorch_model(args.checkpoint, args.config, device)
    input_shape = (args.batch_size, 1, args.input_size[0], args.input_size[1])
    output_path = export_to_onnx(
        model, args.output, input_shape,
        opset_version=args.opset,
        simplify=not args.no_simplify
    )
    if args.verify:
        try:
            import onnxruntime as ort
            dummy_input = torch.randn(input_shape)
            with torch.no_grad():
                torch_output = model(dummy_input)
                if isinstance(torch_output, list):
                    torch_output = torch_output[-1]
            sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
            ort_output = sess.run(None, {"input": dummy_input.numpy()})[0]
            max_diff = np.max(np.abs(torch_output.numpy() - ort_output))
            mean_diff = np.mean(np.abs(torch_output.numpy() - ort_output))
            logger.info(f"Verification: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}")
            if max_diff < 1e-3:
                logger.info("Verification PASSED: outputs match")
            else:
                logger.warning("Verification FAILED: outputs differ significantly")
        except Exception as e:
            logger.warning(f"Verification failed: {e}")
    logger.info(f"ONNX export complete: {output_path}")
    return output_path

if __name__ == "__main__":
    main()
