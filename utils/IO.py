import torch
import onnx
from onnx2torch import convert

def import_model(file_path):

    # Convert to PyTorch
    onnx_model = convert(onnx.load(file_path))

    return onnx_model

def export_model(model, input_shape, file_path):

    dummy_input = torch.randn(*input_shape)

    torch.onnx.export(model,
        dummy_input,
        file_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={ 'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}})

def export_solver_results(result, x_max, y_max, file_name):
    with open(file_name, "w") as f:
        f.write(result + "\n")
        if x_max is not None:
            f.write("(")
            for idx, value in enumerate(x_max):
                f.write(f"(X_{idx} {value})\n")
            for idx, value in enumerate(y_max):
                f.write(f"(Y_{idx} {value})\n")
            f.write(")")
