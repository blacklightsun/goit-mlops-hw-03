import torch

# Завантажуємо TorchScript модель
ts_model = torch.jit.load("model/traced_model.pt")
ts_model.eval()

# Тензор прикладу для форварду
dummy_input = torch.randn(1, 3, 224, 224)

# Експортуємо в ONNX
torch.onnx.export(
    ts_model,
    dummy_input,
    "model/model.onnx",
    opset_version=17,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)
