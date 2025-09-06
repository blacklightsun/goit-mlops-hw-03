import os
import torch
import torchvision.models as models

# Створення папки для збереження моделі
os.makedirs("model", exist_ok=True)

weights = models.MobileNet_V2_Weights.DEFAULT
class_names = weights.meta["categories"]

# Завантаження попередньо навченої моделі MobileNetV2
model = models.mobilenet_v2(weights=weights)

model.eval()  # Перехід у режим оцінки (inference)

# Створення "манекена" для трасування моделі
dummy_input = torch.rand(1, 3, 224, 224)

# Трасування моделі в TorchScript
traced_model = torch.jit.trace(model, dummy_input)

# Збереження моделі
traced_model.save("model/traced_model.pt")
print("✅ Model saved to model/traced_model.pt")

# Збереження імен класів у текстовий файл
with open("model/class_names.txt", "w", encoding="utf-8") as f:
    for class_name in class_names:
        f.write(f"{class_name}\n")
print("✅ Class names saved to model/class_names.txt")
