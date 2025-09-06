import sys
import pathlib

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Шлях до файлу з назвами класів
class_file = pathlib.Path("model/class_names.txt")

# Створюємо порожній список для зберігання назв класів
class_names = []

# Відкриваємо файл для читання
with open(class_file, "r", encoding="utf-8") as f:
    # Читаємо кожен рядок, видаляємо зайві пробіли і символ нового рядка (\n)
    # та додаємо чисту назву до нашого списку
    class_names = [line.strip() for line in f.readlines()]

# Завантаження збереженої моделі
model = torch.jit.load("model/traced_model.pt")
model.eval()

preprocess = transforms.Compose(
    [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
)

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        _, indices = torch.topk(output, 3)
        probabilities = F.softmax(output, dim=1)

        print("🧠 Top-3 predictions:")
        for i in range(3):
            print(
                f"🧠 Top-{i+1} class-> ID: {indices[0][i].item()}, class name: {class_names[indices[0][i].item()]}, probability: {probabilities[0][indices[0][i].item()].item():.1%}"
            )

if __name__ == "__main__":
    predict(sys.argv[1])
