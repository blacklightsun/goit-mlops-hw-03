import onnxruntime as ort
from PIL import Image
import numpy as np

import sys
import pathlib


def predict(image_path):
    # Шлях до файлу з назвами класів
    class_file = pathlib.Path("model/class_names.txt")

    # Створюємо порожній список для зберігання назв класів
    class_names = []

    # Відкриваємо файл для читання
    with open(class_file, "r", encoding="utf-8") as f:
        # Читаємо кожен рядок, видаляємо зайві пробіли і символ нового рядка (\n)
        # та додаємо чисту назву до нашого списку
        class_names = [line.strip() for line in f.readlines()]

    # Завантажуємо модель
    session = ort.InferenceSession("model/model.onnx")

    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    input_array = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
    input_array = np.expand_dims(input_array, axis=0)  # batch_size=1

    outputs = session.run(None, {"input": input_array})
    preds = outputs[0][0]  # вихідний вектор для першого (єдиного) зображення

    # Топ-3 прогнозів
    top3_idx = preds.argsort()[-3:][::-1]  # індекси топ-3 класів

    # softmax для ймовірностей
    probs = np.exp(preds - np.max(preds))
    probs /= probs.sum()

    top3_scores = probs[top3_idx]

    print("🧠 Top-3 predictions:")
    for i, (idx, score) in enumerate(zip(top3_idx, top3_scores), start=1):
        print(
            f"🧠 Top-{i} class-> ID: {idx.item()}, class name: {class_names[idx.item()]}, probability: {score.item():.1%}"
        )


if __name__ == "__main__":
    predict(sys.argv[1])
