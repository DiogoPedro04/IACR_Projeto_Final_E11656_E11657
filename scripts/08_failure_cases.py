import os
import json
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1


# Configurações iniciais

TEST_DIR = "./data/processed/test"
MODEL_PATH = "./models/facenet_lfw.pth"
CLASS_NAMES_PATH = "./models/class_names.json"

OUTPUT_DIR = "./outputs/failure_cases"
EXAMPLES_DIR = os.path.join(OUTPUT_DIR, "examples")
CSV_PATH = os.path.join(OUTPUT_DIR, "failure_cases.csv")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "failure_summary.txt")

IMG_SIZE = 160

MAX_FAILURE_EXAMPLES = 40


# Dispositivo

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# Modelo

class FaceNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(FaceNetClassifier, self).__init__()

        self.backbone = InceptionResnetV1(
            pretrained="vggface2",
            classify=False
        )

        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        embeddings = self.backbone(x)
        outputs = self.classifier(embeddings)
        return outputs


# Grad-CAM

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.target_layer.register_forward_hook(self.forward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output

        def save_gradients(grad):
            self.gradients = grad

        if output.requires_grad:
            output.register_hook(save_gradients)

    def generate(self, input_tensor, target_class):
        self.model.zero_grad()
        self.gradients = None
        self.activations = None

        output = self.model(input_tensor)
        score = output[0, target_class]
        score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Não foi possível obter gradientes/ativações para Grad-CAM.")

        gradients = self.gradients[0]
        activations = self.activations[0]

        weights = gradients.mean(dim=(1, 2))

        cam = torch.zeros(
            activations.shape[1:],
            dtype=torch.float32,
            device=activations.device
        )

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.relu(cam)

        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.detach().cpu().numpy()


# Funções auxiliares

def load_class_names():
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_test_images(test_dir):
    image_paths = []

    for person in os.listdir(test_dir):
        person_dir = os.path.join(test_dir, person)

        if not os.path.isdir(person_dir):
            continue

        for filename in os.listdir(person_dir):
            if filename.lower().endswith(".jpg"):
                image_paths.append(os.path.join(person_dir, filename))

    return sorted(image_paths)


def get_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])


def preprocess_image(image_path, transform, device):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    return tensor


def denormalize_for_display(tensor):
    img = tensor.squeeze(0).detach().cpu()
    img = img.permute(1, 2, 0).numpy()

    img = (img * 0.5) + 0.5
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)

    return img


def overlay_heatmap(image_rgb, cam):
    heatmap = cv2.resize(cam, (image_rgb.shape[1], image_rgb.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap)

    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    heatmap_color = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        image_rgb,
        0.6,
        heatmap_color,
        0.4,
        0
    )

    return heatmap_color, overlay


def save_failure_panel(original_rgb, heatmap_rgb, overlay_rgb, output_path, true_label, pred_label, confidence):
    h, w, _ = original_rgb.shape
    title_space = 80

    panel = np.ones((h + title_space, w * 3, 3), dtype=np.uint8) * 255

    panel[title_space:title_space + h, 0:w] = original_rgb
    panel[title_space:title_space + h, w:2 * w] = heatmap_rgb
    panel[title_space:title_space + h, 2 * w:3 * w] = overlay_rgb

    panel_bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)

    cv2.putText(
        panel_bgr,
        "Original",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2
    )

    cv2.putText(
        panel_bgr,
        "Grad-CAM",
        (w + 10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2
    )

    cv2.putText(
        panel_bgr,
        "Overlay",
        (2 * w + 10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2
    )

    line1 = f"True: {true_label}"
    line2 = f"Pred: {pred_label} | Confidence: {confidence * 100:.2f}%"

    cv2.putText(
        panel_bgr,
        line1,
        (10, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 0),
        1
    )

    cv2.putText(
        panel_bgr,
        line2,
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 0),
        1
    )

    cv2.imwrite(output_path, panel_bgr)


def save_summary(df_all, df_failures):
    total = len(df_all)
    correct = int(df_all["correct"].sum())
    incorrect = len(df_failures)

    accuracy = correct / total * 100
    error_rate = incorrect / total * 100

    lines = []

    lines.append("ANÁLISE DE CASOS DE FALHA")
    lines.append("=" * 60)
    lines.append(f"Número total de imagens avaliadas: {total}")
    lines.append(f"Previsões corretas: {correct}")
    lines.append(f"Previsões incorretas: {incorrect}")
    lines.append(f"Accuracy: {accuracy:.2f}%")
    lines.append(f"Taxa de erro: {error_rate:.2f}%")
    lines.append("")

    lines.append("CONFIANÇA NAS FALHAS")
    lines.append("-" * 60)

    if len(df_failures) > 0:
        lines.append(f"Confiança média nas falhas: {df_failures['confidence'].mean() * 100:.2f}%")
        lines.append(f"Confiança máxima numa falha: {df_failures['confidence'].max() * 100:.2f}%")
        lines.append(f"Confiança mínima numa falha: {df_failures['confidence'].min() * 100:.2f}%")
    else:
        lines.append("Não houve falhas.")
        with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return

    lines.append("")
    lines.append("PARES DE CONFUSÃO MAIS FREQUENTES")
    lines.append("-" * 60)

    confusion_pairs = (
        df_failures
        .groupby(["true_label", "pred_label"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(15)
    )

    for _, row in confusion_pairs.iterrows():
        lines.append(
            f"{row['true_label']} -> {row['pred_label']}: {row['count']} vez(es)"
        )

    lines.append("")
    lines.append("FALHAS COM MAIOR CONFIANÇA")
    lines.append("-" * 60)

    high_conf_failures = df_failures.sort_values("confidence", ascending=False).head(15)

    for _, row in high_conf_failures.iterrows():
        lines.append(
            f"{row['true_label']} -> {row['pred_label']} | confiança={row['confidence'] * 100:.2f}% | imagem={row['image_path']}"
        )

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# Main

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(EXAMPLES_DIR, exist_ok=True)

    device = get_device()
    print(f"Dispositivo usado: {device}")

    class_names = load_class_names()
    num_classes = len(class_names)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model = FaceNetClassifier(num_classes=num_classes)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    target_layer = model.backbone.block8
    gradcam = GradCAM(model, target_layer)

    transform = get_transform()
    image_paths = get_test_images(TEST_DIR)

    print(f"Total de imagens a avaliar: {len(image_paths)}")

    rows = []
    failure_examples_saved = 0

    for idx, image_path in enumerate(image_paths, start=1):
        print(f"[{idx}/{len(image_paths)}] {image_path}")

        try:
            input_tensor = preprocess_image(
                image_path,
                transform,
                device
            )

            # Não usar torch.no_grad(), porque podemos precisar de Grad-CAM nas falhas
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)

            pred_idx = torch.argmax(outputs, dim=1).item()
            confidence = probs[0, pred_idx].item()

            true_label = os.path.basename(os.path.dirname(image_path))
            pred_label = class_names[pred_idx]

            correct = true_label == pred_label

            rows.append({
                "image_path": image_path,
                "true_label": true_label,
                "pred_label": pred_label,
                "correct": correct,
                "confidence": confidence
            })

            if not correct and failure_examples_saved < MAX_FAILURE_EXAMPLES:
                cam = gradcam.generate(input_tensor, pred_idx)

                original_rgb = denormalize_for_display(input_tensor)
                heatmap_rgb, overlay_rgb = overlay_heatmap(original_rgb, cam)

                image_name = os.path.basename(image_path)

                output_name = (
                    f"{failure_examples_saved + 1:02d}_"
                    f"TRUE_{true_label}_PRED_{pred_label}_"
                    f"{image_name}"
                )

                # Evitar problemas com caracteres estranhos no nome do ficheiro
                output_name = output_name.replace("/", "_").replace("\\", "_")

                output_path = os.path.join(EXAMPLES_DIR, output_name)

                save_failure_panel(
                    original_rgb=original_rgb,
                    heatmap_rgb=heatmap_rgb,
                    overlay_rgb=overlay_rgb,
                    output_path=output_path,
                    true_label=true_label,
                    pred_label=pred_label,
                    confidence=confidence
                )

                failure_examples_saved += 1

        except Exception as e:
            print(f"  -> Erro: {e}")

    if len(rows) == 0:
        print("Nenhum resultado produzido.")
        return

    df_all = pd.DataFrame(rows)
    df_failures = df_all[df_all["correct"] == False].copy()

    df_failures.to_csv(
        CSV_PATH,
        index=False,
        encoding="utf-8"
    )

    save_summary(
        df_all=df_all,
        df_failures=df_failures
    )

    print("\nAnálise de falhas concluída!")
    print(f"Total de imagens avaliadas: {len(df_all)}")
    print(f"Número de falhas: {len(df_failures)}")
    print(f"CSV de falhas guardado em: {CSV_PATH}")
    print(f"Resumo guardado em: {SUMMARY_PATH}")
    print(f"Exemplos visuais guardados em: {EXAMPLES_DIR}")


if __name__ == "__main__":
    main()