import os
import json
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from PIL import Image
from facenet_pytorch import InceptionResnetV1
import mediapipe as mp


# Configurações iniciais

TEST_DIR = "./data/processed/test"
MODEL_PATH = "./models/facenet_lfw.pth"
CLASS_NAMES_PATH = "./models/class_names.json"

OUTPUT_DIR = "./outputs/occlusion_analysis"
CSV_PATH = os.path.join(OUTPUT_DIR, "occlusion_results.csv")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "occlusion_summary.txt")
PLOT_PATH = os.path.join(OUTPUT_DIR, "accuracy_by_condition.png")

IMG_SIZE = 160

MAX_IMAGES = None

OCCLUSION_VALUE = 127  


# Regiões faciais do MediaPipe

FACE_REGIONS = {
    "left_eye": [
        33, 7, 163, 144, 145, 153, 154, 155,
        133, 173, 157, 158, 159, 160, 161, 246
    ],
    "right_eye": [
        362, 382, 381, 380, 374, 373, 390, 249,
        263, 466, 388, 387, 386, 385, 384, 398
    ],
    "nose": [
        1, 2, 98, 327, 168, 195, 197, 5,
        4, 45, 275, 440, 344, 278
    ],
    "mouth": [
        61, 146, 91, 181, 84, 17, 314, 405,
        321, 375, 291, 308, 324, 318, 402, 317
    ],
    "face_oval": [
        10, 338, 297, 332, 284, 251, 389, 356,
        454, 323, 361, 288, 397, 365, 379, 378,
        400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21,
        54, 103, 67, 109
    ]
}


CONDITIONS = [
    "original",
    "mask_eyes",
    "mask_nose",
    "mask_mouth",
    "mask_face_oval",
    "mask_outside_face",
    "dark",
    "bright"
]


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

    image_paths = sorted(image_paths)

    if MAX_IMAGES is not None:
        image_paths = image_paths[:MAX_IMAGES]

    return image_paths


def load_image_rgb(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    return np.array(image)


def rgb_to_tensor(image_rgb, device):
    """
    Converte uma imagem RGB uint8 [0,255] para tensor normalizado [-1,1].
    """

    tensor = torch.from_numpy(image_rgb).float() / 255.0
    tensor = tensor.permute(2, 0, 1)

    tensor = (tensor - 0.5) / 0.5

    tensor = tensor.unsqueeze(0)
    tensor = tensor.to(device)

    return tensor


def predict(model, image_rgb, device):
    tensor = rgb_to_tensor(image_rgb, device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)

        pred_idx = torch.argmax(outputs, dim=1).item()
        confidence = probs[0, pred_idx].item()

    return pred_idx, confidence


def landmarks_to_points(landmarks, image_width, image_height, indices):
    points = []

    for idx in indices:
        landmark = landmarks[idx]

        x = int(landmark.x * image_width)
        y = int(landmark.y * image_height)

        points.append([x, y])

    return np.array(points, dtype=np.int32)


def polygon_to_mask(height, width, points):
    mask = np.zeros((height, width), dtype=np.uint8)

    if len(points) >= 3:
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, hull, 1)

    return mask.astype(bool)


def get_face_masks(image_rgb, face_mesh):
    """
    Cria máscaras para olhos, nariz, boca, oval da face e exterior da face.
    """

    results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        return None

    h, w, _ = image_rgb.shape
    landmarks = results.multi_face_landmarks[0].landmark

    masks = {}

    for region_name, indices in FACE_REGIONS.items():
        points = landmarks_to_points(
            landmarks,
            w,
            h,
            indices
        )

        masks[region_name] = polygon_to_mask(
            h,
            w,
            points
        )

    masks["eyes"] = masks["left_eye"] | masks["right_eye"]
    masks["outside_face"] = ~masks["face_oval"]

    return masks


def apply_condition(image_rgb, masks, condition):
    """
    Aplica uma alteração à imagem:
    - mascarar olhos
    - mascarar nariz
    - mascarar boca
    - mascarar face
    - mascarar exterior da face
    - escurecer
    - clarear
    """

    modified = image_rgb.copy()

    if condition == "original":
        return modified

    if condition == "mask_eyes":
        modified[masks["eyes"]] = OCCLUSION_VALUE
        return modified

    if condition == "mask_nose":
        modified[masks["nose"]] = OCCLUSION_VALUE
        return modified

    if condition == "mask_mouth":
        modified[masks["mouth"]] = OCCLUSION_VALUE
        return modified

    if condition == "mask_face_oval":
        modified[masks["face_oval"]] = OCCLUSION_VALUE
        return modified

    if condition == "mask_outside_face":
        modified[masks["outside_face"]] = OCCLUSION_VALUE
        return modified

    if condition == "dark":
        modified = np.clip(modified.astype(np.float32) * 0.55, 0, 255)
        return modified.astype(np.uint8)

    if condition == "bright":
        modified = np.clip(modified.astype(np.float32) * 1.45, 0, 255)
        return modified.astype(np.uint8)

    raise ValueError(f"Condição desconhecida: {condition}")


def save_summary(df, summary_df):
    original_acc = summary_df.loc["original", "accuracy"]

    lines = []

    lines.append("ANÁLISE DE MASCARAMENTO / OCCLUSION STUDY")
    lines.append("=" * 60)
    lines.append(f"Número total de imagens analisadas: {df['image_path'].nunique()}")
    lines.append(f"Número total de previsões avaliadas: {len(df)}")
    lines.append("")

    lines.append("RESULTADOS POR CONDIÇÃO")
    lines.append("-" * 60)

    for condition, row in summary_df.iterrows():
        acc = row["accuracy"] * 100
        drop = (original_acc - row["accuracy"]) * 100
        avg_conf = row["avg_confidence"] * 100

        lines.append(
            f"{condition}: accuracy={acc:.2f}% | queda_vs_original={drop:.2f} pp | confiança_média={avg_conf:.2f}%"
        )

    lines.append("")
    lines.append("INTERPRETAÇÃO RÁPIDA")
    lines.append("-" * 60)
    lines.append("Se mask_eyes, mask_nose ou mask_mouth provocarem grande queda, essas regiões são importantes para a decisão.")
    lines.append("Se mask_outside_face provocar grande queda, há indícios de dependência de cabelo, fundo, roupa ou iluminação.")
    lines.append("Se mask_face_oval não provocar grande queda, isso é preocupante, porque significa que o modelo pode reconhecer sem informação facial central.")
    lines.append("As condições dark e bright testam sensibilidade a alterações de iluminação.")

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_plot(summary_df):
    plot_df = summary_df.copy()
    plot_df["accuracy_percent"] = plot_df["accuracy"] * 100

    plt.figure(figsize=(10, 5))
    plt.bar(plot_df.index, plot_df["accuracy_percent"])

    plt.ylabel("Accuracy (%)")
    plt.xlabel("Condição")
    plt.title("Accuracy por condição de mascaramento")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()

    plt.savefig(PLOT_PATH, dpi=200)
    plt.close()


# Main

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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

    image_paths = get_test_images(TEST_DIR)

    if len(image_paths) == 0:
        print("Não foram encontradas imagens de teste.")
        return

    print(f"Total de imagens a analisar: {len(image_paths)}")
    print(f"Condições testadas: {', '.join(CONDITIONS)}")

    rows = []

    mp_face_mesh = mp.solutions.face_mesh

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:

        for idx, image_path in enumerate(image_paths, start=1):
            print(f"[{idx}/{len(image_paths)}] {image_path}")

            try:
                image_rgb = load_image_rgb(image_path)

                masks = get_face_masks(
                    image_rgb,
                    face_mesh
                )

                if masks is None:
                    print("  -> Face Mesh não detetou face. A saltar.")
                    continue

                true_label = os.path.basename(
                    os.path.dirname(image_path)
                )

                for condition in CONDITIONS:
                    modified_rgb = apply_condition(
                        image_rgb,
                        masks,
                        condition
                    )

                    pred_idx, confidence = predict(
                        model,
                        modified_rgb,
                        device
                    )

                    pred_label = class_names[pred_idx]
                    correct = true_label == pred_label

                    rows.append({
                        "image_path": image_path,
                        "true_label": true_label,
                        "condition": condition,
                        "pred_label": pred_label,
                        "correct": correct,
                        "confidence": confidence
                    })

            except Exception as e:
                print(f"  -> Erro: {e}")

    if len(rows) == 0:
        print("Nenhum resultado foi produzido.")
        return

    df = pd.DataFrame(rows)

    df.to_csv(
        CSV_PATH,
        index=False,
        encoding="utf-8"
    )

    summary_df = df.groupby("condition").agg(
        accuracy=("correct", "mean"),
        avg_confidence=("confidence", "mean"),
        n=("correct", "count")
    )

    # Manter a ordem lógica das condições
    summary_df = summary_df.loc[CONDITIONS]

    save_summary(df, summary_df)
    save_plot(summary_df)

    print("\nAnálise de mascaramento concluída!")
    print(f"CSV guardado em: {CSV_PATH}")
    print(f"Resumo guardado em: {SUMMARY_PATH}")
    print(f"Gráfico guardado em: {PLOT_PATH}")

    print("\nResumo rápido:")
    print(summary_df)


if __name__ == "__main__":
    main()