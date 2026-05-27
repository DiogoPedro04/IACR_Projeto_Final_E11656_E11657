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
import mediapipe as mp


# Configruações iniciais

TEST_DIR = "./data/processed/test"
MODEL_PATH = "./models/facenet_lfw.pth"
CLASS_NAMES_PATH = "./models/class_names.json"

OUTPUT_DIR = "./outputs/region_analysis"
CSV_PATH = os.path.join(OUTPUT_DIR, "region_attention_results.csv")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "region_attention_summary.txt")

IMG_SIZE = 160

MAX_IMAGES = None


# Regiões faciais

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


# Dispositivo

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


#Modelo

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
            raise RuntimeError("Não foi possível obter gradientes/ativações.")

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


# Auxiliares

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


def get_region_masks(image_rgb, face_mesh):
    results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        return None

    h, w, _ = image_rgb.shape
    landmarks = results.multi_face_landmarks[0].landmark

    face_oval_points = landmarks_to_points(
        landmarks, w, h, FACE_REGIONS["face_oval"]
    )
    face_oval_mask = polygon_to_mask(h, w, face_oval_points)

    masks = {}
    used_mask = np.zeros((h, w), dtype=bool)

    # Regiões prioritárias sem sobreposição
    for region_name in ["left_eye", "right_eye", "nose", "mouth"]:
        points = landmarks_to_points(
            landmarks, w, h, FACE_REGIONS[region_name]
        )
        region_mask = polygon_to_mask(h, w, points)

        # remover sobreposição com regiões anteriores
        region_mask = region_mask & (~used_mask)

        masks[region_name] = region_mask
        used_mask = used_mask | region_mask

    remaining_face = face_oval_mask & (~used_mask)
    outside_face = ~face_oval_mask

    masks["remaining_face"] = remaining_face
    masks["outside_face"] = outside_face

    return masks


def compute_region_attention(cam, masks, output_size):
    cam_resized = cv2.resize(cam, output_size)
    cam_resized = np.maximum(cam_resized, 0)

    total_attention = cam_resized.sum()

    if total_attention <= 1e-8:
        return None

    results = {}

    for region_name, mask in masks.items():
        region_attention = cam_resized[mask].sum()
        results[region_name] = (region_attention / total_attention) * 100.0

    # Métrica extra útil
    results["eyes_total"] = results["left_eye"] + results["right_eye"]
    results["inside_face_total"] = (
        results["left_eye"] +
        results["right_eye"] +
        results["nose"] +
        results["mouth"] +
        results["remaining_face"]
    )

    return results


def save_summary(df, summary_path):
    metric_cols = [
        "left_eye",
        "right_eye",
        "eyes_total",
        "nose",
        "mouth",
        "remaining_face",
        "inside_face_total",
        "outside_face"
    ]

    lines = []

    lines.append("ANÁLISE DE ATENÇÃO POR REGIÃO FACIAL")
    lines.append("=" * 50)
    lines.append(f"Número de imagens analisadas: {len(df)}")
    lines.append(f"Accuracy nesta análise: {df['correct'].mean() * 100:.2f}%")
    lines.append("")

    lines.append("MÉDIAS GLOBAIS (%)")
    lines.append("-" * 50)
    for col in metric_cols:
        lines.append(f"{col}: {df[col].mean():.2f}")

    lines.append("")
    lines.append("MÉDIAS NAS PREVISÕES CORRETAS (%)")
    lines.append("-" * 50)

    df_correct = df[df["correct"] == True]
    if len(df_correct) > 0:
        for col in metric_cols:
            lines.append(f"{col}: {df_correct[col].mean():.2f}")
    else:
        lines.append("Sem previsões corretas.")

    lines.append("")
    lines.append("MÉDIAS NAS PREVISÕES INCORRETAS (%)")
    lines.append("-" * 50)

    df_incorrect = df[df["correct"] == False]
    if len(df_incorrect) > 0:
        for col in metric_cols:
            lines.append(f"{col}: {df_incorrect[col].mean():.2f}")
    else:
        lines.append("Sem previsões incorretas.")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# Main

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = get_device()
    print(f"Dispositivo usado: {device}")

    class_names = load_class_names()
    num_classes = len(class_names)

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    model = FaceNetClassifier(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    target_layer = model.backbone.block8
    gradcam = GradCAM(model, target_layer)

    transform = get_transform()
    image_paths = get_test_images(TEST_DIR)

    if len(image_paths) == 0:
        print("Não foram encontradas imagens.")
        return

    print(f"Total de imagens a analisar: {len(image_paths)}")

    mp_face_mesh = mp.solutions.face_mesh

    rows = []

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:

        for idx, image_path in enumerate(image_paths, start=1):
            print(f"[{idx}/{len(image_paths)}] {image_path}")

            try:
                input_tensor = preprocess_image(image_path, transform, device)

                outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=1)
                pred_idx = torch.argmax(outputs, dim=1).item()
                confidence = probs[0, pred_idx].item()

                true_label = os.path.basename(os.path.dirname(image_path))
                pred_label = class_names[pred_idx]
                correct = (true_label == pred_label)

                cam = gradcam.generate(input_tensor, pred_idx)

                image_rgb = denormalize_for_display(input_tensor)

                masks = get_region_masks(image_rgb, face_mesh)
                if masks is None:
                    print("  -> Face Mesh não detetou face. A saltar.")
                    continue

                attention = compute_region_attention(
                    cam,
                    masks,
                    output_size=(image_rgb.shape[1], image_rgb.shape[0])
                )

                if attention is None:
                    print("  -> Não foi possível calcular atenção.")
                    continue

                row = {
                    "image_path": image_path,
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "correct": correct,
                    "confidence": confidence,
                    "left_eye": attention["left_eye"],
                    "right_eye": attention["right_eye"],
                    "eyes_total": attention["eyes_total"],
                    "nose": attention["nose"],
                    "mouth": attention["mouth"],
                    "remaining_face": attention["remaining_face"],
                    "inside_face_total": attention["inside_face_total"],
                    "outside_face": attention["outside_face"]
                }

                rows.append(row)

            except Exception as e:
                print(f"  -> Erro: {e}")

    if len(rows) == 0:
        print("Nenhum resultado foi produzido.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False, encoding="utf-8")
    save_summary(df, SUMMARY_PATH)

    print("\nAnálise concluída!")
    print(f"CSV guardado em: {CSV_PATH}")
    print(f"Resumo guardado em: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()