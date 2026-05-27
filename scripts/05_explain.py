import os
import json
import random
import cv2
import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1


# Configurações iniciais

TEST_DIR = "./data/processed/test"
MODEL_PATH = "./models/facenet_lfw.pth"
CLASS_NAMES_PATH = "./models/class_names.json"
OUTPUT_DIR = "./outputs/gradcam_examples"

IMG_SIZE = 160
NUM_IMAGES = 10


# Dispositivo

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# Modelo

class FaceNetClassifier(nn.Module):
    """
    Modelo FaceNet + camada final de classificação.
    Tem de ser igual ao modelo usado no treino em scripts/model.py.
    """

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
    """
    Implementação simples de Grad-CAM.

    Guarda:
    - ativações da camada alvo
    - gradientes da classe prevista em relação a essa camada
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.target_layer.register_forward_hook(self.forward_hook)

    def forward_hook(self, module, input, output):
        """
        Hook chamado durante o forward.
        Guarda as ativações e regista outro hook para guardar os gradientes.
        """

        self.activations = output

        def save_gradients(grad):
            self.gradients = grad

        if output.requires_grad:
            output.register_hook(save_gradients)

    def generate(self, input_tensor, target_class):
        """
        Gera o mapa Grad-CAM para uma imagem e uma classe alvo.
        """

        self.model.zero_grad()
        self.gradients = None
        self.activations = None

        output = self.model(input_tensor)

        score = output[0, target_class]
        score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Não foi possível obter gradientes/ativações para o Grad-CAM.")

        gradients = self.gradients[0]
        activations = self.activations[0]

        # Peso de cada canal = média dos gradientes nesse canal
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
    return image, tensor


def denormalize_for_display(tensor):
    """
    Converte tensor normalizado [-1,1] para imagem uint8 [0,255].
    """

    img = tensor.squeeze(0).detach().cpu()
    img = img.permute(1, 2, 0).numpy()

    img = (img * 0.5) + 0.5
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)

    return img


def overlay_heatmap(image_rgb, cam):
    """
    Cria o heatmap colorido e a sobreposição com a imagem original.
    """

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


def save_panel(original_rgb, heatmap_rgb, overlay_rgb, output_path, true_label, pred_label):
    """
    Guarda uma imagem final com três painéis:
    1. original
    2. heatmap Grad-CAM
    3. sobreposição
    """

    h, w, _ = original_rgb.shape
    title_space = 60

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

    text_line = f"True: {true_label} | Pred: {pred_label}"

    cv2.putText(
        panel_bgr,
        text_line,
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 0),
        1
    )

    cv2.imwrite(output_path, panel_bgr)


# Main

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = get_device()
    print(f"Dispositivo usado: {device}")

    class_names = load_class_names()
    num_classes = len(class_names)

    print(f"Número de classes: {num_classes}")

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

    # Camada convolucional tardia do FaceNet.
    # É aqui onde se vão buscar as ativações para o Grad-CAM.
    target_layer = model.backbone.block8

    gradcam = GradCAM(
        model=model,
        target_layer=target_layer
    )

    transform = get_transform()

    image_paths = get_test_images(TEST_DIR)

    if len(image_paths) == 0:
        print("Não foram encontradas imagens de teste.")
        return

    selected_images = random.sample(
        image_paths,
        min(NUM_IMAGES, len(image_paths))
    )

    for i, image_path in enumerate(selected_images):
        print(f"[{i + 1}/{len(selected_images)}] A processar: {image_path}")

        _, input_tensor = preprocess_image(
            image_path,
            transform,
            device
        )

        outputs = model(input_tensor)

        pred_idx = torch.argmax(
            outputs,
            dim=1
        ).item()

        true_label = os.path.basename(
            os.path.dirname(image_path)
        )

        pred_label = class_names[pred_idx]

        cam = gradcam.generate(
            input_tensor,
            pred_idx
        )

        original_rgb = denormalize_for_display(
            input_tensor
        )

        heatmap_rgb, overlay_rgb = overlay_heatmap(
            original_rgb,
            cam
        )

        image_name = os.path.basename(image_path)

        output_name = f"{i + 1:02d}_{true_label}_{image_name}"

        output_path = os.path.join(
            OUTPUT_DIR,
            output_name
        )

        save_panel(
            original_rgb=original_rgb,
            heatmap_rgb=heatmap_rgb,
            overlay_rgb=overlay_rgb,
            output_path=output_path,
            true_label=true_label,
            pred_label=pred_label
        )

        print(f"True: {true_label} | Pred: {pred_label}")
        print(f"Guardado em: {output_path}")

    print("\nGrad-CAM concluído!")
    print(f"Resultados guardados em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()