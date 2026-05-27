import os
import random
import cv2
import numpy as np
import mediapipe as mp


# Configurações iniciais

TEST_DIR = "./data/processed/test"
OUTPUT_DIR = "./outputs/segmentation_examples"

NUM_IMAGES = 10


# Regiões Faciais do MediaPipe

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


REGION_COLORS = {
    "left_eye": (255, 0, 0),      # azul
    "right_eye": (0, 255, 0),     # verde
    "nose": (0, 255, 255),        # amarelo
    "mouth": (0, 0, 255),         # vermelho
    "face_oval": (255, 0, 255)    # magenta
}


# Funções auxiliares

def get_test_images(test_dir):
    """
    Vai buscar imagens aleatórias ao conjunto de teste.
    """

    image_paths = []

    for person in os.listdir(test_dir):
        person_dir = os.path.join(test_dir, person)

        if not os.path.isdir(person_dir):
            continue

        for filename in os.listdir(person_dir):
            if filename.lower().endswith(".jpg"):
                image_paths.append(os.path.join(person_dir, filename))

    return image_paths


def landmarks_to_points(landmarks, image_width, image_height, indices):
    """
    Converte landmarks do MediaPipe em pontos x,y da imagem.
    """

    points = []

    for idx in indices:
        landmark = landmarks[idx]

        x = int(landmark.x * image_width)
        y = int(landmark.y * image_height)

        points.append([x, y])

    return np.array(points, dtype=np.int32)


def draw_region(image, points, color, alpha=0.35):
    """
    Desenha uma região facial por cima da imagem.
    """

    overlay = image.copy()

    if len(points) >= 3:
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(overlay, hull, color)

        image = cv2.addWeighted(
            overlay,
            alpha,
            image,
            1 - alpha,
            0
        )

        cv2.polylines(
            image,
            [hull],
            isClosed=True,
            color=color,
            thickness=2
        )

    return image


def segment_face(image_path, face_mesh):
    """
    Deteta a face numa imagem e desenha as regiões faciais.
    """

    image_bgr = cv2.imread(image_path)

    if image_bgr is None:
        print(f"Erro ao ler imagem: {image_path}")
        return None

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        print(f"Nenhuma face detetada em: {image_path}")
        return None

    height, width, _ = image_bgr.shape

    annotated = image_bgr.copy()

    face_landmarks = results.multi_face_landmarks[0].landmark

    for region_name, indices in FACE_REGIONS.items():
        points = landmarks_to_points(
            face_landmarks,
            width,
            height,
            indices
        )

        color = REGION_COLORS[region_name]

        annotated = draw_region(
            annotated,
            points,
            color
        )

        # Escrever o nome da região junto ao primeiro ponto
        if len(points) > 0:
            x, y = points[0]
            cv2.putText(
                annotated,
                region_name,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA
            )

    return annotated


# Main

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image_paths = get_test_images(TEST_DIR)

    if len(image_paths) == 0:
        print("Não foram encontradas imagens de teste.")
        return

    selected_images = random.sample(
        image_paths,
        min(NUM_IMAGES, len(image_paths))
    )

    mp_face_mesh = mp.solutions.face_mesh

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:

        for i, image_path in enumerate(selected_images):
            print(f"[{i + 1}/{len(selected_images)}] A processar: {image_path}")

            annotated = segment_face(image_path, face_mesh)

            if annotated is None:
                continue

            person_name = os.path.basename(os.path.dirname(image_path))
            image_name = os.path.basename(image_path)

            output_name = f"{i + 1:02d}_{person_name}_{image_name}"
            output_path = os.path.join(OUTPUT_DIR, output_name)

            cv2.imwrite(output_path, annotated)

            print(f"Guardado em: {output_path}")

    print("\nSegmentação concluída!")
    print(f"Imagens guardadas em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()