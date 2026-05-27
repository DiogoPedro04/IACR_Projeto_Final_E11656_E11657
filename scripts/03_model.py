import os
import json
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from facenet_pytorch import InceptionResnetV1



# Configurações iniciais


TRAIN_DIR = "./data/processed/train"
TEST_DIR = "./data/processed/test"

MODEL_DIR = "./models"
MODEL_PATH = os.path.join(MODEL_DIR, "facenet_lfw.pth")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")

IMG_SIZE = 160
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.0001


# Escolha do dispositivo

def get_device():
    """
    Escolhe automaticamente o melhor dispositivo disponível:
    - CUDA se houver GPU NVIDIA
    - MPS se estiveres num Mac com Apple Silicon
    - CPU caso contrário
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# Modelo

class FaceNetClassifier(nn.Module):
    """
    Modelo de reconhecimento facial baseado em FaceNet.

    O FaceNet pré-treinado gera um vetor de características com 512 valores.
    Depois adicionamos uma camada Linear para classificar as identidades do LFW.
    """

    def __init__(self, num_classes):
        super(FaceNetClassifier, self).__init__()

        # FaceNet pré-treinado no VGGFace2
        self.backbone = InceptionResnetV1(
            pretrained="vggface2",
            classify=False
        )

        # Camada final para as nossas classes do LFW
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        embeddings = self.backbone(x)
        outputs = self.classifier(embeddings)
        return outputs


# Dados

def get_dataloaders():
    """
    Carrega as imagens de treino e teste usando ImageFolder.

    A estrutura esperada é:
    data/processed/train/Nome_Pessoa/imagem.jpg
    data/processed/test/Nome_Pessoa/imagem.jpg
    """

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),

        # Normalização usada habitualmente com FaceNet:
        # transforma os valores de [0,1] para [-1,1]
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    return train_loader, test_loader, train_dataset.classes


# Treino

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc="Treino", leave=False)

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        # Limpar gradientes anteriores
        optimizer.zero_grad()

        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Estatísticas
        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100 * correct / total:.2f}%"
        })

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


# Avaliação

def evaluate(model, test_loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Validação", leave=False)

        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / total
    val_acc = correct / total

    return val_loss, val_acc


# Main

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    device = get_device()
    print(f"Dispositivo usado: {device}")

    print("\nA carregar dados...")
    train_loader, test_loader, class_names = get_dataloaders()

    num_classes = len(class_names)

    print(f"Número de classes: {num_classes}")
    print(f"Imagens de treino: {len(train_loader.dataset)}")
    print(f"Imagens de teste: {len(test_loader.dataset)}")

    # Guardar nomes das classes para usar mais tarde na inferência e explicações
    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=4, ensure_ascii=False)

    print(f"Classes guardadas em: {CLASS_NAMES_PATH}")

    print("\nA carregar FaceNet pré-treinado...")
    model = FaceNetClassifier(num_classes=num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_acc = 0.0

    print("\nA começar o treino...\n")

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_acc = evaluate(
            model,
            test_loader,
            criterion,
            device
        )

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc * 100:.2f}%")

        # Guardar o melhor modelo
        if val_acc > best_acc:
            best_acc = val_acc

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "num_classes": num_classes,
                "class_names": class_names,
                "best_acc": best_acc,
                "img_size": IMG_SIZE
            }

            torch.save(checkpoint, MODEL_PATH)

            print(f"Novo melhor modelo guardado em: {MODEL_PATH}")
            print(f"Melhor accuracy até agora: {best_acc * 100:.2f}%")

        print("-" * 60)

    print("\nTreino concluído!")
    print(f"Melhor accuracy final: {best_acc * 100:.2f}%")
    print(f"Modelo final guardado em: {MODEL_PATH}")


if __name__ == "__main__":
    main()