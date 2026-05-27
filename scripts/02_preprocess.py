import os
import shutil
from sklearn.model_selection import train_test_split

DATA_DIR = './data/lfw-deepfunneled'
PROCESSED_DIR = './data/processed'
MIN_IMAGES = 10

def get_valid_identities(data_dir, min_images):
    identities = []
    for person in os.listdir(data_dir):
        person_path = os.path.join(data_dir, person)
        if not os.path.isdir(person_path):
            continue
        imgs = [f for f in os.listdir(person_path) if f.endswith('.jpg')]
        if len(imgs) >= min_images:
            identities.append(person)
    return sorted(identities)

def prepare_dataset(data_dir, processed_dir, min_images=10):
    os.makedirs(processed_dir, exist_ok=True)
    train_dir = os.path.join(processed_dir, 'train')
    test_dir = os.path.join(processed_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    identities = get_valid_identities(data_dir, min_images)
    print(f"Identidades válidas (>= {min_images} imagens): {len(identities)}")

    for person in identities:
        person_path = os.path.join(data_dir, person)
        imgs = sorted([f for f in os.listdir(person_path) if f.endswith('.jpg')])

        train_imgs, test_imgs = train_test_split(imgs, test_size=0.2, random_state=42)

        for split, split_imgs in [('train', train_imgs), ('test', test_imgs)]:
            split_person_dir = os.path.join(processed_dir, split, person)
            os.makedirs(split_person_dir, exist_ok=True)
            for img_name in split_imgs:
                src = os.path.join(person_path, img_name)
                dst = os.path.join(split_person_dir, img_name)
                shutil.copy2(src, dst)

    train_total = sum(len(os.listdir(os.path.join(train_dir, p))) for p in os.listdir(train_dir))
    test_total = sum(len(os.listdir(os.path.join(test_dir, p))) for p in os.listdir(test_dir))

    print(f"Imagens de treino: {train_total}")
    print(f"Imagens de teste: {test_total}")
    print("Pré-processamento concluído!")

if __name__ == "__main__":
    prepare_dataset(DATA_DIR, PROCESSED_DIR, MIN_IMAGES)