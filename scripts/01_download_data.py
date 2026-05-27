import os
import pandas as pd

#Este script foi o primeiro a ser feito, explora o dataset do LFW e percebe quantas identidades existem quantas imagens existem
#quais pessoas têm mais do que 10 imagens, e quais são as 10 pessoas com mais imagens.

#Escolheu-se o LFW ,porque é um dataset público e clássico de reconhecimento facial. O objetivo não era criar um sistema comercial gigante 
#mas sim ter um conjunto de faces realistas para treinar um modelo e depois analisar as opções

def explore_lfw():
    data_dir = './data/lfw-deepfunneled' #usou se a versão do deepfunneled porque tem as faces alinhadas, o que facilita o treinamento do modelo depois.
    # A versão original tem as faces em posições variadas, o que pode dificultar o processo de treinamento e análise.
    
    # Contar identidades e imagens
    identities = os.listdir(data_dir) #cada pasta dentro do data_dir representa uma identidade diferente, então contamos quantas pastas existem
    identities = [i for i in identities if os.path.isdir(os.path.join(data_dir, i))] #filtra apenas as pastas para listar as válidas
    
    total_images = 0 
    images_per_person = {}
    
    for identity in identities:
        imgs = os.listdir(os.path.join(data_dir, identity))
        imgs = [i for i in imgs if i.endswith('.jpg')]
        images_per_person[identity] = len(imgs)
        total_images += len(imgs)
    
    print(f"Total de identidades: {len(identities)}")
    print(f"Total de imagens: {total_images}")
    
    # Top 10 pessoas com mais imagens
    top10 = sorted(images_per_person.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop 10 pessoas com mais imagens:")
    for name, count in top10:
        print(f"  {name}: {count} imagens")
    
    # Quantas pessoas têm mais de 10 imagens
    filtered = {k: v for k, v in images_per_person.items() if v >= 10}
    print(f"\nPessoas com >= 10 imagens: {len(filtered)}")

if __name__ == "__main__":
    explore_lfw()