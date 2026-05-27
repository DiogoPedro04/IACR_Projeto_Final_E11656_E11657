import os
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# CONFIGURAÇÕES
# =========================

REGION_CSV = "./outputs/region_analysis/region_attention_results.csv"
OCCLUSION_CSV = "./outputs/occlusion_analysis/occlusion_results.csv"
FAILURE_CSV = "./outputs/failure_cases/failure_cases.csv"

OUTPUT_DIR = "./outputs/final_figures"


# =========================
# FUNÇÕES AUXILIARES
# =========================

def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_region_attention_global():
    df = pd.read_csv(REGION_CSV)

    cols = [
        "eyes_total",
        "nose",
        "mouth",
        "remaining_face",
        "outside_face"
    ]

    means = df[cols].mean().sort_values(ascending=False)

    plt.figure(figsize=(9, 5))
    plt.bar(means.index, means.values)

    plt.ylabel("Atenção média Grad-CAM (%)")
    plt.xlabel("Região")
    plt.title("Distribuição média da atenção Grad-CAM por região")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, "region_attention_global.png")
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Guardado: {output_path}")


def save_region_attention_correct_vs_incorrect():
    df = pd.read_csv(REGION_CSV)

    cols = [
        "eyes_total",
        "nose",
        "mouth",
        "remaining_face",
        "outside_face"
    ]

    correct_means = df[df["correct"] == True][cols].mean()
    incorrect_means = df[df["correct"] == False][cols].mean()

    plot_df = pd.DataFrame({
        "Corretas": correct_means,
        "Incorretas": incorrect_means
    })

    ax = plot_df.plot(kind="bar", figsize=(10, 5))

    ax.set_ylabel("Atenção média Grad-CAM (%)")
    ax.set_xlabel("Região")
    ax.set_title("Atenção por região: previsões corretas vs incorretas")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, "region_attention_correct_vs_incorrect.png")
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Guardado: {output_path}")


def save_occlusion_accuracy():
    df = pd.read_csv(OCCLUSION_CSV)

    condition_order = [
        "original",
        "mask_eyes",
        "mask_nose",
        "mask_mouth",
        "mask_face_oval",
        "mask_outside_face",
        "dark",
        "bright"
    ]

    summary = df.groupby("condition")["correct"].mean()
    summary = summary.loc[condition_order] * 100

    plt.figure(figsize=(10, 5))
    plt.bar(summary.index, summary.values)

    plt.ylabel("Accuracy (%)")
    plt.xlabel("Condição")
    plt.title("Accuracy por condição de mascaramento")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, "occlusion_accuracy.png")
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Guardado: {output_path}")


def save_top_confusions():
    df = pd.read_csv(FAILURE_CSV)

    if len(df) == 0:
        print("Sem falhas para gerar gráfico de confusões.")
        return

    df["confusion_pair"] = df["true_label"] + " -> " + df["pred_label"]

    top_confusions = df["confusion_pair"].value_counts().head(10)

    plt.figure(figsize=(11, 6))
    plt.barh(top_confusions.index[::-1], top_confusions.values[::-1])

    plt.xlabel("Número de ocorrências")
    plt.ylabel("Par de confusão")
    plt.title("Pares de confusão mais frequentes")
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, "top_confusions.png")
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Guardado: {output_path}")


def save_failure_confidence_distribution():
    df = pd.read_csv(FAILURE_CSV)

    if len(df) == 0:
        print("Sem falhas para gerar histograma de confiança.")
        return

    confidence_percent = df["confidence"] * 100

    plt.figure(figsize=(9, 5))
    plt.hist(confidence_percent, bins=20)

    plt.xlabel("Confiança da previsão errada (%)")
    plt.ylabel("Número de falhas")
    plt.title("Distribuição da confiança nos casos de falha")
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, "failure_confidence_distribution.png")
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Guardado: {output_path}")


# =========================
# MAIN
# =========================

def main():
    ensure_output_dir()

    save_region_attention_global()
    save_region_attention_correct_vs_incorrect()
    save_occlusion_accuracy()
    save_top_confusions()
    save_failure_confidence_distribution()

    print("\nVisualizações finais concluídas!")
    print(f"Figuras guardadas em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()