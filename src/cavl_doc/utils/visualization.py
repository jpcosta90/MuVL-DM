# src/utils/visualization.py
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_density(results_df, eer_score, eer_threshold, method_name, dataset_name, metric_type, output_dir='results/plots'):
    """Gera um gráfico de densidade com título e nome de arquivo descritivos."""
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = f"{dataset_name}_{method_name}.png"
    plot_path = os.path.join(output_dir, plot_filename)
    
    plt.figure(figsize=(12, 7))
    sns.kdeplot(data=results_df[results_df['is_equal'] == 1], x='metric_score', label='Pares Iguais (is_equal=1)', fill=True, color='blue', cut=0)
    sns.kdeplot(data=results_df[results_df['is_equal'] == 0], x='metric_score', label='Pares Diferentes (is_equal=0)', fill=True, color='red', cut=0)
    
    plt.axvline(x=eer_threshold, color='black', linestyle='--', label=f'EER Threshold ≈ {eer_threshold:.3f}\n(EER ≈ {eer_score:.3f})')
    
    title = (f'Distribuição de Scores - Dataset: {dataset_name.upper()}\n'
             f'Método: {method_name} | Métrica: {metric_type.capitalize()}')
    plt.title(title, fontsize=14)
    plt.xlabel(f'Score ({metric_type.capitalize()})', fontsize=12)
    plt.ylabel('Densidade', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


def generate_performance_plot(la_cdip_results_path: str, output_dir: str = 'results/plots'):
    """
    Gera o gráfico 'Performance vs. Parameters' usando a lógica de posicionamento manual
    de rótulos para modelos conhecidos e uma posição padrão para os novos.
    
    O Eixo Y é limitado automaticamente até o valor do 'pixel_cosine_baseline' + 1%.
    """
    print("\n--- Gerando gráfico de Performance vs. Parâmetros para LA-CDIP ---")
    
    # --- 1. DADOS ESTÁTICOS (do seu código original) ---
    static_models = [
        ("AlexNet", "AlexNet", 57, 17.33),
        ("VGG-11", "VGG", 129, 14.24),
        ("VGG-13", "VGG", 129, 9.30),
        ("VGG-16", "VGG", 134, 14.74),
        ("VGG-19", "VGG", 139, 17.08),
        ("ResNet-18", "ResNet", 11, 4.98),
        ("ResNet-34", "ResNet", 21, 4.13),
        ("ResNet-50", "ResNet", 23, 10.34),
        ("ResNet-101", "ResNet", 42, 11.31),
        ("ResNet-152", "ResNet", 58, 12.70),
        ("MobileNetV3-S", "MobileNet", 1, 12.74),
        ("MobileNetV3-L", "MobileNet", 4, 8.45),
        ("EfficientNet-0", "EfficientNet", 4, 6.02),
        ("EfficientNet-1", "EfficientNet", 6, 8.88),
        ("EfficientNet-2", "EfficientNet", 7, 7.29),
        ("EfficientNet-3", "EfficientNet", 10, 7.37),
        ("ViT-Base", "ViT", 87, 19.72),
        ("ViT-Large", "ViT", 305, 19.88),
        ("Llama-3.2 11B", "LLM", 11000, 13.95),
        ("InternVL-2.5 8B", "LLM", 8000, 8.58),
        ("Qwen-VL-2.5 7B", "LLM", 7000, 6.61),
        ("InternVL3 2B", "LLM", 2000, 38.98),
        ("InternVL3 8B", "LLM", 8000, 4.04),
        ("InternVL3 14B", "LLM", 14000, 2.85)
    ]
    paper_df = pd.DataFrame(static_models, columns=['method_name', 'arch', 'params', 'eer'])

    # --- 2. DADOS DINÂMICOS (dos seus novos experimentos) ---
    try:
        dynamic_df = pd.read_csv(la_cdip_results_path)
        # Converte para porcentagem se estiver em escala 0-1
        if not dynamic_df.empty and dynamic_df['eer'].max() <= 1.0:
            dynamic_df['eer'] = dynamic_df['eer'] * 100
    except FileNotFoundError:
        dynamic_df = pd.DataFrame(columns=paper_df.columns)

    all_models_df = pd.concat([paper_df, dynamic_df], ignore_index=True)

    # --- 3. LÓGICA DE PLOTAGEM ---
    architecture_colors = {
        "VGG": "green", "ResNet": "purple", "MobileNet": "orange", "EfficientNet": "cyan",
        "ViT": "brown", "LLM": "red", "Fine-Tuned LLM": "magenta", "Baseline": "grey",
        "AlexNet": "blue", "LLM Embedding": "chartreuse", "Fine-Tuned (Head)": "magenta"
    }
    
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    legend_handles = {}

    for _, row in all_models_df.iterrows():
        name, arch, params, eer = row['method_name'], row['arch'], row['params'], row['eer']
        
        # Proteção contra NaN
        if pd.isna(params) or pd.isna(eer):
            continue

        color = architecture_colors.get(arch, "black")
        scatter = ax.scatter(params, eer, color=color, alpha=0.8, edgecolors="black", s=100, label=arch)
        
        if arch not in legend_handles:
            legend_handles[arch] = scatter

        # Lógica de posicionamento manual
        if name == "MobileNetV3-L":
            xytext_offset = (-82, 0)
        elif name == "EfficientNet-2":
            xytext_offset = (-76, 0)
        elif name == "ResNet-34":
            xytext_offset = (5, 0)
        elif name == "InternVL3 8B":
            xytext_offset = (-50, 6)
        elif name == "InternVL3 14B":
            xytext_offset = (-10, -12)
        elif params is not None and params > 1000:
            xytext_offset = (-10, 7)
        else:
            xytext_offset = (5, 0)
        
        ax.annotate(name, (params, eer), fontsize=8, xytext=xytext_offset, textcoords="offset points")
    
    # --- 4. CONFIGURAÇÃO DE EIXOS E TÍTULOS ---
    ax.axhline(y=4.70, color="#666666", linestyle="--", linewidth=1, label="GPT-4o Mini (ref.)")
    ax.axhline(y=2.75, color="black", linestyle="--", linewidth=1.2, label="GPT-4o (ref.)")
    
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (Millions) - Log Scale", fontsize=12)
    ax.set_ylabel("Performance (Lower EER is better)", fontsize=12)
    ax.set_title("Performance vs. Model Parameters on LA-CDIP (ZSL)", fontsize=14)
    ax.set_xlim(left=0.05, right=5e4)
    ax.grid(True, which="major", ls="--", linewidth=0.5)

    # --- 5. AJUSTE DINÂMICO DO EIXO Y (Alteração solicitada) ---
    # Procura o baseline no DataFrame combinado
    baseline_row = all_models_df[all_models_df['method_name'] == 'pixel_cosine_baseline']

    if not baseline_row.empty:
        # Pega o valor do EER
        baseline_val = baseline_row.iloc[0]['eer']
        # Define o limite superior como Baseline + 1.0 (assumindo escala %)
        y_limit = baseline_val + 1.0
        
        # Define os limites: 0 até Baseline + 1%
        ax.set_ylim(0, y_limit)
        print(f"   -> Eixo Y limitado em {y_limit:.2f}% (Baseline {baseline_val:.2f}% + 1%)")
    else:
        print("   -> Aviso: 'pixel_cosine_baseline' não encontrado. Eixo Y automático.")

    # --- 6. LEGENDA E SALVAMENTO ---
    main_handles = list(legend_handles.values())
    main_labels = list(legend_handles.keys())
    ref_handles, ref_labels = ax.get_legend_handles_labels()
    # Filtra apenas labels de referência que contêm 'ref'
    ref_handles = [h for h, l in zip(ref_handles, ref_labels) if 'ref' in l]
    ref_labels = [l for l in ref_labels if 'ref' in l]
    
    ax.legend(main_handles + ref_handles, main_labels + ref_labels, loc='upper right', fontsize=10)

    plt.figtext(0.99, 0.01, 'Baseline vision model results sourced from ICDARWML 2025 paper.', 
                horizontalalignment='right', fontsize=7, color='gray')
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "LA-CDIP_performance_vs_parameters.png")
    plt.savefig(output_path, bbox_inches="tight")
    print(f"   -> Gráfico de performance salvo com sucesso em '{output_path}'")
    plt.close()