import warnings # warnings precisa ser importado primeiro
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import argparse
from datetime import datetime
import json
import sys
import torch

# Adiciona a raiz do projeto ao path para que 'src' seja encontrado.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if project_root not in sys.path:
    sys.path.append(project_root)

if torch.cuda.is_available():
    torch.set_default_dtype(torch.bfloat16)

# ==========================================================
# 1. IMPORTAÇÕES DA SUA BIBLIOTECA 'SRC'
# ==========================================================
from src.models.lvlm_handler import load_model, warm_up_model
from src.data_loaders.documentpairs import DocumentPairDataset
from src.finetuning.trainer import run_training_loop
from src.utils.helpers import SuppressSpecificOutput

# ==========================================================
# 2. FUNÇÃO PRINCIPAL (ORQUESTRADOR DO TREINAMENTO)
# ==========================================================
def main(args):
    """
    Orquestra o processo completo de fine-tuning (Aprendizado Contrastivo)
    usando um arquivo de pares de treinamento.
    """
    # --- 1. SETUP E NOMENCLATURA DINÂMICA DO DIRETÓRIO DE SAÍDA ---
    print("--- 1. Configurando o experimento de fine-tuning ---")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    dataset_name = os.path.basename(os.path.dirname(args.pairs_csv))
    model_slug = args.base_model.replace('/', '_')
    train_size_str = str(args.training_sample_size) if args.training_sample_size else 'full'

    # Formato: DATASET_MODEL_LAYERS_TRAINSIZE_TIMESTAMP
    output_dir_name_parts = [
        dataset_name,
        model_slug,
        str(args.train_vision_layers), # Número de camadas
        train_size_str,               # Tamanho da amostra de treino
        timestamp
    ]
    output_dir = os.path.join('checkpoints', '_'.join(output_dir_name_parts))

    os.makedirs(output_dir, exist_ok=True)
    print(f"   -> Os adaptadores serão salvos em: {output_dir}")

    config_data = vars(args)
    with open(os.path.join(output_dir, 'training_config.json'), 'w') as f:
        json.dump(config_data, f, indent=4)
    print("   -> Configuração do treino salva em 'training_config.json'.")

    # --- 2. CARREGAR DADOS E MODELO BASE ---
    print("\n--- 2. Carregando dados e modelo base ---")
    model, processor, tokenizer, _, _ = load_model(
        model_name=args.base_model, # <--- CORRIGIDO
        adapter_path=None,
        load_in_4bit=True # LoRA geralmente usa 4-bit (QLoRA)
        # projection_output_dim não é necessário aqui
    )
    warm_up_model(model, processor)
    
    full_dataset = DocumentPairDataset(csv_path=args.pairs_csv, base_dir=args.base_image_dir)
    print(f"   -> Dataset completo carregado de '{args.pairs_csv}' com {len(full_dataset.df)} amostras.")
    
    if len(full_dataset.df) == 0:
        print(f"❌ ERRO: O arquivo de dados '{args.pairs_csv}' está vazio. Interrompendo.")
        sys.exit(1)

    if args.training_sample_size and args.training_sample_size < len(full_dataset.df):
        print(f"   -> Criando um subconjunto de treinamento com {args.training_sample_size} amostras...")
        training_df_subset = full_dataset.df.sample(n=args.training_sample_size, random_state=42)
        training_dataset = DocumentPairDataset(csv_path=args.pairs_csv, base_dir=args.base_image_dir)
        training_dataset.df = training_df_subset
    else:
        print("   -> Usando o conjunto de dados de treinamento completo.")
        training_dataset = full_dataset

    print(f"   -> Amostras disponíveis para o treinamento: {len(training_dataset.df)}")
    
    # --- 3. EXECUTAR O TREINAMENTO ---
    warning_message = "Index put requires the source and destination dtypes match"
    print("\n--- 3. Iniciando o ciclo de treinamento ---")
    with SuppressSpecificOutput(warning_message): # <<< REMOVER OU COMENTAR ESTA LINHA
        run_training_loop( # <<< INDENTAR ESTE BLOCO DE VOLTA
            model=model, tokenizer=tokenizer, dataset=training_dataset,
            prompt=args.prompt, output_dir=output_dir,
            num_vision_layers_to_train=args.train_vision_layers,
            num_epochs=args.epochs, learning_rate=args.lr,
            samples_per_epoch=args.samples_per_epoch
        )
    print("\n✅ Treinamento concluído com sucesso!")

# ==========================================================
# 4. PONTO DE ENTRADA DO SCRIPT
# ==========================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Executa o fine-tuning de um modelo com Aprendizado Contrastivo (QLoRA).")
    
    parser.add_argument('--pairs-csv', type=str, required=True, help='Caminho para o arquivo de pares de TREINAMENTO.')
    parser.add_argument('--base-image-dir', type=str, required=True, help='Diretório base onde as imagens estão localizadas.')
    parser.add_argument('--training-sample-size', type=int, default=None, help='Número total de amostras a serem usadas. Se não especificado, usa o dataset completo.')
    parser.add_argument('--base-model', type=str, default='InternVL3-2B', help='Nome do modelo base.')
    parser.add_argument('--prompt', type=str, default='<image> Analyze this document', help='Prompt usado durante o treino.')
    parser.add_argument('--epochs', type=int, default=5, help='Número de épocas.')
    parser.add_argument('--lr', type=float, default=1e-5, help='Taxa de aprendizado.')
    parser.add_argument('--samples-per-epoch', type=int, default=2000, help='Número de amostras aleatórias a serem usadas em CADA época.')
    parser.add_argument('--train-vision-layers', type=int, default=0, help='Número de camadas do vision encoder a serem treinadas (0, 1, 2... ou -1 para todas).')
    
    args = parser.parse_args()
    main(args)