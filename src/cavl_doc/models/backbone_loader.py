import torch
import pandas as pd
from typing import List, Optional, Dict
from torch.utils.data import Dataset, DataLoader
import os
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, BitsAndBytesConfig
import torch.nn as nn
from peft import PeftModel
from cavl_doc.utils.misc import NewConnector # Ou onde quer que NewConnector esteja definida
from cavl_doc.modules.heads import mpProjectionHead
import glob

import logging
from typing import Tuple, Optional # Importar Optional para tipos mais claros

def _get_quantization_config() -> BitsAndBytesConfig:
    """Retorna a configuração padrão de quantização BitsAndBytes."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

logger = logging.getLogger(__name__)

def warm_up_model(model, tokenizer):
    """
    Realiza uma inferência com uma única imagem para "aquecer" o modelo
    InternVL usando seu método específico `model.chat()`.

    Versão corrigida para lidar com a saída de `model.chat` que retorna
    apenas um valor (a string da resposta).

    Args:
        model: O modelo InternVL carregado.
        tokenizer: O tokenizer correspondente ao modelo.
    """
    print("Aquecendo o modelo com uma única imagem de teste...")

    try:
        # 1. Criar um tensor 'pixel_values' falso para uma única imagem
        pixel_values = torch.randn(1, 3, 448, 448, dtype=model.dtype, device=model.device)

        # 2. Definir a lista de patches para uma imagem
        num_patches_list = [pixel_values.size(0)]

        # 3. Criar um prompt e configuração simples
        question = "<image>\nDescreva esta imagem em poucas palavras."
        generation_config = dict(max_new_tokens=10, do_sample=False)

        # 4. Executar a inferência com `model.chat`
        with torch.no_grad():
            # CORREÇÃO: Atribuir a saída a uma única variável 'response'
            response = model.chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question=question,
                generation_config=generation_config,
                num_patches_list=num_patches_list,
                history=None,
                return_history=False
            )

        print(f"Saída de teste gerada (será descartada): '{response.strip()}'")
        print("Aquecimento do modelo concluído com sucesso.")

    except Exception as e:
        print(f"Ocorreu um erro durante o aquecimento do modelo: {e}")
        print("O modelo pode não estar totalmente inicializado. Verifique a mensagem de erro.")

def load_model(
    model_name: str = 'InternVL3-2B',
    adapter_path: Optional[str] = None,
    load_in_4bit: bool = True,
    projection_output_dim: int = 512
) -> Tuple[AutoModelForCausalLM, AutoProcessor, AutoTokenizer, Optional[nn.Module], Optional[nn.Module]]:
    
    model_path = f"OpenGVLab/{model_name}"
    final_model: AutoModelForCausalLM = None
    new_connector_loaded: Optional[nn.Module] = None
    projection_head_loaded: Optional[nn.Module] = None

    # Determinar tipo de checkpoint
    is_lora_checkpoint = False
    is_connector_head_checkpoint = False
    is_rl_head_checkpoint = False 
    rl_head_ckpt_path = None # [NOVO] O caminho será determinado
    effective_load_in_4bit = load_in_4bit

    if adapter_path and os.path.isdir(adapter_path):
        lora_config_path = os.path.join(adapter_path, 'adapter_config.json')
        connector_head_ckpt_path = os.path.join(adapter_path, 'training_checkpoint.pt')
        
        # VVV --- [LÓGICA DE BUSCA MODIFICADA] --- VVV
        # Procura por *qualquer* checkpoint de cabeça de aluno (student_head)
        epoch_files = glob.glob(os.path.join(adapter_path, 'student_head_epoch_*.pt'))
        
        if os.path.exists(lora_config_path):
            is_lora_checkpoint = True
            logger.info(f"Detectado checkpoint LoRA em: {adapter_path}")
        
        elif os.path.exists(connector_head_ckpt_path):
            is_connector_head_checkpoint = True
            logger.info(f"Detectado checkpoint Connector/Head (Antigo) em: {adapter_path}")
            if load_in_4bit:
                logger.warning("Checkpoints Connector/Head requerem bfloat16. Ignorando load_in_4bit=True.")
            effective_load_in_4bit = False 
        
        elif len(epoch_files) > 0:
            is_rl_head_checkpoint = True
            
            # Pega a época mais recente (ex: 1, 2, 3 -> pega 3)
            epoch_files.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]), reverse=True)
            rl_head_ckpt_path = epoch_files[0] # Pega o checkpoint da época mais alta
            
            logger.info(f"Detectado checkpoint RL-Head (Aluno). Usando a época mais recente: {os.path.basename(rl_head_ckpt_path)}")
        # ^^^ --- [FIM DA LÓGICA MODIFICADA] --- ^^^
        
        else:
            logger.warning(f"Diretório de checkpoint '{adapter_path}' não contém 'adapter_config.json', 'training_checkpoint.pt' ou 'student_head_epoch_*.pt'. Carregando apenas modelo base.")
    
    elif adapter_path:
        logger.warning(f"Caminho do checkpoint '{adapter_path}' não encontrado. Carregando apenas modelo base.")

    # --- 1. Carregar o Modelo Base ---
    # (O restante desta seção não muda)
    print(f"--- 1. Carregando o Modelo Base: {model_path} ---")
    quantization_config_obj = None
    model_dtype = torch.bfloat16 
    if effective_load_in_4bit:
        print("    -> Configurando para carregar em 4-bit (QLoRA).")
        quantization_config_obj = _get_quantization_config()
    else:
        print("    -> Configurando para carregar em bfloat16 (precisão total).")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=model_dtype,
            device_map="auto", 
            quantization_config=quantization_config_obj
        )
        base_model.eval()
    except Exception as e:
        logger.error(f"Falha ao carregar modelo base {model_path}: {e}", exc_info=True)
        raise e

    # --- 2. Processar Checkpoint (se aplicável) ---
    if is_lora_checkpoint:
        # ... (lógica do LoRA não muda) ...
        print(f"--- 2. Anexando Adaptadores LoRA de: {adapter_path} ---")
        try:
            final_model = PeftModel.from_pretrained(base_model, adapter_path)
            final_model.eval()
            print("    -> Adaptadores LoRA aplicados com sucesso.")
        except Exception as e:
            logger.error(f"Erro ao carregar adaptadores LoRA: {e}", exc_info=True)
            final_model = base_model

    elif is_connector_head_checkpoint:
        # ... (lógica do Connector/Head antigo não muda) ...
        print(f"--- 2. Carregando Camadas Connector/Head (Antigo) de: {adapter_path} ---")
        try:
            final_model = base_model
            final_model.eval() 
            ckpt_path = os.path.join(adapter_path, 'training_checkpoint.pt')
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            VISION_OUTPUT_DIM = 1024
            LLM_INPUT_DIM = 1536 
            if 'new_connector_state_dict' in checkpoint:
                new_connector_loaded = NewConnector(VISION_OUTPUT_DIM, LLM_INPUT_DIM)
                new_connector_loaded.load_state_dict(checkpoint['new_connector_state_dict'])
                new_connector_loaded = new_connector_loaded.to(final_model.device).to(model_dtype).eval()
                print("    -> Camada NewConnector (Antiga) carregada com sucesso.")
            if 'projection_head_state_dict' in checkpoint:
                projection_head_loaded = mpProjectionHead(input_dim=LLM_INPUT_DIM, output_dim=projection_output_dim)
                projection_head_loaded.load_state_dict(checkpoint['projection_head_state_dict'])
                projection_head_loaded = projection_head_loaded.to(final_model.device).to(model_dtype).eval()
                print("    -> Camada ProjectionHead (Antiga) carregada com sucesso.")
        except Exception as e:
            logger.error(f"Erro ao carregar camadas Connector/Head (Antigo): {e}", exc_info=True)
            final_model = base_model
            new_connector_loaded = None
            projection_head_loaded = None

    # VVV --- [LÓGICA MODIFICADA] --- VVV
    elif is_rl_head_checkpoint:
        print(f"--- 2. Carregando Camada RL-Head (Aluno) de: {os.path.basename(rl_head_ckpt_path)} ---")
        try:
            final_model = base_model
            final_model.eval() 
            
            # [MODIFICADO] Usa o caminho do arquivo encontrado
            checkpoint = torch.load(rl_head_ckpt_path, map_location='cpu')

            LLM_INPUT_DIM = 1536 

            projection_head_loaded = ProjectionHead(input_dim=LLM_INPUT_DIM, output_dim=projection_output_dim)
            projection_head_loaded.load_state_dict(checkpoint)
            
            projection_head_loaded = projection_head_loaded.to(final_model.device).to(model_dtype).eval()
            print("    -> Camada RL-Head (Aluno) carregada com sucesso.")

        except Exception as e:
            logger.error(f"Erro ao carregar camada RL-Head: {e}", exc_info=True)
            projection_head_loaded = None
    # ^^^ --- [FIM DA LÓGICA MODIFICADA] --- ^^^
            
    else:
        final_model = base_model

    # --- 3. Carregar Processor e Tokenizer ---
    # (Não muda)
    try:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        print("✅ Processador e Tokenizer carregados.")
    except Exception as e:
        logger.error(f"Falha ao carregar processor/tokenizer de {model_path}: {e}", exc_info=True)
        raise e

    print("✅ Carregamento concluído.")
    return final_model, processor, tokenizer, new_connector_loaded, projection_head_loaded