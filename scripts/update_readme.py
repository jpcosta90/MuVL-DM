import pandas as pd
import glob
import argparse
import os
import re # Usaremos expressões regulares para uma substituição segura

# (Opcional: importe sua função de plotagem se quiser reativá-la no futuro)
from src.utils.visualization import generate_performance_plot

def generate_all_tables_string(results_pattern: str) -> str:
    """
    Função testada e validada: encontra os CSVs de resultado, gera as tabelas
    Markdown e as retorna como uma única string.
    """
    result_files = glob.glob(results_pattern)
    if not result_files:
        return "### No Results Found\n\n*Execute avaliações para gerar arquivos de resultado.*"

    df_list = [pd.read_csv(f) for f in result_files]
    results_df = pd.concat(df_list, ignore_index=True).drop_duplicates()

    if 'dataset' not in results_df.columns:
        return "### Erro\n\n*Os arquivos de resultado não contêm a coluna 'dataset'.*"

    all_tables_string = ""
    unique_datasets = sorted(results_df['dataset'].unique())
    
    for dataset_name in unique_datasets:
        dataset_df = results_df[results_df['dataset'] == dataset_name].copy()

        dataset_df['Link Figura'] = dataset_df.apply(
            lambda row: f"[Link](results/plots/{row['dataset']}_{row['method_name']}.png)", axis=1
        )
        
        if 'eer' in dataset_df.columns:
            dataset_df = dataset_df.rename(columns={'eer': 'EER (%)'})
            if dataset_df['EER (%)'].max() <= 1.0:
                dataset_df['EER (%)'] = (dataset_df['EER (%)'] * 100).round(2)
        
        display_columns = {
            'method_name': 'Method', 'EER (%)': 'EER (%)', 'model': 'Model/Adapter',
            'metric': 'Metric', 'Link Figura': 'Link Figura'
        }
        existing_cols = {k: v for k, v in display_columns.items() if k in dataset_df.columns}
        table_df = dataset_df[list(existing_cols.keys())].rename(columns=existing_cols)
        
        # Preenche valores 'nan' para uma exibição mais limpa
        table_df.fillna('N/A', inplace=True)
        
        if 'EER (%)' in table_df.columns:
            table_df = table_df.sort_values(by='EER (%)', ascending=True)

        markdown_table = table_df.to_markdown(index=False)
        all_tables_string += f"### {dataset_name} Results\n\n" + markdown_table + "\n\n"
        
    return all_tables_string.strip() # .strip() para remover espaços extras no final

def update_readme(readme_path: str, all_tables_content: str):
    """
    Substitui o conteúdo entre os marcadores no README de forma segura
    usando expressões regulares.
    """
    # Marcadores de início e fim da seção que queremos isolar
    start_marker = "## Results"
    end_marker = "### Performance vs. Parameters (LA-CDIP Dataset)"

    try:
        with open(readme_path, 'r', encoding='utf-8') as f:
            readme_content = f.read()
    except FileNotFoundError:
        print(f"❌ ERRO: Arquivo README não encontrado em '{readme_path}'")
        return

    # Constrói o novo bloco de texto que será inserido
    # Garante que haja linhas em branco antes e depois do conteúdo
    replacement_block = f"{start_marker}\n\n{all_tables_content}\n\n{end_marker}"

    # Usa uma expressão regular para encontrar e substituir o bloco inteiro de uma só vez
    # re.DOTALL é crucial para que '.' corresponda a quebras de linha
    pattern = re.compile(f"{re.escape(start_marker)}.*?{re.escape(end_marker)}", re.DOTALL)
    
    new_readme_content, replacements_made = pattern.subn(replacement_block, readme_content)

    if replacements_made == 0:
        print(f"❌ ERRO: Marcadores '{start_marker}' e '{end_marker}' não encontrados no README. Nenhuma atualização foi feita.")
        return

    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(new_readme_content)
        
    print(f"✅ README.md em '{readme_path}' foi atualizado com sucesso!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Atualiza automaticamente as tabelas de resultados no README.md.")
    parser.add_argument('--results-pattern', type=str, default='results/*_master_results.csv', help="Padrão para encontrar os arquivos CSV de resultados.")
    parser.add_argument('--readme-path', type=str, default='README.md', help="Caminho para o arquivo README.md a ser atualizado.")
    args = parser.parse_args()
    
    # (Descomente a linha abaixo para reativar a geração do gráfico de performance)
    generate_performance_plot(la_cdip_results_path='results/LA-CDIP_master_results.csv')
    
    # 1. Gera a string de Markdown com as tabelas (lógica já validada)
    all_tables_md = generate_all_tables_string(args.results_pattern)
    
    # 2. Atualiza o README com a nova função segura
    update_readme(args.readme_path, all_tables_md)

    print("\n✅ Processo de atualização do README concluído!")