import pandas as pd
import re

meta_dicts = {
    'OpenAI o1-preview': {'url': 'https://openai.com/o1/'},
    'Claude 3.5 Sonnet-0620': {'url': 'https://www.anthropic.com/news/claude-3-5-sonnet'},
    'LLaMA-3.1-405B-Instruct': {'url': 'https://huggingface.co/meta-llama/Meta-Llama-3.1-405B-Instruct'},
    'GPT-4o-2024-05-13': {'url': 'https://openai.com/index/hello-gpt-4o/'},
    'GPT-4o mini': {'url': 'https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/'},
    'Claude 3 Opus': {'url': 'https://www.anthropic.com/news/claude-3-family'},
    'GPT-4-Turbo-2024-04-09': {'url': 'https://platform.openai.com/docs/models/'},
    'Gemini 1.5 Pro-API-0409': {'url': 'https://arxiv.org/pdf/2403.05530'},
    'Gemini 1.5 Pro-API-0514': {'url': 'https://arxiv.org/pdf/2403.05530'},
    'Mistral Large 2': {'url': 'https://mistral.ai/news/mistral-large-2407/'},
    'Spark4.0': {'url': 'https://xinghuo.xfyun.cn/sparkapi'},
    'Yi-Large-preview': {'url': 'https://platform.01.ai/'},
    'LLaMA-3-70B-Instruct': {'url': 'https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct'},
    'Qwen-Max-0428': {'url': 'https://qwenlm.github.io/blog/qwen-max-0428/'},
    'Claude 3 Sonnet': {'url': 'https://www.anthropic.com/news/claude-3-family'},
    'Reka Core-20240415': {'url': 'https://arxiv.org/pdf/2404.12387'},
    'MAmmoTH2-8x7B-Plus': {'url': 'https://huggingface.co/TIGER-Lab/MAmmoTH2-8x7B-Plus'},
    'DeepSeek-V2': {'url': 'https://arxiv.org/abs/2405.04434'},
    'Command R+': {'url': 'https://huggingface.co/CohereForAI/c4ai-command-r-plus'},
    'Yi-1.5-34B-Chat': {'url': 'https://huggingface.co/01-ai/Yi-1.5-34B-Chat'},
    'Mistral-Large': {'url': 'https://mistral.ai/news/mistral-large/'},
    'Qwen1.5-72B-Chat': {'url': 'https://huggingface.co/Qwen/Qwen1.5-72B-Chat'},
    'Mistral-Medium': {'url': 'https://mistral.ai/technology/#models'},
    'Gemini 1.0 Pro': {'url': 'https://blog.google/technology/ai/google-gemini-ai/'},
    'Reka Flash-20240226': {'url': 'https://arxiv.org/pdf/2404.12387'},
    'Mistral-Small': {'url': 'https://mistral.ai/technology/#models'},
    'LLaMA-3-8B-Instruct': {'url': 'https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct'},
    'Command R': {'url': 'https://huggingface.co/CohereForAI/c4ai-command-r-v01'},
    'Qwen1.5-32B-Chat': {'url': 'https://huggingface.co/Qwen/Qwen1.5-32B-Chat'},
    'GPT-3.5-Turbo-0125': {'url': 'https://platform.openai.com/docs/models/'},
    'Claude 3 Haiku': {'url': 'https://www.anthropic.com/news/claude-3-family'},
    'Yi-34B-Chat': {'url': 'https://huggingface.co/01-ai/Yi-1.5-34B-Chat'},
    'Mixtral-8x7B-Instruct-v0.1': {'url': 'https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1'},
    'Starling-LM-7B-beta': {'url': 'https://huggingface.co/Nexusflow/Starling-LM-7B-beta'},
    'Yi-1.5-9B-Chat': {'url': 'https://huggingface.co/01-ai/Yi-1.5-9B-Chat'},
    'Gemma-1.1-7B-IT': {'url': 'https://huggingface.co/google/gemma-1.1-7b-it'},
    'Vicuna-33B-v1.3': {'url': 'https://huggingface.co/lmsys/vicuna-33b-v1.3'},
    'LLaMA-2-70B-Chat': {'url': 'https://huggingface.co/meta-llama/Llama-2-70b-chat-hf'},
    'Mistral-7B-Instruct-v0.2': {'url': 'https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2'},
    'Qwen1.5-7B-Chat': {'url': 'https://huggingface.co/Qwen/Qwen1.5-7B-Chat'},
    'Reka Edge-20240208': {'url': 'https://arxiv.org/pdf/2404.12387'},
    'Zephyr-7B-β': {'url': 'https://huggingface.co/HuggingFaceH4/zephyr-7b-beta'},
    'LLaMA-2-7B-Chat': {'url': 'https://huggingface.co/meta-llama/Llama-2-7b-chat-hf'},
    'Yi-6B-Chat': {'url': 'https://huggingface.co/01-ai/Yi-6B-Chat'},
    'Qwen1.5-MoE-A2.7B-Chat': {'url': 'https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B-Chat'},
    'Gemma-1.1-2B-IT': {'url': 'https://huggingface.co/google/gemma-1.1-2b-it'},
    'Vicuna-7B-v1.5': {'url': 'https://huggingface.co/lmsys/vicuna-7b-v1.5'},
    'OLMo-7B-Instruct': {'url': 'https://huggingface.co/allenai/OLMo-7B-Instruct'},
    'Qwen1.5-4B-Chat': {'url': 'https://huggingface.co/Qwen/Qwen1.5-4B-Chat'},
    'JetMoE-8B-Chat': {'url': 'https://huggingface.co/jetmoe/jetmoe-8b-chat'},
    'MPT-7B-Chat': {'url': 'https://huggingface.co/mosaicml/mpt-7b-chat'},
    'MAP-Neo-Instruct-v0.1': {'url': 'https://huggingface.co/m-a-p/neo_7b_instruct_v0.1'},
    'Qwen2-72B-Instruct': {'url': 'https://qwenlm.github.io/blog/qwen2/'},
    
    'LLaMA-3-70B': {'url': 'https://huggingface.co/meta-llama/Meta-Llama-3-70B'},
    'Qwen1.5-72B': {'url': 'https://huggingface.co/Qwen/Qwen1.5-72B'},
    'Yi-34B': {'url': 'https://huggingface.co/01-ai/Yi-1.5-34B'},
    'Qwen1.5-32B': {'url': 'https://huggingface.co/Qwen/Qwen1.5-32B'},
    'Mixtral-8x7B': {'url': 'https://huggingface.co/mistralai/Mixtral-8x7B-v0.1'},
    'LLaMA-2-70B': {'url': 'https://huggingface.co/meta-llama/Llama-2-70b'},
    'Qwen1.5-MoE-A2.7B': {'url': 'https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B'},
    'Qwen1.5-7B': {'url': 'https://huggingface.co/Qwen/Qwen1.5-7B'},
    'LLaMA-3-8B': {'url': 'https://huggingface.co/meta-llama/Meta-Llama-3-8B'},
    'Mistral-7B': {'url': 'https://huggingface.co/mistralai/Mistral-7B-v0.1'},
    'Gemma-7B': {'url': 'https://huggingface.co/google/gemma-7b'},
    'Yi-6B': {'url': 'https://huggingface.co/01-ai/Yi-6B'},
    'Qwen1.5-4B': {'url': 'https://huggingface.co/Qwen/Qwen1.5-4B'},
    'JetMoE-8B': {'url': 'https://huggingface.co/jetmoe/jetmoe-8b'},
    'DeepSeek-7B': {'url': 'https://huggingface.co/deepseek-ai/deepseek-llm-7b-base'},
    'Phi-2': {'url': 'https://huggingface.co/microsoft/phi-2'},
    'DeepSeekMoE-16B': {'url': 'https://huggingface.co/deepseek-ai/deepseek-moe-16b-base'},
    'LLaMA-2-7B': {'url': 'https://huggingface.co/meta-llama/Llama-2-7b'},
    'Gemma-2B': {'url': 'https://huggingface.co/google/gemma-2b'},
    'OLMo-7B': {'url': 'https://huggingface.co/allenai/OLMo-7B'},
    'MPT-7B': {'url': 'https://huggingface.co/mosaicml/mpt-7b'},
}

proprietary_models = [
    'OpenAI o1-preview',
    'Claude 3.5 Sonnet-0620',
    'GPT-4o-2024-05-13',
    'GPT-4o mini',
    'Claude 3 Opus',
    'GPT-4-Turbo-2024-04-09',
    'Gemini 1.5 Pro-API-0409',
    'Gemini 1.5 Pro-API-0514',
    'Mistral Large 2',
    'Spark4.0',
    'Yi-Large-preview',
    'Qwen-Max-0428',
    'Claude 3 Sonnet',
    'Reka Core-20240415',
    'Mistral-Large',
    'Mistral-Medium',
    'Gemini 1.0 Pro',
    'Reka Flash-20240226',
    'Mistral-Small',
    'GPT-3.5-Turbo-0125',
    'Claude 3 Haiku',
    'Reka Edge-20240208'
]


def generate_html_table(input_file, output_file, table_id='table1'):
    # Read the Excel file
    df = pd.read_excel(input_file, header=None)
    # print(df)
    for col_idx in df.columns[1:]:
    # Scores are in the rows below the first one
        scores = pd.to_numeric(df.iloc[1:, col_idx], errors='coerce')
        # Check if there are enough scores to proceed
        if scores.dropna().empty:
            continue
        # print(scores)
        # Find the index of the highest score
        first_max_idx = scores.idxmax()  # Adjust index for header row
        # Apply bold formatting
        df.iloc[first_max_idx, col_idx] = f"<b>{df.iloc[first_max_idx, col_idx]}</b>"
        
        # Nullify the highest score to find the second highest
        scores[first_max_idx] = pd.NA  # Adjust index back for zero-based indexing
        # Check again for remaining valid scores
        if scores.dropna().empty:
            continue
        
        second_max_idx = scores.idxmax()  # Adjust index for header row
        # Apply underline formatting
        if pd.notna(second_max_idx):
            df.iloc[second_max_idx, col_idx] = f"<u>{df.iloc[second_max_idx, col_idx]}</u>"

        
    
    # for t in df.iloc[1:, 0]:
    #     print(f"'{t}': {{'url': ''}},")
    
    # Update the first column to include the hyperlink and formatting
    df.iloc[1:, 0] = df.iloc[1:, 0].apply(lambda x: f'''<td style="text-align: center;width: 200px;"><a href="{meta_dicts[x]['url']}" target="_blank"><b>{x}</b></a></td>''')
    
    df.iloc[0] = df.iloc[0].apply(
        lambda x: f'<td class="js-sort-number" style="background-color:#b3b3b3ff;"><strong><a  style="color:#000000ff;"><b>{x}</b></a></strong></td>' if 'MixEval' not in str(x)
        else f'<td class="js-sort-number" style="background-color:#b3b3b3ff;"><strong><a  style="color:#000000ff;"><b>{f"{x}<br>🔥"}</b></a></strong></td>'
        )
    
    # Convert the DataFrame to an HTML string, without headers and index
    html_table = df.to_html(index=False, escape=False, header=False)
    # print(type(html_table))
    # Add the table with specific class and ID
    if table_id=='table1':
        html_table = html_table.replace('<table border="1" class="dataframe">', '<table class="js-sort-table" id="table1"  style="border: 2px solid #999999ff;">').replace('<td><td ', '<td ').replace('</td></td>', '</td>')
    elif table_id=='table2':
        html_table = html_table.replace('<table border="1" class="dataframe">', '<table class="js-sort-table hidden" id="table2"  style="border: 2px solid #999999ff;">').replace('<td><td ', '<td ').replace('</td></td>', '</td>')
    else:
        raise ValueError(f'Invalid table_id: {table_id}.')
    
    # Remove thead and tbody tags
    html_table = html_table.replace('<thead>', '').replace('</thead>', '').replace('nan', '').replace(' (Mixed)', '<br>(Mixed)').replace("Arena Elo (0527)", "Arena Elo<br> (0527)")
    html_table = html_table.replace('<tbody>', '').replace('</tbody>', '')

    # Replace the default <tr>, <th>, and <td> to include style
    pattern = r'(<tr>\n\s+<td style="text-align: center;width: 200px;"><a href="([^"]+)" target="_blank"><b>(.*?)</b></a></td>)'
    matches = re.findall(pattern, html_table)
    # print(matches)
    for match in matches:
        for p_m in proprietary_models:
            is_pm = False
            if p_m in match:
                is_pm = True
                break
        if is_pm:
            html_table = html_table.replace(match[0], match[0].replace('<tr>\n', '<tr style="background-color: #ecececff;">\n'))
        else:
            html_table = html_table.replace(match[0], match[0].replace('<tr>\n', '<tr style="background-color: #fcfcfcff;">\n'))
        
    # html_table = html_table.replace('<tr>', '<tr style="background-color: rgba(255, 208, 80, 0.15);">')
    # html_table = html_table.replace('<td>', '<td style="padding: 8px; border: 1px solid #ddd; text-align: center;">')
    # html_table = html_table.replace('<th>', '<th style="background-color: #f4f4f4; padding: 8px; border: 1px solid #ddd; text-align: center;">')
    
    # Write the HTML Table to a file
    with open(output_file, "w") as file:
        file.write(html_table)

generate_html_table('data/scores_chat.xlsx', 'static/utils/output_html_tab_chat.html')
generate_html_table('data/scores_base.xlsx', 'static/utils/output_html_tab_base.html', table_id='table2')
