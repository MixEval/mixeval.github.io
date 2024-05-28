import pandas as pd
import re

meta_dicts = {
    'GPT-4o': {'url': 'https://openai.com/index/hello-gpt-4o/'},
    'Claude 3 Opus': {'url': 'https://www.anthropic.com/news/claude-3-family'},
    'GPT-4-Turbo': {'url': 'https://platform.openai.com/docs/models/'},
    'Gemini 1.5 Pro': {'url': 'https://arxiv.org/pdf/2403.05530'},
    'Yi-Large': {'url': 'https://platform.01.ai/'},
    'LLaMA-3-70B-Instruct': {'url': 'https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct'},
    'Qwen-Max-0428': {'url': 'https://qwenlm.github.io/blog/qwen-max-0428/'},
    'Claude 3 Sonnet': {'url': 'https://www.anthropic.com/news/claude-3-family'},
    'Reka Core': {'url': 'https://arxiv.org/pdf/2404.12387'},
    'MAmmoTH2-8x7B-Plus': {'url': 'https://huggingface.co/TIGER-Lab/MAmmoTH2-8x7B-Plus'},
    'DeepSeek-V2': {'url': 'https://arxiv.org/abs/2405.04434'},
    'Command R+': {'url': 'https://huggingface.co/CohereForAI/c4ai-command-r-plus'},
    'Yi-1.5-34B-Chat': {'url': 'https://huggingface.co/01-ai/Yi-1.5-34B-Chat'},
    'Mistral-Large': {'url': 'https://mistral.ai/news/mistral-large/'},
    'Qwen1.5-72B-Chat': {'url': 'https://huggingface.co/Qwen/Qwen1.5-72B-Chat'},
    'Mistral-Medium': {'url': 'https://mistral.ai/technology/#models'},
    'Gemini 1.0 Pro': {'url': 'https://blog.google/technology/ai/google-gemini-ai/'},
    'Reka Flash': {'url': 'https://arxiv.org/pdf/2404.12387'},
    'Mistral-Small': {'url': 'https://mistral.ai/technology/#models'},
    'LLaMA-3-8B-Instruct': {'url': 'https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct'},
    'Command R': {'url': 'https://huggingface.co/CohereForAI/c4ai-command-r-v01'},
    'Qwen1.5-32B-Chat': {'url': 'https://huggingface.co/Qwen/Qwen1.5-32B-Chat'},
    'GPT-3.5-Turbo': {'url': 'https://platform.openai.com/docs/models/'},
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
    'Reka Edge': {'url': 'https://arxiv.org/pdf/2404.12387'},
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
}

proprietary_models = [
    'GPT-4o',
    'Claude 3 Opus',
    'GPT-4-Turbo',
    'Gemini 1.5 Pro',
    'Yi-Large',
    'Qwen-Max-0428',
    'Claude 3 Sonnet',
    'Reka Core',
    'Mistral-Large',
    'Mistral-Medium',
    'Gemini 1.0 Pro',
    'Reka Flash',
    'Mistral-Small',
    'GPT-3.5-Turbo',
    'Claude 3 Haiku',
    'Reka Edge'
]

def generate_html_table(input_file, output_file):
    # Read the Excel file
    df = pd.read_excel(input_file, header=None)
    
    # for t in df.iloc[1:, 0]:
    #     print(f"'{t}': {{'url': ''}},")
    
    # Update the first column to include the hyperlink and formatting
    df.iloc[1:, 0] = df.iloc[1:, 0].apply(lambda x: f'''<td style="text-align: left;width: 200px;"><a href="{meta_dicts[x]['url']}"><b>{x}</b></a></td>''')
    
    df.iloc[0] = df.iloc[0].apply(lambda x: f'<td class="js-sort-number"><strong>{x}</strong></td>')
    
    # Convert the DataFrame to an HTML string, without headers and index
    html_table = df.to_html(index=False, escape=False, header=False)
    # print(type(html_table))
    # Add the table with specific class and ID
    html_table = html_table.replace('<table border="1" class="dataframe">', '<table class="js-sort-table" id="table1">').replace('<td><td ', '<td ').replace('</td></td>', '</td>')
    
    # Remove thead and tbody tags
    html_table = html_table.replace('<thead>', '').replace('</thead>', '').replace('nan', '').replace(' (Mixed)', '<br>(Mixed)')
    html_table = html_table.replace('<tbody>', '').replace('</tbody>', '')

    # Replace the default <tr>, <th>, and <td> to include style
    pattern = r'(<tr>\n\s+<td style="text-align: left;width: 200px;"><a href="([^"]+)"><b>(.*?)</b></a></td>)'
    matches = re.findall(pattern, html_table)
    # print(matches)
    for match in matches:
        for p_m in proprietary_models:
            is_pm = False
            if p_m in match:
                is_pm = True
                break
        if is_pm:
            html_table = html_table.replace(match[0], match[0].replace('<tr>\n', '<tr style="background-color: rgba(117, 209, 215, 0.1);">\n'))
        else:
            html_table = html_table.replace(match[0], match[0].replace('<tr>\n', '<tr style="background-color: rgba(255, 208, 80, 0.15);">\n'))
        
    # html_table = html_table.replace('<tr>', '<tr style="background-color: rgba(255, 208, 80, 0.15);">')
    # html_table = html_table.replace('<td>', '<td style="padding: 8px; border: 1px solid #ddd; text-align: center;">')
    # html_table = html_table.replace('<th>', '<th style="background-color: #f4f4f4; padding: 8px; border: 1px solid #ddd; text-align: center;">')
    
    # Write the HTML Table to a file
    with open(output_file, "w") as file:
        file.write(html_table)

generate_html_table('static/utils/scores.xlsx', 'static/utils/output_html_tab.html')
