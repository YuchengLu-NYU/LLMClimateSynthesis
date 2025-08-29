#%%
import pandas as pd
import json
from pathlib import Path
from config import MODEL_CONFIGS, TASK_CONFIGS


def parse_jsonl_outputs(filename) -> pd.DataFrame:
    outputs = []
    with open(filename, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                # Extract just the prediction
                prediction = data['response']['body']['choices'][0]['message']['content']
                outputs.append(prediction)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error parsing line: {e}")
                continue
    
    df = pd.DataFrame(outputs, columns=['result'])
    return df

def parse_response_task_model(task, model, response_dir: str = "responses") -> pd.DataFrame:

    response_path = Path(response_dir)
    if not response_path.exists():
        raise FileNotFoundError(f"Response directory not found: {response_dir}")
    
    # Construct the expected directory name
    task_model_dir_name = f"{task}_{model}"
    task_model_dir = response_path / task_model_dir_name
    
    if not task_model_dir.exists() or not task_model_dir.is_dir():
        raise FileNotFoundError(f"Task-model directory not found: {task_model_dir}")
    
    all_responses = []
    
    for json_file in task_model_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                filename = json_file.stem
                row_id = None
                if filename.startswith("row"):
                    try:
                        row_id = int(filename.split('_')[0][3:])
                    except (ValueError, IndexError):
                        pass
                
                timestamp = None
                if '_' in filename:
                    timestamp_parts = filename.split('_')[1:4] 
                    if len(timestamp_parts) >= 2:
                        timestamp = '_'.join(timestamp_parts)
                
                response_record = {
                    'model': model,
                    'task': task,
                    'row_id': row_id,
                    'timestamp': timestamp,
                    'filename': json_file.name,
                    'llm_output': data.get('choices', [{}])[0].get('message', {}).get('content'),
                }
                
                if 'usage' in data:
                    usage = data['usage']
                    response_record.update({
                        'prompt_tokens': usage.get('prompt_tokens'),
                        'completion_tokens': usage.get('completion_tokens'),
                        'total_tokens': usage.get('total_tokens'),
                    })
                    
                    completion_details = usage.get('completion_tokens_details', {})
                    if completion_details:
                        response_record['reasoning_tokens'] = completion_details.get('reasoning_tokens')
                
                all_responses.append(response_record)
                
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error parsing {json_file}: {e}")
                continue
    
    if not all_responses:
        print("No valid response files found")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_responses)
    df = df.sort_values('row_id')
    # drop duplicates from multiple runs of the same row
    df = df.sort_values('timestamp').drop_duplicates('row_id', keep='first')
    
    return df


def extract_confidence(text):
    # nowdays models are following instructions pretty well, just in case though
    text = text.lower()
    # Check for direct starts
    if text.startswith('low'):
        return 'low'
    elif text.startswith('medium'):
        return 'medium'
    elif text.startswith('high'): 
        return 'high'
    
    if 'output:' in text:
        output_text = text.split('output:')[1].strip()
        if 'low' in output_text:
            return 'low'
        elif 'medium' in output_text:
            return 'medium'
        elif 'high' in output_text:
            return 'high'
        
    if 'low' in text:
        return 'low'
    elif 'medium' in text:
        return 'medium' 
    elif 'high' in text:
        return 'high'
    return None



def get_summarization_results(response_dir):
    deepseek_df = parse_response_task_model('summarize', 'deepseek', response_dir)
    deepseek_df = deepseek_df.rename(columns={'llm_output': 'content_ds'})
    o3mini_df = parse_response_task_model('summarize', 'o3mini', response_dir) 
    o3mini_df = o3mini_df.rename(columns={'llm_output': 'content_o3'})
    gpt4o_df = parse_response_task_model('summarize', 'gpt4o', response_dir)
    gpt4o_df = gpt4o_df.rename(columns={'llm_output': 'content_gpt4o'})
    merged_df = deepseek_df[['row_id', 'content_ds']]
    merged_df = pd.merge(merged_df, o3mini_df[['row_id', 'content_o3']], on='row_id', how='outer')
    merged_df = pd.merge(merged_df, gpt4o_df[['row_id', 'content_gpt4o']], on='row_id', how='outer')
    merged_df.to_csv("summarization_results.csv", index=False)
    return merged_df


