#%%
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
from datetime import datetime
import json
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
from config import MODEL_CONFIGS, TASK_CONFIGS


def compose_system_prompt(task_type: str):
    task_config = TASK_CONFIGS[task_type]
    with open(task_config["prompt_file"], "r", encoding='utf-8') as file:
        system_prompt = file.read()
    if task_config["shots"] > 0:
        with open(task_config["demo_file"], "r", encoding='utf-8') as file:
            demo_examples = json.load(file)
        system_prompt = system_prompt.format(**demo_examples)

    return system_prompt

def compose_classification_content(evidence, masked_statement):
    text = f"""Evidence excerpt: {evidence}\nConclusion statement: {masked_statement}\n\n\nOutput:"""
    return text
    
def compose_summarization_content(evidence):
    text = f"""Evidence excerpt: {evidence}\n\n\nConclusion statement:"""
    return text
    
def compose_reference_content(dir_name, header, masked_statement):
    if dir_name == 'IPCC_AR6_WGI_md':
        dir_name = 'IPCC AR6 WGI'
    elif dir_name == 'IPCC_AR6_WGII_md':
        dir_name = 'IPCC AR6 WGII'
    elif dir_name == 'IPCC_AR6_WGIII_md':
        dir_name = 'IPCC AR6 WGIII'
    text = f"""Section: {dir_name} - {header.strip()}\nStatement: {masked_statement}\n\n\nOutput:"""
    return text

def compose_content(row, task_type):
    if task_type == "summarize":
        return compose_summarization_content(row["evidence"])
    elif task_type == "reference_only":
        return compose_reference_content(row["dir_name"], row["header"], row["conclusion"])
    else:
        return compose_classification_content(row["evidence"], row["conclusion"])




class chatAPI:
    def __init__(self, model: str, api_key: str, system_prompt: str, temperature: float = 0.1):
        self.model = model
        self.config = MODEL_CONFIGS[model]
        self.temperature = temperature
        self.api_key = api_key
        self.base_url = self.config.get("base_url")
        if self.base_url:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            self.client = OpenAI(api_key=self.api_key)
        self.system_prompt = system_prompt
    
    def get_completion_o3(self, text, system_prompt=None, reasoning_effort= "medium"):
        """
        o3-mini has different input format (developer role,no temperature, and reasoning_effort)
        """
        if system_prompt is None:
            system_prompt = self.system_prompt

        messages = [{"role": "developer", "content": system_prompt},
                    {"role": "user", "content": text}]
        
        response = self.client.chat.completions.create(
            model=self.config["model_pointer"],
            messages=messages,
            reasoning_effort=reasoning_effort
        )
        return response
    
    def get_completion_standard(self, text, system_prompt=None):
        if system_prompt is None:
            system_prompt = self.system_prompt

        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}]
        response = self.client.chat.completions.create(
            model=self.config["model_pointer"],
            messages=messages,
            temperature=self.temperature
        )
        return response

    def get_completion(self, text, system_prompt=None):
        if self.model == "o3-mini":
            return self.get_completion_o3(text, system_prompt)
        else:
            return self.get_completion_standard(text, system_prompt)


    def create_completion_json(self, index, text, system_prompt=None):
        if system_prompt is None:
            system_prompt = self.system_prompt
        if self.model == "o3-mini":
            request = {
            "custom_id": f"request-{index}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.config["model_pointer"],
                "messages": [
                    {"role": self.config["role_key"], "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                "reasoning_effort": "medium",
                }
            }
        else:
            request = {
                "custom_id": f"request-{index}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.config["model_pointer"],
                    "messages": [
                        {"role": self.config["role_key"], "content": system_prompt},
                        {"role": "user", "content": text}
                    ],
                    "temperature": self.temperature,
                    "logprobs": True,
                }
            }
        return request
    
    def create_batch_jsonl(self, df, task_type, system_prompt=None, output_path=None):
        requests = []
        for index, row in df.iterrows():
            text = compose_content(row, task_type)
            request = self.create_completion_json(index, text, system_prompt)
            requests.append(request)
        requests_jsonl = "\n".join([json.dumps(request) for request in requests])
        
        if output_path is None:
            PROJECT_ROOT = Path(__file__).parent.parent
            output_path = PROJECT_ROOT / "jsonl" / f"{self.model}_{task_type}.jsonl"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as file:
            file.write(requests_jsonl)
        print(f"Saved {len(requests)} requests to {output_path}")
        return requests_jsonl



def process_single_row(i, row, task, model, out_dir, client):
    content = compose_content(row, task)
    response = client.get_completion(content)
    response_dict = response.model_dump()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    with open(out_dir / f"row{i}_{timestamp}.json", "w", encoding='utf-8') as f:
        json.dump(response_dict, f, ensure_ascii=False, indent=2)
    return response_dict


def process_single_row_thread_safe(i, row, task, model, out_dir, api_key, system_prompt):
    # create openai client within thread to avoid any risk of inteference
    client = chatAPI(model, api_key, system_prompt)
    content = compose_content(row, task)
    response = client.get_completion(content)
    response_dict = response.model_dump()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    with open(out_dir / f"row{i}_{timestamp}.json", "w", encoding='utf-8') as f:
        json.dump(response_dict, f, ensure_ascii=False, indent=2)
    return response_dict

#%%

if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).parent
    DATA_DIR = SCRIPT_DIR.parent / "data"
    df = pd.read_csv(DATA_DIR / "arguments.csv")
    #print(df.columns)
    #print(df.shape)
    #print(df.head())

    # prepare demo once in advance
    np.random.seed(2025)
    PROMPTS_DIR = Path(__file__).parent / "prompts"
    classification_demo = pd.concat([
        df[df['confidence'] == conf].sample(n=1) 
        for conf in df['confidence'].unique()
    ]).reset_index(drop=True)
    classification_json = {}
    for i, row in classification_demo.iterrows():
        classification_json[f"evidence_excerpt{i+1}"] = row["evidence"]
        classification_json[f"conclusion_statement{i+1}"] = row["masked_statement"]
        classification_json[f"true_confidence{i+1}"] = row["confidence"]
    with open(PROMPTS_DIR / "classification_demos.json", "w", encoding='utf-8') as f:
        json.dump(classification_json, f, ensure_ascii=False, indent=2)
    print("classification demo created")

    summarize_demo = df.sample(n=1).reset_index(drop=True)
    summarize_json = {
        "evidence_excerpt": summarize_demo["evidence"].iloc[0],
        "conclusion_statement": summarize_demo["conclusion"].iloc[0]
    }
    with open(PROMPTS_DIR / "summarize_demos.json", "w", encoding='utf-8') as f:
        json.dump(summarize_json, f, ensure_ascii=False, indent=2)



#%%
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path, override=True)

    RESPONSE_DIR = Path(__file__).parent.parent / "responses"
    RESPONSE_DIR.mkdir(parents=True, exist_ok=True)

    for task in TASK_CONFIGS:
        system_prompt = compose_system_prompt(task)
        for model in MODEL_CONFIGS:
            print(f"Running {task} with {model}")
            api_key = os.getenv(MODEL_CONFIGS[model]["api_key_env"])
            if api_key is None:
                raise ValueError(f"API key not found for {model}")

            response_dir = RESPONSE_DIR / f"{task}_{model}"
            response_dir.mkdir(parents=True, exist_ok=True)

            with ThreadPoolExecutor(max_workers=4) as ex:
                futures = [
                    ex.submit(process_single_row_thread_safe, i, row, task, model, response_dir, api_key, system_prompt)
                    for i, row in df.iterrows()
                ]
                for fut in as_completed(futures):
                    _ = fut.result()  

            '''
            client = chatAPI(model, api_key, system_prompt)
            client.create_batch_jsonl(df, task, system_prompt)
            for i, row in df.head(2).iterrows():
                process_single_row(i, row, task, model, response_dir, client)
                
                with open("content_check.txt", "a", encoding='utf-8') as f:
                    f.write(f"Running {task} with {model}\n")
                    f.write(f"System prompt: {system_prompt}\n\n\n")
                    f.write(f"Content: {content}\n\n\n")
                    f.write(f"Response: {response_dict}")
                    f.write("\n\n" + "="*80 + "\n\n")
            '''

                
    


# %%



# %%
