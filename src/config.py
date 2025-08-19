from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

MODEL_CONFIGS = {
    "gpt-4o": {
        "api_key_env": "OPENAI_API_KEY",
        "role_key": "system",
        "model_pointer": "gpt-4o-2024-08-06"
    },
    "o3-mini": {
        "api_key_env": "OPENAI_API_KEY",
        "role_key": "developer",  # o3 is odd
        "model_pointer": "o3-mini"
    },
    "deepseek-R1": {
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY", 
        "role_key": "system",
        "model_pointer": "deepseek-reasoner"
    }
}

TASK_CONFIGS = {
    "zeroshot_contextual": {
        "prompt_file": SCRIPT_DIR / "prompts/zeroshot_contextual.txt",
        "shots": 0,
        "demo_file": None
    },
    "fewshot_contextual": {
        "prompt_file": SCRIPT_DIR / "prompts/fewshot_contextual.txt",
        "shots": 3,
        "demo_file": SCRIPT_DIR / "prompts/classification_demos.json"
    },
    "reference_only": {
        "prompt_file": SCRIPT_DIR / "prompts/reference_only.txt",
        "shots": 0,
        "demo_file": None
    },
    "summarize": {
        "prompt_file": SCRIPT_DIR / "prompts/summarize.txt",
        "shots": 1,
        "demo_file": SCRIPT_DIR / "prompts/summarize_demos.json"
    }
}
