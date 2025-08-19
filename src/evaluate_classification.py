#%%
from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import tiktoken
encoding = tiktoken.get_encoding("cl100k_base")

import utils
from config import MODEL_CONFIGS, TASK_CONFIGS

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"
df= pd.read_csv(DATA_DIR / "arguments.csv")


#%%
# Create histogram of confidence values
plt.figure(figsize=(10, 6))
ax = df['confidence'].value_counts().reindex(['high', 'medium', 'low']).plot(kind='bar')
plt.title('Distribution of Confidence Levels')
plt.xlabel('Confidence Level')
plt.ylabel('Count')
plt.xticks(rotation=0)

# Add count labels on top of each bar
for i, v in enumerate(df['confidence'].value_counts().reindex(['high', 'medium', 'low'])):
    ax.text(i, v, str(v), ha='center', va='bottom')

plt.tight_layout()
plt.show()

#%%
# Create histogram of evidence lengths
plt.figure(figsize=(12, 6))
df['evidence_length'] = df['evidence'].apply(lambda x: len(encoding.encode(x)))
plt.hist(df['evidence_length'], bins=50, edgecolor='black')
plt.title('Distribution of Evidence Lengths')
plt.xlabel('Number of Tokens')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Print summary statistics
print("\nEvidence Length Statistics (in tokens):")
print(df['evidence_length'].describe())

#%%
# Create histogram of masked statement lengths
plt.figure(figsize=(12, 6))
df['masked_length'] = df['masked_statement'].apply(lambda x: len(encoding.encode(x)))
plt.hist(df['masked_length'], bins=50, edgecolor='black')
plt.title('Distribution of Masked Statement Lengths')
plt.xlabel('Number of Tokens')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Print summary statistics
print("\nMasked Statement Length Statistics (in tokens):")
print(df['masked_length'].describe())


def merge_responses(df, task, model, response_dir="responses"):
    responses_df = utils.parse_response_task_model(task, model, response_dir=response_dir)
    responses_df['row_id'] = responses_df['row_id'].astype(int)
    merged_df = pd.merge(df, responses_df, left_index=True, right_on='row_id', how='inner')
    merged_df['llm_confidence'] = merged_df['llm_output'].apply(utils.extract_confidence)
    merged_df = merged_df.dropna(subset=['llm_confidence'])
    return merged_df


    
#%%


RESPONSE_DIR = SCRIPT_DIR.parent / "responses"


for task in ['zeroshot_contextual', 'fewshot_contextual','reference_only']:
    for model in MODEL_CONFIGS.keys():
        response_df = utils.parse_response_task_model(task, model, response_dir=RESPONSE_DIR)
        print(f"Response shape: {response_df.shape}")

        print(f"Evaluating {task} with {model}")
        merged_df = merge_responses(df, task, model, response_dir=RESPONSE_DIR)
        print(f"Merged shape: {merged_df.shape}")
        print(merged_df.head())

        print("\nClassification Report:")
        print(classification_report(merged_df['confidence'], merged_df['llm_confidence'], target_names=['low', 'medium', 'high']))
