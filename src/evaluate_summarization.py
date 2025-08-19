#%%
from openai import OpenAI
from pathlib import Path
import os
import re
import pandas as pd
from dotenv import load_dotenv

import numpy as np
from rouge import Rouge
from bert_score import BERTScorer

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


# might need to change these slightly if running on server with GPU or just google colab
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"

df= pd.read_csv(DATA_DIR / "arguments.csv")
summarization_df = pd.read_csv("summarization_results.csv")
df = pd.merge(df, summarization_df, left_index=True, right_on='row_id', how='inner')
#%%
def rogue_compare_text_columns(df, col1, col2):
    """
    Calculate ROUGE scores between two text columns in a DataFrame using the rouge package.

    Args:
        df (pandas.DataFrame): DataFrame containing the text columns
        col1 (str): Name of the first text column
        col2 (str): Name of the second text column

    Returns:
        dict: Dictionary containing average ROUGE scores
        pandas.DataFrame: DataFrame with row-by-row ROUGE scores
    """
    # A bit of error handling
    if col1 not in df.columns or col2 not in df.columns:
        raise ValueError(f"Columns {col1} and {col2} must exist in the DataFrame")
    df_clean = df.dropna(subset=[col1, col2])

    
    rouge = Rouge()
    results_df = pd.DataFrame(index=df_clean.index)
    rouge1_f = []
    rouge1_p = []
    rouge1_r = []
    rouge2_f = []
    rouge2_p = []
    rouge2_r = []
    rougeL_f = []
    rougeL_p = []
    rougeL_r = []

    for i, (idx, row) in enumerate(df_clean.iterrows()):
        try:
            text1 = str(row[col1]).strip()
            text2 = str(row[col2]).strip()

            if not text1 or not text2:
                continue

            scores = rouge.get_scores(text1, text2)[0]

            results_df.at[idx, 'rouge1_precision'] = scores['rouge-1']['p']
            results_df.at[idx, 'rouge1_recall'] = scores['rouge-1']['r']
            results_df.at[idx, 'rouge1_f1'] = scores['rouge-1']['f']

            results_df.at[idx, 'rouge2_precision'] = scores['rouge-2']['p']
            results_df.at[idx, 'rouge2_recall'] = scores['rouge-2']['r']
            results_df.at[idx, 'rouge2_f1'] = scores['rouge-2']['f']

            results_df.at[idx, 'rougeL_precision'] = scores['rouge-l']['p']
            results_df.at[idx, 'rougeL_recall'] = scores['rouge-l']['r']
            results_df.at[idx, 'rougeL_f1'] = scores['rouge-l']['f']

            rouge1_p.append(scores['rouge-1']['p'])
            rouge1_r.append(scores['rouge-1']['r'])
            rouge1_f.append(scores['rouge-1']['f'])

            rouge2_p.append(scores['rouge-2']['p'])
            rouge2_r.append(scores['rouge-2']['r'])
            rouge2_f.append(scores['rouge-2']['f'])

            rougeL_p.append(scores['rouge-l']['p'])
            rougeL_r.append(scores['rouge-l']['r'])
            rougeL_f.append(scores['rouge-l']['f'])

        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue

    avg_scores = {
        'rouge-1': {
            'p': np.mean(rouge1_p) if rouge1_p else 0,
            'r': np.mean(rouge1_r) if rouge1_r else 0,
            'f': np.mean(rouge1_f) if rouge1_f else 0
        },
        'rouge-2': {
            'p': np.mean(rouge2_p) if rouge2_p else 0,
            'r': np.mean(rouge2_r) if rouge2_r else 0,
            'f': np.mean(rouge2_f) if rouge2_f else 0
        },
        'rouge-l': {
            'p': np.mean(rougeL_p) if rougeL_p else 0,
            'r': np.mean(rougeL_r) if rougeL_r else 0,
            'f': np.mean(rougeL_f) if rougeL_f else 0
        }
    }

    return avg_scores, results_df

rogue_scores_ds, detailed_scores_ds = rogue_compare_text_columns(df, 'conclusion', 'content_ds')
print(rogue_scores_ds)

rogue_scores_o3, detailed_scores_o3 = rogue_compare_text_columns(df, 'conclusion', 'content_o3')
print(rogue_scores_o3)

rogue_scores_gpt4o, detailed_scores_gpt4o = rogue_compare_text_columns(df, 'conclusion', 'content_gpt4o')
print(rogue_scores_gpt4o)



#%%
def bert_score_detailed_report(df, llm_col, ref_col):
    """
    Calculate BERTScore metrics and provide detailed reports.

    Args:
        df (pandas.DataFrame): DataFrame containing the text columns
        llm_col (str): Name of the LLM-generated text column
        ref_col (str): Name of the reference text column

    Returns:
        dict: Dictionary with average scores
        pandas.DataFrame: DataFrame with detailed scores per row
    """
    working_df = df.copy()
    working_df = working_df.dropna(subset=[llm_col, ref_col])

    scorer = BERTScorer(lang="en")

    working_df[llm_col] = working_df[llm_col].astype(str)
    working_df[ref_col] = working_df[ref_col].astype(str)

    working_df = working_df[
        ~((working_df[llm_col].str.lower() == 'nan') |
          (working_df[ref_col].str.lower() == 'nan'))
    ]

    # Calculate scores for each row individually to handle errors
    scores_df = pd.DataFrame(index=working_df.index)
    scores_df['precision'] = float('nan')
    scores_df['recall'] = float('nan')
    scores_df['f1'] = float('nan')

    all_P = []
    all_R = []
    all_F1 = []

    for idx, row in working_df.iterrows():
        try:
            llm_text = row[llm_col]
            ref_text = row[ref_col]

            if not llm_text.strip() or not ref_text.strip():
                continue

            P, R, F1 = scorer.score([llm_text], [ref_text])

            scores_df.at[idx, 'precision'] = P.item()
            scores_df.at[idx, 'recall'] = R.item()
            scores_df.at[idx, 'f1'] = F1.item()

            all_P.append(P.item())
            all_R.append(R.item())
            all_F1.append(F1.item())

        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            print(f"LLM text: {llm_text[:50]}...")
            print(f"Ref text: {ref_text[:50]}...")
            continue

    avg_scores = {
        'precision': np.mean(all_P) if all_P else float('nan'),
        'recall': np.mean(all_R) if all_R else float('nan'),
        'f1': np.mean(all_F1) if all_F1 else float('nan')
    }

    return avg_scores, scores_df

bert_scores_4o, detailed_scores = bert_score_detailed_report(df, 'content_gpt4o', 'conclusion')
print(bert_scores_4o)

bert_scores_o3, detailed_scores = bert_score_detailed_report(df, 'content_o3', 'conclusion')
print(bert_scores_o3)

bert_scores_ds, detailed_scores = bert_score_detailed_report(df, 'content_ds', 'conclusion')
print(bert_scores_ds)


#%%
client = OpenAI(api_key=OPENAI_API_KEY)


EVALUATION_PROMPT_TEMPLATE = """
Scientific Conclusion Evaluation
You are an expert evaluator assessing the quality of LLM-generated scientific conclusions. Your task is to evaluate how well a model has synthesized scientific literature according to specific criteria. For each submission, you will be provided with:
1. The original scientific passage
2. The LLM-generated conclusion
3. The expected guidelines for the conclusion

Evaluation Criteria (Score each on a scale of 1-5):

{criteria}

Evaluation Process:

{steps}

Now evaluate:

**Original Passage**: {passage}

**LLM-Generated Conclusion**: {conclusion}

{guidelines_section}

Your evaluation must follow this exact format:
**Evaluation**:
* **Relevance**: Score: X/5
* **Faithfulness**: Score: X/5
* **Confidence Level Appropriateness**: Score: X/5
**Overall Score**: X.XX/5 (weighted calculation)
"""

# Metric 1: Relevance
RELEVANCE_CRITERIA = """
1. Relevance (Weight: 30%)
* 5: Perfectly captures the core scientific findings and key quantitative details
* 4: Identifies most important findings but misses minor details
* 3: Captures some key findings but omits several important elements
* 2: Focuses primarily on peripheral information rather than central findings
* 1: Fails to identify the main scientific findings
"""

# Metric 2: Faithfulness
FAITHFULNESS_CRITERIA = """
2. Faithfulness (Weight: 40%)
* 5: Completely faithful to the original text with no misrepresentations or distortions
* 4: Largely faithful with only minor inaccuracies that don't affect the core meaning
* 3: Generally faithful but contains some misrepresentations of moderate importance
* 2: Contains significant misrepresentations or fabricated information
* 1: Fundamentally misrepresents the scientific content or contradicts the original text
"""

# Metric 3: Confidence Level Appropriateness
CONFIDENCE_CRITERIA = """
3. Confidence Level Appropriateness (Weight: 30%)
* 5: All confidence levels expressed in conclusion statement strictly follow from scientific text
* 4: Contain confidence level statements with minor inaccuracies or somewhat dubious nature
* 3: Preserves some uncertainty statements but omits or misrepresents others
* 2: Significantly understates or overstates confidence in findings
* 1: Completely misrepresents or omits critical uncertainty statements and confidence levels
"""

# Evaluation Steps
EVALUATION_STEPS = """
1. First, carefully read the original scientific passage and identify:
   * The main scientific findings and their importance
   * The precise wording of quantitative details
   * All statements of uncertainty, limitations, or confidence levels
2. Review the LLM-generated conclusion, evaluating:
   * Whether it focuses on the most important findings
   * Whether it accurately represents the original content
   * Whether it appropriately preserves uncertainty and confidence levels
3. For each criterion, provide:
   * A score (1-5) ONLY
4. Calculate a weighted overall score using the weights provided
"""


def get_scientific_eval_score(
    criteria: str, steps: str, passage: str, conclusion: str, metric_name: str, guidelines_section: str = ""
):
    prompt = EVALUATION_PROMPT_TEMPLATE.format(
        criteria=criteria,
        steps=steps,
        passage=passage,
        conclusion=conclusion,
        guidelines_section=guidelines_section
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1, # less than 1 to increase stability
        max_tokens=200,  # Increased to capture score and justification snippet
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    content = response.choices[0].message.content
    # Extract score from content (format: "Score: X/5")
    score_pattern = r"Score: (\d+)/5"
    match = re.search(score_pattern, content)
    if match:
        return int(match.group(1))
    else:
        # Fallback in case the expected pattern isn't found
        return content

evaluation_metrics = {
    "Relevance": (RELEVANCE_CRITERIA, EVALUATION_STEPS),
    "Faithfulness": (FAITHFULNESS_CRITERIA, EVALUATION_STEPS),
    "Confidence": (CONFIDENCE_CRITERIA, EVALUATION_STEPS)
}
weights = {"Relevance": 0.3, "Faithfulness": 0.4, "Confidence": 0.3}


results = []
for idx, row in df.iterrows():
    original_text = row['evidence']  

    for model_name, content_column in [
        ("DeepSeek-R1", "content_ds"),
        ("o3-mini", "content_o3"),
        ("GPT-4o", "content_gpt4o")
    ]:
        summary = row[content_column]

        if pd.isna(summary) or summary.strip() == "":
            continue

        model_scores = {"file_name": row['file_name'], "model": model_name}

        for eval_type, (criteria, steps) in evaluation_metrics.items():
            score = get_scientific_eval_score(
                criteria, steps, original_text, summary, eval_type
            )
            model_scores[f"{eval_type}_score"] = score
            model_scores[f"{eval_type}_weight"] = weights[eval_type]

        weighted_sum = sum(
            model_scores[f"{metric}_score"] * model_scores[f"{metric}_weight"]
            for metric in weights.keys()
        )
        model_scores["overall_score"] = round(weighted_sum, 2)

        results.append(model_scores)

eval_df = pd.DataFrame(results)
model_avg_scores = eval_df.groupby("model")[
    ["Relevance_score", "Faithfulness_score", "Confidence_score", "overall_score"]
].mean().round(2)

print(model_avg_scores)