"""
This script processes cleaned markdown files from the preprocessing pipeline and extracts
evidence-conclusion pairs with confidence levels to create the final arguments dataset.

The pipeline:
1. Parse markdown files into sections based on headers
2. Extract evidence-conclusion pairs (last paragraph as conclusion)
3. Parse confidence levels from conclusions, separate distinct confidence statements into multiple rows
4. Apply data cleaning and filtering
5. Generate final arguments.csv dataset
"""

#%%
import os
import pandas as pd
from pathlib import Path
import re
import nltk
from nltk.tokenize import sent_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def extract_conclusions(text):
    """
    Extract conclusion from text by taking the last paragraph.
    Everything before the last paragraph is considered evidence.
    Only returns conclusion if there are multiple paragraphs.
    
    Args:
        text (str): The text content to extract conclusion from
        
    Returns:
        tuple: (evidence, conclusion) where:
            - evidence: string containing all paragraphs except the last, joined by double newlines
            - conclusion: string containing the last paragraph
        None: if text has 1 or fewer paragraphs
    """
    # Split content into paragraphs (split by double newline)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    # Return None if only one or two paragraphs
    if len(paragraphs) <= 2:
        return None
        
    # Join all paragraphs except last with double newlines for evidence
    evidence = '\n\n'.join(paragraphs[:-1])
    conclusion = paragraphs[-1]
    
    return evidence, conclusion

def extract_sections(markdown_text, file_name):
    """
    Divides markdown text into sections based on headers (#).
    Returns a list of dictionaries containing file_name, header, content, and parsed conclusions.
    """
    lines = markdown_text.split('\n')
    sections = []
    current_header = None
    current_content = []
    
    for line in lines:
        header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
        
        if header_match:
            if current_header is not None:
                content = '\n'.join(current_content).strip()
                conclusions = extract_conclusions(content)
                if conclusions is not None:
                    evidence, conclusion = conclusions
                    section_dict = {
                        'file_name': file_name,
                        'header': current_header,
                        'content': content,
                        'evidence': evidence,
                        'conclusion': conclusion
                    }
                    sections.append(section_dict)
            
            current_header = line.strip()
            current_content = []
        else:
            current_content.append(line)
    
    # Don't forget the last section
    if current_header is not None:
        content = '\n'.join(current_content).strip()
        conclusions = extract_conclusions(content)
        if conclusions is not None:
            evidence, conclusion = conclusions
            section_dict = {
                'file_name': file_name,
                'header': current_header,
                'content': content,
                'evidence': evidence,
                'conclusion': conclusion
            }
            sections.append(section_dict)
    
    return sections


#%%
'''
test_file_A = "E:/Github/LLMClimateSynthesis/test_files/IPCC_AR6_WGI_md/chapter113_222_Volcanic_Aerosol_Forcing.md"
test_file_B = "E:/Github/LLMClimateSynthesis/test_files/IPCC_AR6_WGII_md/IPCC_AR6_WGII_Chapter04.md"

with open(test_file_A, 'r', encoding='utf-8') as f:
    content = f.read()
sections_A = extract_sections(content, "chapter113_222_Volcanic_Aerosol_Forcing")
print(sections_A)

with open(test_file_B, 'r', encoding='utf-8') as f:
    content = f.read()
sections_B = extract_sections(content, "IPCC_AR6_WGII_Chapter04")
print(sections_B[0])
'''


#%%
def parse_confidence(section):
    """
    Parse a conclusion text to extract both the statement and its confidence level.
    Handles IPCC confidence levels: very low, low, medium, high, very high
    If multiple confidence statements exist, splits into separate conclusions.
    
    Args:
        text (str): The conclusion text
        
    Returns:
        list[dict]: List of dictionaries, each containing:
            - masked_statement: The main conclusion statement with confidence level masked
            - confidence: The confidence level (if any)
        None: If no conclusion is found or if no confidence statements are found
    """
    text = section.get('conclusion')
    
    if text is None:
        return None

    confidence_pattern = r'\(?(?:very )?(?:low|medium|high) confidence\)?'
    matches = list(re.finditer(confidence_pattern, text, re.IGNORECASE))
    
    if not matches:
        return None
    
    sentences = sent_tokenize(text)
    
    # Find which sentences contain confidence statements
    confidence_indices = []
    for i, sentence in enumerate(sentences):
        if re.search(confidence_pattern, sentence, re.IGNORECASE):
            confidence_indices.append(i)
    
    results = []
    # Process each confidence statement and its associated sentences
    for i, conf_idx in enumerate(confidence_indices):
        # Get all sentences from previous confidence statement (or start) up to this one
        start_idx = confidence_indices[i-1] + 1 if i > 0 else 0
        statement_sentences = sentences[start_idx:conf_idx + 1]
        
        full_text = ' '.join(statement_sentences)
        
        match = re.search(confidence_pattern, statement_sentences[-1], re.IGNORECASE)
        if match:
            # Extract confidence level by removing optional brackets and 'confidence'
            confidence = match.group(0).replace('(', '').replace(')', '').replace(' confidence', '').strip()
            # Create masked statement
            statement = re.sub(confidence_pattern, "(<MASKED> confidence)", full_text, flags=re.IGNORECASE)
            
            results.append({
                'masked_statement': statement.strip(),
                'confidence': confidence.lower(),
            })
    
    return results if results else None


#%%
'''
parsed_confidence_A = parse_confidence(sections_A[0])

# Flatten the results into a single list of confidence statements
parsed_confidence_B = []
for section in sections_B:
    confidence_statements = parse_confidence(section)
    if confidence_statements is not None:
        for statement in confidence_statements:
            parsed_confidence_B.append({
                'file_name': section['file_name'],
                'header': section['header'],
                'content': section['content'],
                'evidence': section['evidence'],
                'conclusion': section['conclusion'],
                'masked_statement': statement['masked_statement'],
                'confidence': statement['confidence']
        })
'''
#%%
def parse_single_markdown(file_path):
    """
    Parse a single markdown file to extract sections and their confidence statements.
    
    Args:
        file_path (str): Path to the markdown file to parse
        
    Returns:
        list: List of dictionaries, each containing:
            - dir_name: Name of the directory
            - file_name: Name of the source file
            - header: Section header
            - content: Full section content
            - evidence: Evidence paragraphs before the conclusion
            - conclusion: Conclusion paragraph
            - masked_statement: Statement with confidence level masked as (<MASKED> confidence)
            - confidence: Extracted confidence level (very low/low/medium/high/very high)
    """

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    path = Path(file_path)
    dir_name = path.parent.stem
    sections = extract_sections(content, path.stem)

    parsed_sections = []
    for section in sections:
        confidence_statements = parse_confidence(section)
        if confidence_statements is not None:
            for statement in confidence_statements:
                parsed_sections.append({
                    'dir_name': dir_name,
                    'file_name': section['file_name'],
                    'header': section['header'],
                    'content': section['content'],
                    'evidence': section['evidence'],
                    'conclusion': section['conclusion'],
                    'masked_statement': statement['masked_statement'],
                    'confidence': statement['confidence']
                })
    
    return parsed_sections



#%%
'''
test_file_A = "E:/Github/LLMClimateSynthesis/test_files/IPCC_AR6_WGI_md/chapter112_221_Solar_and_Orbital_Forcing.md"
test_file_B = "E:/Github/LLMClimateSynthesis/test_files/IPCC_AR6_WGII_md/IPCC_AR6_WGII_Chapter04.md"

parsed_A = parse_single_markdown(test_file_A)
parsed_B = parse_single_markdown(test_file_B)
'''
#%%
def process_markdown_files(directory):
    """
    Process all markdown files in a directory and extract sections with confidence statements.
    
    Args:
        directory (str): Path to directory containing markdown files
        
    Returns:
        pandas.DataFrame: DataFrame containing all extracted sections with columns:
            - dir_name: Name of the directory
            - file_name: Name of the source file
            - header: Section header
            - content: Full section content
            - evidence: Evidence paragraphs before the conclusion
            - conclusion: Conclusion paragraph
            - masked_statement: Statement with confidence level masked
            - confidence: Extracted confidence level
    """
    # Convert directory to Path object
    dir_path = Path(directory)
    all_sections = []
    
    # Process each markdown file
    for md_file in dir_path.glob('**/*.md'):
        print(f"\nProcessing: {md_file}")
        try:
            parsed_sections = parse_single_markdown(str(md_file))
            if parsed_sections:
                all_sections.extend(parsed_sections)
        except Exception as e:
            print(f"Error processing {md_file}: {str(e)}")
            continue
    
    # Convert to DataFrame
    if not all_sections:
        return pd.DataFrame(columns=['dir_name', 'file_name', 'header', 'content', 'evidence', 
                                   'conclusion', 'masked_statement', 'confidence'])
    
    df = pd.DataFrame(all_sections)
    
    # Ensure consistent column order
    columns = ['dir_name', 'file_name', 'header', 'content', 'evidence', 
              'conclusion', 'masked_statement', 'confidence']
    df = df.reindex(columns=columns)
    
    return df

#%%
'''
markdown_dir = "E:/Github/LLMClimateSynthesis/test_files/test_md"
df = process_markdown_files(markdown_dir)
output_dir = Path("E:/Github/LLMClimateSynthesis/test_files/test_md")
output_path = output_dir / "parsed_sections.csv"
df.to_csv(output_path, index=False)
'''

    
#%%
master_dir = "E:/Github/LLMClimateSynthesis/"
for dir in ["IPCC_AR6_WGI", "IPCC_AR6_WGII", "IPCC_AR6_WGIII"]:
    path = master_dir + dir + "_md"
    print(f"Processing {path}")
    df = process_markdown_files(path)
    output_path = master_dir + dir + ".csv"
    df.to_csv(output_path, index=False)
    print(f"\nProcessed {len(df)} sections")
    print(f"Results saved to: {output_path}")

# Combine all three CSV files into one master file
print("\nCombining CSV files...")
combined_df = pd.DataFrame()
for dir in ["IPCC_AR6_WGI", "IPCC_AR6_WGII", "IPCC_AR6_WGIII"]:
    csv_path = master_dir + dir + ".csv"
    df = pd.read_csv(csv_path)
    combined_df = pd.concat([combined_df, df], ignore_index=True)

save_path = master_dir + "data/arguments.csv"
combined_df.to_csv(save_path, index=False)
print(f"Combined {len(combined_df)} total sections saved to: {save_path}")


#%%
'''
Now clean the data and some validation
'''
df = pd.read_csv(save_path)
print(df.shape)

#%%
df.columns = df.columns.str.strip()

def print_unique_headers(df):
    print("Unique headers in dataset:")
    unique_headers = df.header.unique()
    for i, header in enumerate(unique_headers, 1):
        print(f"{i}. {header}")
    print(f"\nTotal unique headers: {len(unique_headers)}")

print_unique_headers(df)

# %%
# Remove section references from conclusions
def remove_section_references(text):
    if pd.isna(text):
        return text
    cleaned_text = re.sub(r'\([^)]*Section[^)]*\)', '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text

df['conclusion'] = df['conclusion'].apply(remove_section_references)
df['masked_statement'] = df['masked_statement'].apply(remove_section_references)


# %%
# Remove rows with empty headers or appendix headers
df = df[df.header.str.strip() != '']  
df = df[df.header.str.strip() != '#']  
df = df[~df.header.str.startswith('# Appendix')]  
df = df[~df.header.str.startswith('# Cross-Chapter')] 
df = df[~df.header.str.startswith('# Cross Chapter')] 
df = df[~df.header.str.startswith('# Summary statement')] 
df = df[~df.header.str.startswith('# Executive Summary')] 
df = df[~df.header.str.startswith('# Box')] 
df = df[~df.header.str.startswith('# Legend')] 
df = df[~df.header.str.startswith('# FAQ')] 
df = df[~df.header.str.startswith('# Chapter')] 

print_unique_headers(df)
# %%
# Check if last paragraph contains conclusion phrase and return the matching phrase
def has_conclusion_phrase(text):
    conclusion_phrases = [
        'to conclude',
        'in conclusion', 
        'we conclude',
        'it is concluded',
        'in summary',
        'overall,'
    ]
    
    # Split into paragraphs and get last one
    paragraphs = text.split('\n\n')
    last_para = paragraphs[-1].lower() if paragraphs else ''
    
    # Check if any conclusion phrase exists in last paragraph
    for phrase in conclusion_phrases:
        if phrase in last_para:
            return phrase
    return None

# Add column with matching conclusion phrase (None if no match)
df['conclusion_phrase'] = df['content'].apply(has_conclusion_phrase)

print(f"\nRows with conclusion phrase: {df['conclusion_phrase'].notna().sum()}")
print(f"Total rows: {len(df)}")

df_short = df[df['conclusion_phrase'].notna()]


# %%
# Filter to keep only high, medium, and low confidence levels
df_short = df_short[df_short['confidence'].isin(['high', 'medium', 'low'])]

print(f"\nRows after filtering confidence levels: {len(df_short)}")
print("\nConfidence level counts:")
print(df_short['confidence'].value_counts())


# %%
# Remove duplicate masked statements
df_unique = df_short.drop_duplicates(subset=['masked_statement'])

print(f"\nRows after removing duplicate masked statements: {len(df_unique)}")
print(f"Number of duplicates removed: {len(df_short) - len(df_unique)}")

# Save deduplicated dataset
df_unique.to_csv(save_path, index=False)