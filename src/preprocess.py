"""
This script processes raw PDF files downloaded from the Intergovernmental Panel on Climate Change (IPCC)
and produces clean markdown files ready for evidence-conclusion extraction.

Original reports are found here:
https://www.ipcc.ch/working-group/wg{1-3}/

The pipeline consists of three main steps:
1. Split large PDF reports into individual chapters based on bookmarks/table of contents
2. Convert the chapter PDFs to markdown format using magic-pdf (MinerU)
    Note: Output markdown may vary slightly depending on your environment
    Installation: https://github.com/opendatalab/MinerU
3. Clean markdown files by removing PDF artifacts while preserving content structure
"""

import fitz
import re
import os
import argparse
import subprocess
from pathlib import Path
import sys

sys.stdout.reconfigure(line_buffering=True) 


def sanitize_filename(title):
    """
    Sanitizes a string to create a valid and safe filename.
    
    Args:
        title (str): The original filename/title
        
    Returns:
        str: A sanitized filename that is safe for all operating systems
    """
    sanitized = re.sub(r'\s+', '_', title)
    sanitized = sanitized.replace('.', '')
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', sanitized)
    sanitized = ''.join(char for char in sanitized if ord(char) < 128)
    sanitized = re.sub(r'_+', '_', sanitized)
    sanitized = sanitized.strip('_')
    if len(sanitized) > 200:
        sanitized = sanitized[:200].rstrip('_')
        
    return sanitized or 'unnamed'


def extract_bookmarks(pdf_path):
    """
    Extracts bookmarks (outline) from a PDF and returns chapter start pages.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        list: List of (title, page) tuples representing chapters
    """
    doc = fitz.open(pdf_path)
    bookmarks = []
    for item in doc.get_toc():
        level, title, page = item  
        bookmarks.append((title, page - 1))  # Convert to 0-indexed
    doc.close()
    return bookmarks


def split_pdf_by_bookmarks(pdf_path, bookmarks, output_prefix="chapter", output_dir=None):
    """
    Splits the PDF into chapters based on bookmark pages, ensuring pages that are boundaries
    between chapters are included in both chapters.
    
    Args:
        pdf_path (str): Path to the PDF file
        bookmarks (list): List of (title, page) tuples
        output_prefix (str): Prefix for output files
        output_dir (str, optional): Directory to save split PDFs. If None, creates '{pdf_name}_chapters' 
                                  in the same directory as the input PDF
                                  
    Returns:
        str: Path to the output directory containing split PDFs
    """
    doc = fitz.open(pdf_path)
    num_pages = len(doc)
    
    if output_dir is None:
        # Get the path without extension
        pdf_path_no_ext = os.path.splitext(pdf_path)[0]
        # Create output directory by appending '_chapters'
        output_dir = f"{pdf_path_no_ext}_chapters"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Splitting PDF into {len(bookmarks)} chapters...")
    
    for i, (title, start_page) in enumerate(bookmarks):
        # For the end page, if we're not at the last bookmark,
        # use the start of the next section. Otherwise use the last page
        if i + 1 < len(bookmarks):
            # If next chapter starts at page X, include page X in current chapter
            end_page = bookmarks[i + 1][1]
        else:
            end_page = num_pages - 1  # -1 because we want inclusive page number
        
        # Create new document
        new_doc = fitz.open()
        new_doc.insert_pdf(doc, from_page=max(0, start_page), to_page=end_page)
        
        # Create sanitized filename
        clean_title = sanitize_filename(title)
        output_file = os.path.join(output_dir, f"{output_prefix}{i+1:02d}_{clean_title}.pdf")
            
        new_doc.save(output_file)
        new_doc.close()
        print(f"  Saved: {os.path.basename(output_file)} (pages {start_page+1} to {end_page+1})")
    
    doc.close()
    print(f"All {len(bookmarks)} chapters saved to: {output_dir}")
    return output_dir


def convert_pdfs_to_markdown(input_dir, output_dir=None):
    """
    Convert all PDFs in the specified directory to Markdown using magic-pdf (MinerU).
    
    Args:
        input_dir (str): Directory containing PDF files to convert
        output_dir (str, optional): Directory to save markdown files. If None, creates '{input_dir}_md'
        
    Returns:
        str: Path to the output directory containing markdown files
    """
    input_path = Path(input_dir).resolve()
    
    if output_dir is None:
        output_path = Path(str(input_path) + "_md").resolve()
    else:
        output_path = Path(output_dir).resolve()

    # Ensure the output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all PDF files in the directory
    pdf_files = list(input_path.glob('*.pdf'))
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return str(output_path)
    
    print(f"Converting {len(pdf_files)} PDFs to Markdown...")

    # Process all PDF files in the directory
    for pdf_file in pdf_files:
        if not pdf_file.exists():
            print(f"Warning: File not found: {pdf_file}")
            continue

        output_subdir = output_path / pdf_file.stem  # Create a subdirectory for each PDF
        os.makedirs(output_subdir, exist_ok=True)
        
        # Skip if output directory exists and contains files
        if output_subdir.exists() and any(output_subdir.iterdir()):
            print(f"  Skipping {pdf_file.name} - already converted")
            continue

        print(f"  Converting {pdf_file.name} to Markdown...")

        try:
            # Run magic-pdf command - it will create the output directory
            subprocess.run(["magic-pdf", "-p", str(pdf_file), "-o", str(output_subdir)], 
                         check=True, capture_output=True, text=True)
            print(f"  Successfully converted: {pdf_file.name}")
        except subprocess.CalledProcessError as e:
            print(f"  Error processing {pdf_file.name}: {e}")
        except FileNotFoundError:
            print(f"  Error: magic-pdf command not found. Please install MinerU.")
            print(f"  Installation instructions: https://github.com/opendatalab/MinerU")
            return str(output_path)

    print(f"All PDFs converted to Markdown in: {output_path}")
    return str(output_path)

def clean_markdown_files(markdown_dir):
    """
    Clean the markdown files in the specified directory by removing common artifacts
    from PDF-to-markdown conversion and standardizing formatting.
    
    Args:
        markdown_dir (str): Directory containing markdown files   
    """
    markdown_dir = Path(markdown_dir)
    if not markdown_dir.exists():
        print(f"Markdown directory not found: {markdown_dir}")
        return
    
    md_files = list(markdown_dir.rglob("*.md"))
    if not md_files:
        print(f"No markdown files found in {markdown_dir}")
        return
    
    print(f"Cleaning {len(md_files)} markdown files...")
    
    for md_file in md_files:
        try:
            with open(md_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            cleaned_content = _clean_markdown_content(content)
            
            if cleaned_content != content:
                with open(md_file, 'w', encoding='utf-8') as f:
                    f.write(cleaned_content)
                print(f"  Cleaned: {md_file.name}")
            else:
                print(f"  No changes: {md_file.name}")
                
        except Exception as e:
            print(f"  Error processing {md_file.name}: {e}")


def _clean_markdown_content(content):
    """
    Apply conservative cleaning operations to markdown content that are safe for 
    rule-based evidence-conclusion extraction.
    
    Args:
        content (str): Raw markdown content
        
    Returns:
        str: Cleaned markdown content
    """
    content = re.sub(r'\r\n', '\n', content)  
    content = re.sub(r'\r', '\n', content)    
    content = '\n'.join(line.rstrip() for line in content.split('\n'))
    content = re.sub(r'^\s*\d+\s*$', '', content, flags=re.MULTILINE)
    content = re.sub(r'^\s*[ivxlcdm]+\s*$', '', content, flags=re.MULTILINE | re.IGNORECASE)
    content = re.sub(r'^[\s\.\-_]{3,}$', '', content, flags=re.MULTILINE)
    content = re.sub(r'^.*\.{6,}\s*\d+\s*$', '', content, flags=re.MULTILINE)
    unicode_artifacts = {
        '\uf0b7': '•',   
        '\uf020': ' ',   
        '\uf0d8': '•',   
        '\u200b': '',    
        '\u00a0': ' ',   
        '\u2019': "'",   
        '\u2018': "'",   
        '\u201c': '"',   
        '\u201d': '"',   
        '\u2013': '–',   
        '\u2014': '—',   
        '\u2026': '...', 
    }
    for artifact, replacement in unicode_artifacts.items():
        content = content.replace(artifact, replacement)
    
    content = re.sub(r'([a-z])-\s*\n\s*([a-z])', r'\1\2', content)
    content = re.sub(r'[ \t]{2,}', ' ', content)
    content = re.sub(r'^(#{1,6})\s+(.+)$', r'\1 \2', content, flags=re.MULTILINE)
    content = re.sub(r'^\s+$', '', content, flags=re.MULTILINE)
    content = re.sub(r'\n{4,}', '\n\n\n', content)
    content = re.sub(r'!\[.*?\]\(images/[^)]+\)', '', content)
    content = content.strip()
    
    if content and not content.endswith('\n'):
        content += '\n'
    
    return content

    

def preprocess_ipcc_pdf(pdf_path, output_dir=None, skip_split=False, skip_convert=False, skip_clean=False):
    """
    Args:
        pdf_path (str): Path to the IPCC PDF file
        output_dir (str, optional): Base output directory
        skip_split (bool): Skip the PDF splitting step
        skip_convert (bool): Skip the markdown conversion step
        skip_clean (bool): Skip the markdown cleaning step
        
    Returns:
        tuple: (chapters_dir, markdown_dir) paths
    """
    pdf_path = os.path.abspath(pdf_path)
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    print(f"Starting IPCC PDF preprocessing pipeline...")
    print(f"Input file: {pdf_path}")
    
    # Determine output directories
    if output_dir:
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        chapters_dir = os.path.join(output_dir, f"{base_name}_chapters")
        markdown_dir = os.path.join(output_dir, f"{base_name}_md")
    else:
        chapters_dir = None  # Will be auto-generated
        markdown_dir = None   # Will be auto-generated
    
    # Step 1: Split PDF into chapters (unless skipped)
    if skip_split:
        print("Skipping PDF splitting step...")
        if not chapters_dir or not os.path.exists(chapters_dir):
            raise ValueError("Cannot skip splitting - chapters directory not found")
    else:
        print("\n=== Step 1: Splitting PDF into chapters ===")
        bookmarks = extract_bookmarks(pdf_path)
        
        if not bookmarks:
            print("No bookmarks found in PDF. Cannot split into chapters.")
            print("Proceeding with the full PDF...")
            # Create a temporary directory with the full PDF
            chapters_dir = chapters_dir or f"{os.path.splitext(pdf_path)[0]}_chapters"
            os.makedirs(chapters_dir, exist_ok=True)
            
            # Copy the original PDF to the chapters directory
            import shutil
            full_pdf_name = f"full_{os.path.basename(pdf_path)}"
            shutil.copy2(pdf_path, os.path.join(chapters_dir, full_pdf_name))
            print(f"Copied full PDF to: {chapters_dir}")
        else:
            print(f"Found {len(bookmarks)} bookmarks/chapters")
            chapters_dir = split_pdf_by_bookmarks(pdf_path, bookmarks, output_dir=chapters_dir)
    
    # Step 2: Convert PDFs to Markdown (unless skipped)
    if skip_convert:
        print("Skipping markdown conversion step...")
        markdown_dir = markdown_dir or f"{chapters_dir}_md"
    else:
        print("\n=== Step 2: Converting PDFs to Markdown ===")
        markdown_dir = convert_pdfs_to_markdown(chapters_dir, markdown_dir)
        
        # Step 3: Clean the markdown files (unless skipped)
        if not skip_clean:
            print("\n=== Step 3: Cleaning Markdown Files ===")
            clean_markdown_files(markdown_dir)
        else:
            print("Skipping markdown cleaning step...")
    
    print(f"\n=== Preprocessing Complete ===")
    print(f"Chapters directory: {chapters_dir}")
    print(f"Markdown directory: {markdown_dir}")
    
    return chapters_dir, markdown_dir


def main():
    parser = argparse.ArgumentParser(
        description="Process IPCC PDF files into markdown format",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("pdf_file", help="Path to the IPCC PDF file to process")
    parser.add_argument("--output-dir", "-o", help="Base output directory (default: same as input file)")
    parser.add_argument("--skip-split", action="store_true", 
                       help="Skip the PDF splitting step")
    parser.add_argument("--skip-convert", action="store_true", 
                       help="Skip the markdown conversion step")
    parser.add_argument("--skip-clean", action="store_true", 
                       help="Skip the markdown cleaning step")
    
    args = parser.parse_args()
    
    try:
        chapters_dir, markdown_dir = preprocess_ipcc_pdf(
            args.pdf_file,
            output_dir=args.output_dir,
            skip_split=args.skip_split,
            skip_convert=args.skip_convert,
            skip_clean=args.skip_clean
        )
        print(f"\nPreprocessing completed successfully!")
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 