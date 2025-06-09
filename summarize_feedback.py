import os
import logging
import argparse
from dotenv import load_dotenv
import pandas as pd
from openai import OpenAI
from pathlib import Path
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m-%y %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('feedback_summary.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# ALCF primer for context
ALCF_PRIMER = """The Argonne Leadership Computing Facility (ALCF) is a U.S. Department of Energy (DOE) Office of Science user facility that provides supercomputing resources and expertise to the scientific and engineering community. ALCF's mission is to accelerate major scientific discoveries and engineering breakthroughs for humanity by designing and providing world-leading computing facilities in partnership with the computational science community. The facility supports a wide range of research areas including climate science, materials science, physics, chemistry, and more. ALCF provides access to high-performance computing systems, scientific software, and user support services to enable researchers to advance their work."""

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze and summarize user feedback from CSV files.')
    parser.add_argument('-i', '--input', required=True, help='Input glob pattern for CSV files with user feedback (e.g., "data/*.csv")')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Set the logging level')
    return parser.parse_args()

def analyze_feedback(client, feedback_data):
    """Analyze the feedback using the LLM and generate a summary and action items."""
    # Prepare the feedback data for analysis
    feedback_text = "\n\n".join([
        f"Response {i+1}: {row['user_response']}"
        for i, row in feedback_data.iterrows()
    ])

    prompt = f"""{ALCF_PRIMER}

Below are user feedback responses about ALCF's software development environment:

{feedback_text}

Please provide:
1. A concise summary of the main themes and concerns expressed in the feedback
2. The top three actionable items that ALCF should consider to address these concerns

Format your response as follows:

SUMMARY:
[Your summary here]

TOP THREE ACTION ITEMS:
1. [First action item]
2. [Second action item]
3. [Third action item]
"""

    try:
        response = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL_NAME", "gpt-4"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7  # Slightly higher temperature for more creative insights
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error analyzing feedback: {str(e)}")
        return "ERROR: Failed to analyze feedback"

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Set logging level from arguments
    logger.setLevel(args.log_level)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    
    # Initialize OpenAI client
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", "local-key"),
        base_url=os.environ.get("OPENAI_API_BASE", "http://localhost:7285/v1")
    )
    
    # Find all matching files
    input_files = glob.glob(args.input)
    if not input_files:
        logger.error(f"No files found matching pattern: {args.input}")
        return
    
    logger.info(f"Found {len(input_files)} files matching pattern: {args.input}")
    
    # Process each file
    for input_file in input_files:
        try:
            logger.info(f"Processing file: {input_file}")
            feedback_df = pd.read_csv(input_file)
            logger.info(f"Successfully read {len(feedback_df)} feedback entries from {input_file}")
            
            # Analyze the feedback
            analysis_result = analyze_feedback(client, feedback_df)
            
            # Generate output filename
            input_path = Path(input_file)
            output_path = input_path.with_suffix('.md')
            
            # Write results to markdown file
            with open(output_path, 'w') as f:
                f.write("# Feedback Analysis Results\n\n")
                f.write(analysis_result)
            logger.info(f"Analysis results saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error processing file {input_file}: {str(e)}")
            continue

if __name__ == "__main__":
    main() 