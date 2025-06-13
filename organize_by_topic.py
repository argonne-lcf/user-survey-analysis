import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
import argparse
import logging
import os
from pathlib import Path
from itertools import product
from openai import OpenAI
from dotenv import load_dotenv
from pptx import Presentation
from pptx.util import Inches, Pt
import io

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Set global matplotlib font size
plt.rcParams.update({'font.size': 18})

# Load environment variables
load_dotenv()

# ALCF primer for context
ALCF_PRIMER = """The Argonne Leadership Computing Facility (ALCF) is a U.S. Department of Energy (DOE) Office of Science user facility that provides supercomputing resources and expertise to the scientific and engineering community. ALCF's mission is to accelerate major scientific discoveries and engineering breakthroughs for humanity by designing and providing world-leading computing facilities in partnership with the computational science community. The facility supports a wide range of research areas including climate science, materials science, physics, chemistry, and more. ALCF provides access to high-performance computing systems, scientific software, and user support services to enable researchers to advance their work."""

def parse_args():
    parser = argparse.ArgumentParser(description='Organize analysis results by topic')
    parser.add_argument('-i', '--input', type=str, default='analysis_results.csv',
                      help='Input CSV file path (default: analysis_results.csv)')
    parser.add_argument('-o', '--output-dir', type=str, default='topic_analysis',
                      help='Output directory for topic analysis (default: topic_analysis)')
    return parser.parse_args()

def plot_comma_separated_histogram(series, title, figsize=(10, 6)):
    """Create a histogram for comma-separated values."""
    all_values = []
    for value in series:
        if pd.notna(value):
            all_values.extend([v.strip() for v in str(value).split(',')])
    
    value_counts = Counter(all_values)
    
    plt.figure(figsize=figsize)
    bars = plt.bar(value_counts.keys(), value_counts.values())
    
    plt.title(title, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height}',
                ha='center', va='bottom')
    
    return plt.gcf()

def plot_regular_histogram(series, title, figsize=(10, 6)):
    """Create a regular histogram."""
    plt.figure(figsize=figsize)
    value_counts = series.value_counts()
    bars = plt.bar(value_counts.index, value_counts.values)
    
    plt.title(title, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height}',
                ha='center', va='bottom')
    
    return plt.gcf()

def plot_2d_histogram(df, col1, col2, title, figsize=(12, 8)):
    """Create a 2D histogram between two columns."""
    col1_values = set()
    col2_values = set()
    
    for val1, val2 in zip(df[col1], df[col2]):
        if pd.notna(val1) and pd.notna(val2):
            col1_values.update([v.strip() for v in str(val1).split(',')])
            col2_values.update([v.strip() for v in str(val2).split(',')])
    
    col1_values = sorted(list(col1_values))
    col2_values = sorted(list(col2_values))
    
    hist_data = np.zeros((len(col2_values), len(col1_values)))
    
    for val1, val2 in zip(df[col1], df[col2]):
        if pd.notna(val1) and pd.notna(val2):
            val1_list = [v.strip() for v in str(val1).split(',')]
            val2_list = [v.strip() for v in str(val2).split(',')]
            
            for v1, v2 in product(val1_list, val2_list):
                i = col1_values.index(v1)
                j = col2_values.index(v2)
                hist_data[j, i] += 1
    
    plt.figure(figsize=figsize)
    sns.heatmap(hist_data, 
                xticklabels=col1_values,
                yticklabels=col2_values,
                cmap='YlOrRd',
                annot=True,
                fmt='.0f',
                cbar_kws={'label': 'Count'})
    
    plt.title(title, pad=20)
    plt.xlabel(col1.title())
    plt.ylabel(col2.title())
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    return plt.gcf()

def analyze_feedback(client, feedback_data):
    """Analyze the feedback using the LLM and generate a summary and action items."""
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
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error analyzing feedback: {str(e)}")
        return "ERROR: Failed to analyze feedback"

def create_powerpoint(topic, markdown_content, plots_dir):
    """Create a PowerPoint presentation with the markdown content and plots."""
    prs = Presentation()
    
    # Title slide
    title_slide = prs.slides.add_slide(prs.slide_layouts[0])
    title = title_slide.shapes.title
    subtitle = title_slide.placeholders[1]
    title.text = f"Feedback Analysis: {topic.title()}"
    subtitle.text = "ALCF User Survey Results"
    
    # Set font size for title and subtitle
    title.text_frame.paragraphs[0].font.size = Pt(24)
    subtitle.text_frame.paragraphs[0].font.size = Pt(12)
    
    # Summary slide
    summary_slide = prs.slides.add_slide(prs.slide_layouts[1])
    title = summary_slide.shapes.title
    content = summary_slide.placeholders[1]
    title.text = "Summary"
    title.text_frame.paragraphs[0].font.size = Pt(24)
    
    # Extract summary from markdown content
    summary = markdown_content.split("TOP THREE ACTION ITEMS:")[0].replace("SUMMARY:", "").strip()
    content.text = summary
    for paragraph in content.text_frame.paragraphs:
        paragraph.font.size = Pt(12)
    
    # Action items slide
    action_slide = prs.slides.add_slide(prs.slide_layouts[1])
    title = action_slide.shapes.title
    content = action_slide.placeholders[1]
    title.text = "Top Three Action Items"
    title.text_frame.paragraphs[0].font.size = Pt(24)
    
    # Extract action items from markdown content
    action_items = markdown_content.split("TOP THREE ACTION ITEMS:")[1].strip()
    content.text = action_items
    for paragraph in content.text_frame.paragraphs:
        paragraph.font.size = Pt(12)
    
    # Add plot slides
    for plot_file in plots_dir.glob("*.png"):
        slide = prs.slides.add_slide(prs.slide_layouts[5])
        title = slide.shapes.title
        title.text = plot_file.stem.replace("_", " ").title()
        title.text_frame.paragraphs[0].font.size = Pt(24)
        
        # Add the plot
        left = Inches(1)
        top = Inches(2)
        width = Inches(8)
        height = Inches(4.5)
        slide.shapes.add_picture(str(plot_file), left, top, width, height)
    
    return prs

def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")
    
    # Read the CSV file
    logger.info(f"Reading input file: {args.input}")
    df = pd.read_csv(args.input)
    
    # Initialize OpenAI client
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", "local-key"),
        base_url=os.environ.get("OPENAI_API_BASE", "http://localhost:7285/v1")
    )
    
    # Get all unique topics
    all_topics = set()
    for topics in df['topic'].dropna():
        all_topics.update(t.strip() for t in str(topics).split(','))
    
    # Process each topic
    for topic in all_topics:
        logger.info(f"Processing topic: {topic}")
        topic_dir = output_dir / topic.replace(" ", "_")
        topic_dir.mkdir(exist_ok=True)
        
        # Filter responses for this topic
        topic_responses = df[df['topic'].str.contains(topic, na=False)]
        topic_responses.to_csv(topic_dir / 'all_responses.csv', index=False)
        
        # Filter negative responses
        negative_responses = topic_responses[topic_responses['sentiment'] == 'negative']
        negative_responses.to_csv(topic_dir / 'negative_responses.csv', index=False)
        
        # Create plots directory (now always, not just for negative)
        plots_dir = topic_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # Generate plots from ALL responses for this topic
        columns_to_plot = {
            'sentiment': 'regular',
            'machine': 'comma_separated',
            'feedback': 'regular',
            'emotion': 'regular',
            'actionability': 'regular',
            'specificity': 'regular',
            'impact_area': 'regular',
            'resolution_status': 'regular'
        }
        
        for col, plot_type in columns_to_plot.items():
            title = f'{col.title()} Distribution - {topic.title()}'
            if plot_type == 'comma_separated':
                fig = plot_comma_separated_histogram(topic_responses[col], title)
            else:
                fig = plot_regular_histogram(topic_responses[col], title)
            
            fig.savefig(plots_dir / f'{col}_distribution.png', bbox_inches='tight', dpi=300)
            plt.close(fig)
        
        # Generate 2D histograms from ALL responses for this topic
        correlations = [
            ('machine', 'emotion', 'Machine vs Emotion'),
            ('feedback', 'actionability', 'Feedback Type vs Actionability'),
            ('impact_area', 'resolution_status', 'Impact Area vs Resolution Status'),
            ('specificity', 'actionability', 'Specificity vs Actionability')
        ]
        
        for col1, col2, title_prefix in correlations:
            title = f'{title_prefix} Distribution - {topic.title()}'
            fig = plot_2d_histogram(topic_responses, col1, col2, title)
            fig.savefig(plots_dir / f'{col1}_{col2}_2d_histogram.png', bbox_inches='tight', dpi=300)
            plt.close(fig)
        
        if not negative_responses.empty:
            # Analyze negative feedback
            analysis_result = analyze_feedback(client, negative_responses)
            
            # Save markdown file
            with open(topic_dir / 'negative_feedback_analysis.md', 'w') as f:
                f.write("# Feedback Analysis Results\n\n")
                f.write(analysis_result)
            
            # Create PowerPoint presentation
            prs = create_powerpoint(topic, analysis_result, plots_dir)
            prs.save(topic_dir / f'{topic.replace(" ", "_")}_analysis.pptx')
            
            logger.info(f"Completed processing for topic: {topic}")
        else:
            logger.info(f"No negative responses found for topic: {topic}")

if __name__ == '__main__':
    main() 