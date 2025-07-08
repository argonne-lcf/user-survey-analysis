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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Generate analysis plots from CSV data')
    parser.add_argument('-i', '--input', type=str, default='analysis_results.csv',
                      help='Input CSV file path (default: analysis_results.csv)')
    parser.add_argument('-o', '--output-dir', type=str, default='plots',
                      help='Output directory for plots (default: plots)')
    return parser.parse_args()

# Function to create histograms for comma-separated values
def plot_comma_separated_histogram(series, title, figsize=(10, 6)):
    # Split and count all values
    all_values = []
    for value in series:
        if pd.notna(value):
            all_values.extend([v.strip() for v in str(value).split(',')])
    
    # Count occurrences
    value_counts = Counter(all_values)
    
    # Create the plot
    plt.figure(figsize=figsize)
    bars = plt.bar(value_counts.keys(), value_counts.values())
    
    # Customize the plot
    plt.title(title, pad=20, fontsize=20)
    plt.xticks(rotation=45, ha='right', fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Categories', fontsize=20)
    plt.ylabel('Count', fontsize=20)
    plt.tight_layout()
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height}',
                ha='center', va='bottom')
    
    return plt.gcf()

# Function to create regular histograms
def plot_regular_histogram(series, title, figsize=(10, 6)):
    plt.figure(figsize=figsize)
    value_counts = series.value_counts()
    bars = plt.bar(value_counts.index, value_counts.values)
    
    plt.title(title, pad=20, fontsize=20)
    plt.xticks(rotation=45, ha='right', fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Categories', fontsize=20)
    plt.ylabel('Count', fontsize=20)
    plt.tight_layout()
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height}',
                ha='center', va='bottom')
    
    return plt.gcf()

def plot_2d_histogram(df, col1, col2, title, figsize=(12, 8)):
    """
    Create a 2D histogram between two columns that may contain comma-separated values.
    """
    # Get unique values for each column
    col1_values = set()
    col2_values = set()
    
    for val1, val2 in zip(df[col1], df[col2]):
        if pd.notna(val1) and pd.notna(val2):
            col1_values.update([v.strip() for v in str(val1).split(',')])
            col2_values.update([v.strip() for v in str(val2).split(',')])
    
    # Convert to sorted lists for consistent ordering
    col1_values = sorted(list(col1_values))
    col2_values = sorted(list(col2_values))
    
    # Create the 2D histogram data
    hist_data = np.zeros((len(col2_values), len(col1_values)))
    
    # Fill the histogram data
    for val1, val2 in zip(df[col1], df[col2]):
        if pd.notna(val1) and pd.notna(val2):
            val1_list = [v.strip() for v in str(val1).split(',')]
            val2_list = [v.strip() for v in str(val2).split(',')]
            
            # Count all combinations
            for v1, v2 in product(val1_list, val2_list):
                i = col1_values.index(v1)
                j = col2_values.index(v2)
                hist_data[j, i] += 1
    
    # Create the plot
    plt.figure(figsize=figsize)
    sns.heatmap(hist_data, 
                xticklabels=col1_values,
                yticklabels=col2_values,
                cmap='YlOrRd',
                annot=True,
                fmt='.0f',
                cbar_kws={'label': 'Count'})
    
    plt.title(title, pad=20, fontsize=20)
    plt.xlabel(col1.title(), fontsize=20)
    plt.ylabel(col2.title(), fontsize=20)
    plt.xticks(rotation=45, ha='right', fontsize=20)
    plt.yticks(rotation=0, fontsize=20)
    plt.tight_layout()
    
    return plt.gcf()

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")
    
    # Read the CSV file
    logger.info(f"Reading input file: {args.input}")
    df = pd.read_csv(args.input)
    
    # Normalize classification columns to lowercase to avoid case inconsistencies in plots
    classification_columns = ['sentiment', 'topic', 'machine', 'feedback', 'emotion', 
                            'actionability', 'specificity', 'ticket_status', 'software_topic']
    for col in classification_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower()
    logger.info("Normalized classification columns to lowercase")
    
    # Create a mapping of question numbers to their text
    question_texts = df[['question_number', 'question_text']].drop_duplicates().set_index('question_number')['question_text'].to_dict()
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create plots for each relevant column
    columns_to_plot = {
        'sentiment': 'regular',
        'topic': 'comma_separated',
        'machine': 'comma_separated',
        'feedback': 'regular',
        'emotion': 'regular',
        'actionability': 'regular',
        'specificity': 'regular',
        'ticket_status': 'regular',
        'software_topic': 'comma_separated'
    }
    
    # Get unique question numbers
    question_numbers = df['question_number'].unique()
    
    # Create plots for each question
    for question in question_numbers:
        logger.info(f"Generating plots for question: {question}")
        question_df = df[df['question_number'] == question]
        question_text = question_texts[question]
        
        # Create a subdirectory for this question
        question_dir = output_dir / question
        question_dir.mkdir(exist_ok=True)
        
        # Create individual plots for this question
        for col, plot_type in columns_to_plot.items():
            logger.info(f"Generating plot for {col} in question {question}")
            title = f'{col.title()} Distribution\n{question}: {question_text}'
            if plot_type == 'comma_separated':
                fig = plot_comma_separated_histogram(question_df[col], title)
            else:
                fig = plot_regular_histogram(question_df[col], title)
            
            # Save individual plot
            output_path = question_dir / f'{col}_distribution.png'
            fig.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
            logger.info(f"Saved plot to: {output_path}")
        
        # Create 2D histograms for this question
        correlations = [
            ('topic', 'emotion', 'Topic vs Emotion'),
            ('sentiment', 'emotion', 'Sentiment vs Emotion'),
            ('feedback', 'actionability', 'Feedback Type vs Actionability'),
            ('specificity', 'actionability', 'Specificity vs Actionability'),
            ('machine', 'topic', 'Machine vs Topic')
        ]
        
        for col1, col2, title_prefix in correlations:
            logger.info(f"Generating 2D histogram between {col1} and {col2} for question {question}")
            title = f'{title_prefix} Distribution\n{question}: {question_text}'
            fig = plot_2d_histogram(question_df, col1, col2, title)
            output_path = question_dir / f'{col1}_{col2}_2d_histogram.png'
            fig.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
            logger.info(f"Saved 2D histogram to: {output_path}")
        
        # Create combined plot for this question
        logger.info(f"Generating combined plot for question {question}")
        fig, axes = plt.subplots(len(columns_to_plot), 1, figsize=(12, 4*len(columns_to_plot)))
        fig.suptitle(f'Analysis Results Distribution\n{question}: {question_text}', fontsize=20, y=1.02)
        
        for i, (col, plot_type) in enumerate(columns_to_plot.items()):
            if plot_type == 'comma_separated':
                plot_comma_separated_histogram(question_df[col], f'{col.title()} Distribution')
            else:
                plot_regular_histogram(question_df[col], f'{col.title()} Distribution')
        
        plt.tight_layout()
        # combined_output_path = question_dir / 'analysis_histograms.png'
        # plt.savefig(combined_output_path, bbox_inches='tight', dpi=300)
        # plt.close()
        # logger.info(f"Saved combined plot to: {combined_output_path}")
    
    # Create overall plots (across all questions)
    logger.info("Generating overall plots across all questions")
    overall_dir = output_dir / 'overall'
    overall_dir.mkdir(exist_ok=True)
    
    # Create individual plots for overall data
    for col, plot_type in columns_to_plot.items():
        logger.info(f"Generating overall plot for {col}")
        title = f'{col.title()} Distribution - Overall (All Questions)'
        if plot_type == 'comma_separated':
            fig = plot_comma_separated_histogram(df[col], title)
        else:
            fig = plot_regular_histogram(df[col], title)
        
        # Save individual plot
        output_path = overall_dir / f'{col}_distribution.png'
        fig.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        logger.info(f"Saved plot to: {output_path}")
    
    # Create overall 2D histograms
    for col1, col2, title_prefix in correlations:
        logger.info(f"Generating overall 2D histogram between {col1} and {col2}")
        title = f'{title_prefix} Distribution - Overall (All Questions)'
        fig = plot_2d_histogram(df, col1, col2, title)
        output_path = overall_dir / f'{col1}_{col2}_2d_histogram.png'
        fig.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        logger.info(f"Saved 2D histogram to: {output_path}")

if __name__ == '__main__':
    main() 