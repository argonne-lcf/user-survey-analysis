import os
import logging
import argparse
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s - %(levelname)s - %(message)s',
   datefmt='%m-%y %H:%M:%S',
   handlers=[
      logging.StreamHandler(),
      logging.FileHandler('survey_extraction.log')
   ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def parse_arguments():
   """Parse command line arguments."""
   parser = argparse.ArgumentParser(description='Extract survey data from Excel file to CSV')
   parser.add_argument(
      '-e',
      '--excel-file',
      type=str,
      help='Path to the Excel file containing survey data',
      required=True
   )
   parser.add_argument(
      '-o',
      '--output-file',
      type=str,
      help='Path to the output CSV file',
      default='survey_responses.csv'
   )
   parser.add_argument(
      '-t',
      '--text-questions',
      type=str,
      help='Path to JSON file containing list of text questions to process',
      required=True
   )
   parser.add_argument(
      '--log-level',
      type=str,
      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
      default='INFO',
      help='Set the logging level'
   )
   return parser.parse_args()

def load_survey_data(excel_path):
   """Load the survey data from Excel file."""
   try:
      logger.info(f"Loading survey data from: {excel_path}")
      # List all sheet names for debugging
      all_sheets = pd.ExcelFile(excel_path).sheet_names
      logger.info(f"Available sheet names: {all_sheets}")
      # First read the contents sheet
      contents_df = pd.read_excel(excel_path, sheet_name='Contents')
      logger.info(f"Successfully loaded Contents sheet with {len(contents_df)} rows")
      
      # Process the contents sheet to get question mappings
      question_mappings = process_contents_sheet(contents_df)
      
      # Load all question sheets
      question_sheets = {}
      for sheet_name in question_mappings['sheet_names']:
         try:
            sheet_df = pd.read_excel(excel_path, sheet_name=sheet_name)
            question_sheets[sheet_name] = sheet_df
            logger.info(f"Successfully loaded sheet {sheet_name} with {len(sheet_df)} responses")
         except Exception as e:
            logger.error(f"Error loading sheet {sheet_name}: {str(e)}")
            raise
      
      return {
         'contents': contents_df,
         'question_mappings': question_mappings,
         'question_sheets': question_sheets
      }
   except Exception as e:
      logger.error(f"Error loading Excel file: {str(e)}")
      raise

def process_contents_sheet(contents_df):
   """Process the contents sheet to extract question mappings."""
   sections = []
   questions = []
   sheet_names = []
   
   # Process columns in pairs (question column and hyperlink column)
   for i in range(0, len(contents_df.columns), 2):
      if i + 1 >= len(contents_df.columns):
         break
         
      # Get the question column and hyperlink column
      question_col = contents_df.iloc[:, i]
      hyperlink_col = contents_df.iloc[:, i + 1]
      
      # Get section title (header of question column)
      section_title = contents_df.columns[i]
      logger.info(f"Processing section: {section_title}")
      sections.append(section_title)
      
      # Process each row in the pair of columns
      for j in range(len(question_col)):
         
         question = question_col.iloc[j]
         sheet_name = hyperlink_col.iloc[j]

         if pd.isna(question) or pd.isna(sheet_name):
            logger.debug(f"Skipping row {i} in column pair {j} due to missing data")
            continue
         
         logger.debug(f"Processing question: {question} -> sheet: {sheet_name}")
         questions.append(question)
         sheet_names.append(sheet_name)
   
   logger.info(f"Processed {len(sheet_names)} questions from contents sheet")
   logger.info(f"Sheet names found: {sheet_names}")
   
   return {
      'sections': sections,
      'questions': questions,
      'sheet_names': sheet_names
   }

def extract_responses_to_csv(survey_data, output_file, text_questions):
    """Extract all responses and save to CSV format.
    
    Args:
        survey_data (dict): Dictionary containing survey data
        output_file (str): Path to output CSV file
        text_questions (list): List of question objects containing question_number and question_text
    """
    # Create a list to store all responses
    all_responses = []
    
    # Create a mapping of question numbers to question text from the JSON file
    question_mapping = {q['question_number']: q['question_text'] for q in text_questions}
    
    # Process each question sheet
    for sheet_name, sheet_df in survey_data['question_sheets'].items():
        # Skip if not in text_questions list
        if sheet_name not in question_mapping:
            logger.debug(f"Skipping sheet {sheet_name} as it's not in text questions list")
            continue
            
        question_text = question_mapping[sheet_name]
        
        # Get responses from the second column (assuming first column is question number)
        if len(sheet_df.columns) >= 2:
            responses = sheet_df.iloc[:, 1].dropna().tolist()
            
            # Add each response to our list
            for response in responses:
                if isinstance(response, str) and response.strip():
                    # Replace newlines with semicolons
                    cleaned_response = response.strip().replace('\n', '; ')
                    all_responses.append({
                        'question_number': sheet_name,
                        'question_text': question_text,
                        'user_response_text': cleaned_response
                    })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(all_responses)
    df.to_csv(output_file, index=False)
    logger.info(f"Saved {len(df)} responses to {output_file}")

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Set logging level from arguments
    logger.setLevel(args.log_level)
    
    # Load text questions from JSON file
    try:
        with open(args.text_questions, 'r') as f:
            text_questions = json.load(f)
        logger.info(f"Loaded {len(text_questions)} text questions from {args.text_questions}")
    except Exception as e:
        logger.error(f"Error loading text questions file: {str(e)}")
        raise
    
    # Load the survey data
    survey_data = load_survey_data(args.excel_file)
    
    # Log the structure of the loaded data
    logger.info("Loaded survey data structure:")
    logger.info(f"Number of sections: {len(survey_data['question_mappings']['sections'])}")
    logger.info(f"Number of questions: {len(survey_data['question_mappings']['questions'])}")
    logger.info(f"Number of sheets: {len(survey_data['question_sheets'])}")
    
    # Extract responses to CSV
    extract_responses_to_csv(survey_data, args.output_file, text_questions)

if __name__ == "__main__":
    main() 