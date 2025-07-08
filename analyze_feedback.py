import os
import logging
import argparse
from dotenv import load_dotenv
import pandas as pd
import json
from openai import OpenAI
from datetime import datetime

# Configure logging
logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s - %(levelname)s - %(message)s',
   datefmt='%m-%y %H:%M:%S',
   handlers=[
      logging.StreamHandler(),
      logging.FileHandler('feedback_analysis.log')
   ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# ALCF primer for context
ALCF_PRIMER = """The Argonne Leadership Computing Facility (ALCF) is a U.S. Department of Energy (DOE) Office of Science user facility that provides supercomputing resources and expertise to the scientific and engineering community. ALCF's mission is to accelerate major scientific discoveries and engineering breakthroughs for humanity by designing and providing world-leading computing facilities in partnership with the computational science community. The facility supports a wide range of research areas including climate science, materials science, physics, chemistry, and more. ALCF provides access to high-performance computing systems, scientific software, and user support services to enable researchers to advance their work."""

def parse_arguments():
   """Parse command line arguments."""
   parser = argparse.ArgumentParser(description='Analyze open-ended user-feedback from CSV file, producing another CSV file with the results.')
   parser.add_argument('-i', '--input', default='survey_responses.csv', help='Input CSV file with survey responses')
   parser.add_argument('-o', '--output', default='analysis_results.csv', help='Output CSV file for analysis results')
   parser.add_argument('-l', '--llm-config', default='data/llm_questions.json', help='Path to LLM questions that can be asked for each Q&A pair, in json format')
   parser.add_argument('-t', '--text-questions', default='data/text_questions.json', help='Path to text questions that were asked of the users and which LLM questions should be asked for each response, in json format')
   parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                      help='Set the logging level')
   return parser.parse_args()

def load_config_files(llm_config_path, text_config_path):
   """Load the LLM questions and text questions configuration files."""
   with open(llm_config_path, 'r') as f:
      llm_questions = json.load(f)
   with open(text_config_path, 'r') as f:
      text_questions = json.load(f)
   return llm_questions, text_questions

def validate_llm_response(response, categories):
   """Validate that the LLM response matches one of the expected categories."""
   if isinstance(response, str):
      response = response.strip()
      # Convert categories to lowercase for case-insensitive comparison
      categories_lower = [cat.lower() for cat in categories]
      response_lower = response.lower()
      
      if response_lower in categories_lower:
         # Return the original response (preserving case)
         return response
      # Check if response is a comma-separated list of categories
      if ',' in response:
         responses = [r.strip().lower() for r in response.split(',')]
         if all(r in categories_lower for r in responses):
            return response
   return None

def analyze_response(client, question_text, user_response, llm_question_config):
   """Analyze a single response using the LLM."""
   prompt = f"""{ALCF_PRIMER}

Survey Question: {question_text}
User Response: {user_response}

{llm_question_config['text']}

Allowed categories: {', '.join(llm_question_config['categories'])}

{llm_question_config['answer_format']}"""

   try:
      response = client.chat.completions.create(
         model=os.environ.get("OPENAI_MODEL_NAME", "gpt-4"),
         messages=[{"role": "user", "content": prompt}]
      )
      result = response.choices[0].message.content.strip()
      
      # Validate the response
      validated_result = validate_llm_response(result, llm_question_config['categories'])
      if validated_result is None:
         logger.warning(f"Invalid LLM response: {result}; expected categories: {llm_question_config['categories']}")
         
         # Retry once with the invalid response included in the prompt
         retry_prompt = f"""{ALCF_PRIMER}

Survey Question: {question_text}
User Response: {user_response}

{llm_question_config['text']}

Allowed categories: {', '.join(llm_question_config['categories'])}

{llm_question_config['answer_format']}

Note: Your previous response "{result}" was not one of the allowed categories. Please choose from the allowed categories only."""
         
         try:
            retry_response = client.chat.completions.create(
               model=os.environ.get("OPENAI_MODEL_NAME", "gpt-4"),
               messages=[{"role": "user", "content": retry_prompt}],
               temperature=0.1
            )
            retry_result = retry_response.choices[0].message.content.strip()
            
            # Validate the retry response
            validated_retry_result = validate_llm_response(retry_result, llm_question_config['categories'])
            if validated_retry_result is None:
               logger.warning(f"Retry also failed with invalid response: {retry_result}; expected categories: {llm_question_config['categories']}")
               return "INVALID_RESPONSE"
            return validated_retry_result
         except Exception as retry_e:
            logger.error(f"Error during retry: {str(retry_e)}")
            return "INVALID_RESPONSE"
      
      return validated_result
   except Exception as e:
      logger.error(f"Error analyzing response: {str(e)}")
      return "ERROR"

def save_results(results, output_file, all_llm_types):
   """Save results to CSV file."""
   results_df = pd.DataFrame(results)
   results_df.to_csv(output_file, index=False)
   logger.info(f"Results saved to {output_file}")

def main():
   # Parse command line arguments
   args = parse_arguments()
   
   # Set logging level from arguments
   logger.setLevel(args.log_level)
   logging.getLogger('httpx').setLevel(logging.WARNING)
   
   # Load configuration files
   llm_questions, text_questions = load_config_files(args.llm_config, args.text_questions)
   
   # Initialize OpenAI client
   client = OpenAI(
      api_key=os.environ.get("OPENAI_API_KEY", "local-key"),
      base_url=os.environ.get("OPENAI_API_BASE", "http://localhost:7285/v1")
   )
   
   # Read survey responses
   try:
      responses_df = pd.read_csv(args.input)
   except Exception as e:
      logger.error(f"Error reading input file: {str(e)}")
      return
   
   # Create a list to store results
   results = []
   
   # Get all possible LLM question types for column names
   all_llm_types = list(llm_questions.keys())

   row_counter = 0
   total_rows = len(responses_df)
   
   # Process each response
   for _, row in responses_df.iterrows():
      row_counter += 1
      if row_counter % 10 == 0:
         logger.info(f"Processed {row_counter} of {total_rows} rows")
         # Save results periodically
         save_results(results, args.output, all_llm_types)
      
      question_number = row['question_number']
      question_text = row['question_text']
      user_response = row['user_response_text']
      
      # Skip empty responses
      if pd.isna(user_response) or user_response.strip() == "":
         logger.warning(f"Skipping empty response for question {question_number}")
         continue
      
      # Find the question configuration
      question_config = next((q for q in text_questions if q['question_number'] == question_number), None)
      if not question_config:
         logger.warning(f"No configuration found for question {question_number}")
         continue
      
      # Create a result dictionary with base information
      result = {
         'question_number': question_number,
         'question_text': question_text,
         'user_response': user_response
      }
      
      # Initialize all LLM analysis columns as empty
      for llm_type in all_llm_types:
         result[llm_type] = None
      
      # Analyze the response for each configured LLM question
      for llm_question_type in question_config['llm_questions']:
         llm_config = llm_questions[llm_question_type]
         analysis_result = analyze_response(client, question_text, user_response, llm_config)
         result[llm_question_type] = analysis_result
      
      results.append(result)
   
   # Save final results to CSV
   save_results(results, args.output, all_llm_types)
   logger.info(f"Analysis complete. Final results saved to {args.output}")

if __name__ == "__main__":
   main() 