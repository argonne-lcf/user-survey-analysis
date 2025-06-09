import pandas as pd
import os
import argparse

def split_analysis_by_question(input_file='analysis_results.csv', output_dir='question_analysis'):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Get unique question numbers
    question_numbers = df['question_number'].unique()
    
    # Split and save each question's data to a separate file
    for q_num in question_numbers:
        # Filter data for this question
        question_data = df[df['question_number'] == q_num]
        
        # Create question directory (replace any special characters)
        safe_q_num = q_num.replace('.', '_')
        question_dir = os.path.join(output_dir, f'question_{safe_q_num}')
        os.makedirs(question_dir, exist_ok=True)
        
        # Save all responses for this question
        all_responses_file = os.path.join(question_dir, 'all_responses.csv')
        question_data.to_csv(all_responses_file, index=False)
        print(f"Created {all_responses_file} with {len(question_data)} responses")
        
        # Filter negative sentiment responses
        negative_responses = question_data[question_data['sentiment'] == 'negative']
        if not negative_responses.empty:
            # Save all negative responses
            negative_file = os.path.join(question_dir, 'all_negative_responses.csv')
            negative_responses.to_csv(negative_file, index=False)
            print(f"Created {negative_file} with {len(negative_responses)} negative responses")
            
            # Get all unique topics from negative responses
            all_topics = set()
            for topics in negative_responses['topic'].dropna():
                all_topics.update(t.strip() for t in str(topics).split(','))
            
            # Create a file for each topic's negative responses
            for topic in all_topics:
                topic_responses = negative_responses[negative_responses['topic'].str.contains(topic, na=False)]
                topic_file = os.path.join(question_dir, f'negative_{topic}_responses.csv')
                topic_responses.to_csv(topic_file, index=False)
                print(f"Created {topic_file} with {len(topic_responses)} responses")

def parse_args():
    parser = argparse.ArgumentParser(description='Split analysis results by question into separate CSV files')
    parser.add_argument('-i', '--input', 
                      default='analysis_results.csv',
                      help='Input CSV file (default: analysis_results.csv)')
    parser.add_argument('-o', '--output-dir',
                      default='question_analysis',
                      help='Output directory for split files (default: question_analysis)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    split_analysis_by_question(input_file=args.input, output_dir=args.output_dir) 