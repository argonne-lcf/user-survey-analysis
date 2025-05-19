# User Survey Analysis

Framework that uses LLMs to process open-ended user-survey responses to help provide a quick overview of sentiment, subject matter, and machines mentioned.

# Setup

Start with a recent python3 release.

```shell
python -m venv venv
. venv/bin/activate
python -m pip install -r requirements.txt
```

The tool uses the OpenAI interface for LLM interfacing. I use Argo via the [Argo Bridge](https://github.com/AdvancedPhotonSource/argo_bridge).

# Extract Survey Data

The `extract_survey_data.py` script is intended to extract the user survey data from an excel sheet and put it in a CSV output file which includes these columns: `question_number, question_text, user_response_text`.

This script needs to be updated if the spreedsheet format changes from year-to-year.

Run instructions:
```shell
usage: extract_survey_data.py [-h] -e EXCEL_FILE [-o OUTPUT_FILE] -t TEXT_QUESTIONS
                              [--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}]

Extract survey data from Excel file to CSV

options:
  -h, --help            show this help message and exit
  -e EXCEL_FILE, --excel-file EXCEL_FILE
                        Path to the Excel file containing survey data
  -o OUTPUT_FILE, --output-file OUTPUT_FILE
                        Path to the output CSV file
  -t TEXT_QUESTIONS, --text-questions TEXT_QUESTIONS
                        Path to JSON file containing list of text questions to process
  --log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Set the logging level
```

In this case the `TEXT_QUESTIONS` refers to the `inputs/text_questions.json` input file.

# Analyze Results

The `analyze_feedback.py` takes the formatted user feedback CSV file as input, along with two json files (described in next sections), then uses an LLM to analyze the open ended responses to our survey questions.

```shell
usage: analyze_feedback.py [-h] [--input INPUT] [--output OUTPUT] [--llm-config LLM_CONFIG]
                           [--text-questions TEXT_QUESTIONS]
                           [--log-level {DEBUG,INFO,WARNING,ERROR}]

Analyze open-ended user-feedback from CSV file, producing another CSV file with the results.

options:
  -h, --help            show this help message and exit
  --input INPUT         Input CSV file with survey responses
  --output OUTPUT       Output CSV file for analysis results
  --llm-config LLM_CONFIG
                        Path to LLM questions that can be asked for each Q&A pair, in json
                        format
  --text-questions TEXT_QUESTIONS
                        Path to text questions that were asked of the users and which LLM
                        questions should be asked for each response, in json format
  --log-level {DEBUG,INFO,WARNING,ERROR}
                        Set the logging level
```

# Input JSON files: `text_questions.json` and `llm_questions.json`

The `text_question.json` file should contain a list of dictionaries, with each entry corresponding to a question that was asked of the user that would yield an open-eded response, e.g.:
```json
[
   {
      "question_number": "Q3.1",
      "question_text": "Please comment on ALCF's role in the progress of your science or engineering project.",
      "llm_questions": [ "sentiment", "topic", "machine", "feedback", "emotion" ]
   },
   ...
]
```

Description:
* `question_number`: should be the unique question number used to identify the question in the survey.
* `question_text`: should be the question posed to the user.
* `llm_questions`: A list of the questions to be asked for the LLM to answer. Every question will be ask for a Question-Response pair. These are keyed to the questions included in `llm_questions.json`.

The `llm_questions.json` file should include a dictionary of questions that will be asked of the LLM for each Question-Response pair.

For example:
```json
{
   "sentiment": {
      "text": "Given the survey questions and the user's response, classify the response with the given categories.",
      "categories": ["positive", "neutral", "negative"],
      "answer_format": "Please only respond with the category that best describes the user's response and no other text."
   },
   "topic": {
      "text": "Given the survey questions and the user's response, which topic(s) do you think the feedback is most relavent?",
      "categories": ["operations", "software", "environment", "documentation", "support", "training", "other", "none"],
      "answer_format": "Please only respond with the topic or topics from the allowed categories that you think the feedback is most relavent to. Use a comma separated list for multiple topics."
   },
   "machine": {
      "text": "Given the survey questions and the user's response, to which supercomputer is the user referring?",
      "categories": ["polaris", "aurora", "theta", "frontier", "sophia", "other", "none"],
      "answer_format": "Please only respond with the supercomputer from the allowed categories that you think the user is referring to. Use a comma separated list for multiple supercomputers."
   },
   ...
}
```

