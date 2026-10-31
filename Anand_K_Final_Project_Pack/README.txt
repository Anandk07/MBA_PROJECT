# Final MBA Project Pack

Project: AI-Driven Customer Churn Prediction and Retention Analytics
Student: Anand K
Register Number: EC2552001011280
Guide: Dr Veena Christy

## Contents
- `Anand_K_Final_MBA_Project_Report_426_Responses.docx` — full report, with opening pages following the supplied SRM front-page template sequence.
- `Survey_Responses_426_Cleaned.csv` — supplied questionnaire export with only a punctuation-only complaint-frequency label standardized for grouped analysis.
- `Survey_Frequency_Tables.csv` — generated response counts and percentages; blanks are excluded and percentages use answered responses for each question.
- `Survey_Rating_Summary.csv` — rating means and 4–5 rating counts/percentages; blanks are excluded per question.
- `analyse_survey_and_demo_model.py` — reproducible survey analysis source code and optional separate synthetic-model demonstration.
- `analysis_outputs/*_distribution.png` — generated satisfaction, reliability, value, continue, and switch rating charts.

## Run
Install dependencies:
`pip install pandas matplotlib scikit-learn`

Run survey analysis:
`python analyse_survey_and_demo_model.py --survey Survey_Responses_426_Cleaned.csv`

Optional survey-based switching-intention model demonstration (uses ratings 4–5 on question 15 as high stated intention, with questions 1–13 as model inputs; compares models against a majority-class baseline):
`python analyse_survey_and_demo_model.py --survey Survey_Responses_426_Cleaned.csv --survey-switch-demo`

This model predicts survey-reported switching intention, not verified actual churn. Its single 75/25 holdout evaluation is exploratory and should not be presented as validated real-customer churn performance.

Run its automated tests from this project folder:
`python -m unittest discover -s tests -v`

If the separately labelled synthetic churn CSV is available, the optional demonstration can be run:
`python analyse_survey_and_demo_model.py --survey Survey_Responses_426_Cleaned.csv --synthetic-model Synthetic_Churn_Model_Demo_300_Not_Real_Customers.csv`

If the synthetic model results CSV is open or locked, the script saves results to a timestamped alternate CSV in the output directory.

## Data interpretation note
The supplied survey contains 426 records and 17 questionnaire items. It has no verified churn outcome. The switching-intention question must not be treated as actual churn. The optional synthetic model demonstration is not validated real-world model performance.

## Before academic submission
- Confirm the institution's required report structure, formatting, certificate wording, and signatories with your guide.
- Insert the official institution-approved similarity/plagiarism report; do not invent a similarity percentage.
- Verify that the attached questionnaire exactly matches the live form, including consent text and answer options.
