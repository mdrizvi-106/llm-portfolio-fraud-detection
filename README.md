# LLM Financial Anomaly Detection

## Problem
Financial fraud is costly and difficult to detect manually at scale. 
This project explores how Large Language Models can be used to flag 
anomalies in credit card transaction data — simulating a real compliance workflow.
Note: LLM analysis was run on a 5,000 transaction sample due to API cost constraints. Production implementation would require batch processing or a self-hosted model.

## Data
- Dataset: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (Kaggle)
- 284,807 transactions, 0.17% fraudulent
- Sample of 5,000 transactions used for LLM analysis

## Method
- Loaded and sampled transaction data using Pandas
- Built a plain-English summary of key statistics (transaction amounts, fraud count, top values)
- Sent the summary to **LLaMA 3.1 8B** via the **Groq API**
- Prompted the model as a financial compliance analyst to identify anomalies and explain reasoning

## Results
The LLM successfully identified anomalies based on:
- High-value transactions significantly above the average (£88.40)
- Unusual transaction patterns with no clear merchant context
- High-frequency patterns within short timeframes

## Business Relevance
In a real compliance workflow, this approach could be used to triage 
flagged transactions before human review — reducing analyst workload 
while maintaining explainability. LLM-generated reasoning provides 
audit-ready justification for each flagged case, unlike black-box models.

## Tools & Skills
Python · Pandas · Groq API · LLaMA 3.1 · LLM Integration · Financial Analysis
