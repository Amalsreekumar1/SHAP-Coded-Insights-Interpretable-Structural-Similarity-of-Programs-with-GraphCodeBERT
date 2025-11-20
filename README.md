# SHAP-Coded-Insights-Interpretable-Structural-Similarity-of-Programs-with-GraphCodeBERT
An interpretable code similarity analysis tool designed for educational feedback.
It compares a student’s code with a correct reference solution using GraphCodeBERT embeddings, AST structural similarity, token overlap, and unified diff.
The system also uses SHAP explainability to show which features contributed to the similarity score.
# Features
* GraphCodeBERT Semantic Similarity:
  
   Measures contextual code similarity using transformer embeddings.
* AST Structural Comparison:
  
  Supports both C (pycparser) and Java (javalang).
* Token Overlap Analysis:
  
  Lightweight lexical similarity metric.
* Unified Diff:
  
  Highlights exact line-level differences between student and correct code.
* SHAP-Based Explainability:
  
  Shows which features (GraphCodeBERT, AST, tokens, line count) influenced the similarity score.
* CLI Tool:
  
  Compare any two code files directly from terminal.

# Installation
* Install dependencies:

  pip install -r requirements.txt
* Run the CLI:

  python code_similarity.py

# Output Includes
* Combined similarity score
* Structural diff (line-level)
* SHAP explanation (feature contributions)
* Student-friendly textual feedback

# How It Works
The similarity score is computed using:
| Component                | Weight |
| ------------------------ | ------ |
| GraphCodeBERT Similarity | 0.50   |
| AST Similarity           | 0.35   |
| Token Overlap            | 0.15   |

SHAP explains the importance of each feature in the model’s output, making the system transparent and ideal for education.

# Acknowledgements
* Microsoft GraphCodeBERT
* SHAP Explainability Framework
* javalang & pycparser
* difflib unified diff
