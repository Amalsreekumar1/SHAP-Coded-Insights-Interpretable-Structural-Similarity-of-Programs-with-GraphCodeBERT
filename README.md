# SHAP-CodedInsights: Interpretable Structural Similarity of Programs with GraphCodeBERT

An interpretable code similarity analysis tool designed for educational feedback. It compares a student's code with a correct reference solution using GraphCodeBERT embeddings, AST structural similarity, token overlap, and unified diff. The system also uses SHAP explainability to show which features contributed to the similarity score.

## 🌟 Key Features

### 🔍 Multi-Dimensional Code Analysis
- **GraphCodeBERT Semantic Similarity**: Measures contextual code similarity using transformer embeddings that capture control and data flow
- **AST Structural Comparison**: Analyzes program structure through Abstract Syntax Trees for both C (pycparser) and Java (javalang)
- **Token Overlap Analysis**: Lightweight lexical similarity metric for quick baseline comparison
- **Unified Diff Visualization**: Highlights exact line-level differences between student and correct code

### 📊 Intelligent Scoring System
- **Weighted Similarity Calculation**:
  - GraphCodeBERT Similarity: 50%
  - AST Similarity: 35%
  - Token Overlap: 15%
- **Categorical Feedback**: Excellent, Great, Partial, or Significant similarity ratings
- **Numerical Scores**: Range from 0.00 to 1.00 for precise assessment

### 💡 Explainable AI Feedback
- **SHAP-Based Explainability**: Shows which features (GraphCodeBERT, AST, tokens, line count) influenced the similarity score
- **Feature Attribution**: Understand why code is similar or different through detailed breakdowns
- **Targeted Recommendations**: Actionable feedback for students to improve their code

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/SHAP-CodedInsights.git
cd SHAP-CodedInsights

# Install dependencies
pip install -r requirements.txt

# Run the tool
python main.py
```

### Usage Example

```bash
Code Comparison Tool
====================
Enter the programming language (java or c): java
Enter the path to the student's java code file (e.g., student.java): student_solution.java
Enter the path to the correct java solution file (e.g., correct.java): instructor_solution.java

JAVA Code Comparison: student_solution.java vs instructor_solution.java
Combined Similarity with correct solution: 0.87

Explanation of Differences:
--- Student
+++ Correct
@@ -1,11 +1,14 @@
-public class CircleAreaSimple {
+public class CircleAreaFormula {
+    public static double calculateArea(double r) {
+        return Math.PI * r * r;
+    }
-    double area = 3.14f * radius * radius;
-    System.out.printf("%.2f\n", area);
+    System.out.printf("%.2f\n", calculateArea(radius));

SHAP Explanation:
Here's why these codes are similar or different:
- GraphCodeBERT similarity: Increases similarity by 0.0192
- AST similarity: Decreases similarity by 0.0110
- Token overlap: Decreases similarity by 0.0252
- Tokens in student code: Increases similarity by 0.0020
- Tokens in correct code: Decreases similarity by 0.0230
- Lines in student code: Increases similarity by 0.0124

Feedback:
  Workscorrectly,butstructurediffersnoticeably.
```

## 📋 Technical Architecture

### Core Components
1. **Preprocessing Module**: Normalizes code by removing comments and whitespace
2. **Language Detection**: Automatically identifies Java or C code
3. **Semantic Analysis**: Uses Microsoft's GraphCodeBERT for contextual understanding
4. **Structural Analysis**: Compares Abstract Syntax Trees for structural similarity
5. **Lexical Analysis**: Computes token overlap similarity
6. **SHAP Explainer**: Provides feature importance for interpretable results
7. **Feedback Generator**: Creates student-friendly feedback based on analysis

### Similarity Scoring Breakdown
| Component                | Weight | Description                              |
|--------------------------|--------|------------------------------------------|
| GraphCodeBERT Similarity | 50%    | Semantic similarity using embeddings     |
| AST Similarity           | 35%    | Structural similarity through parse trees|
| Token Overlap            | 15%    | Lexical similarity of common tokens      |

### Feedback Categories
- **Excellent (> 0.98)**: Nearly identical code
- **Great (0.90 - 0.98)**: Minor structural or logical differences
- **Partial (0.65 - 0.90)**: Functionally related, structurally distinct
- **Significant (≤ 0.65)**: Major structural or logical divergence

## 🛠 Requirements

```
transformers==4.35.0
torch==2.9.1
shap==0.43.0
scikit-learn==1.3.2
javalang==0.13.0
pycparser==2.21
```

## 📁 Project Structure

```
SHAP-CodedInsights/
├── main.py              # Main application script
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── student_solution.java # Example student code
├── instructor_solution.java # Example correct solution
└── docs/               # Documentation and examples
```

## 🎯 Educational Benefits

### For Students
- **Transparent Assessment**: Understand exactly how similarity scores are calculated
- **Targeted Improvement**: Receive specific feedback on structural and semantic differences
- **Self-Directed Learning**: Learn from detailed explanations rather than just scores

### For Instructors
- **Automated Grading**: Reduce manual effort in code assessment
- **Consistent Evaluation**: Standardized similarity measurement across submissions
- **Insightful Analytics**: Understand common student mistakes and approaches

## 🔮 Future Enhancements

- **Extended Language Support**: Add Python, C++, and JavaScript support
- **Behavioral Analysis**: Compare program execution traces for functional equivalence
- **Adaptive Learning**: Refine feedback based on usage patterns and outcomes
- **Web Interface**: Create a GUI dashboard for easy access and visualization
- **Integration APIs**: Connect with learning management systems (LMS)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- **Microsoft GraphCodeBERT Team** for the pre-trained model
- **SHAP (SHapley Additive exPlanations)** framework for explainable AI
- **javalang** and **pycparser** communities for robust parsing libraries
- **Python difflib** for unified difference generation

