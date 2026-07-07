import re
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import shap
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from typing import List, Dict, Set, Tuple
import os
import difflib
import javalang
from pycparser import c_parser, c_ast
import subprocess
import tempfile
from collections import Counter

tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
model = AutoModel.from_pretrained("microsoft/graphcodebert-base")

def normalize_code(code: str) -> str:
    code = re.sub(r"//.*$", "", code, flags=re.MULTILINE)
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    lines = [line.strip() for line in code.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)

def detect_language(code: str) -> str:
    if "class " in code:
        return "java"
    elif "int " in code or "void " in code or "#include" in code:
        return "c"
    return "unknown"

def get_embedding(code: str):
    inputs = tokenizer(code, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

def compute_gcb_similarity(code1: str, code2: str) -> float:
    emb1 = get_embedding(code1)
    emb2 = get_embedding(code2)
    return F.cosine_similarity(emb1, emb2).item()

def token_overlap_similarity(code1: str, code2: str) -> float:
    tokens1 = code1.split()
    tokens2 = code2.split()
    c1 = Counter(tokens1)
    c2 = Counter(tokens2)
    common_tokens_count = sum((c1 & c2).values())
    total_tokens = max(len(tokens1), len(tokens2))
    return common_tokens_count / total_tokens if total_tokens > 0 else 1.0

def _preprocess_c_code(code: str) -> str:
    fake_header = """
    typedef int FILE;
    int scanf(int arg1, int *arg2);
    int printf(int arg1, long arg2);
    double log(double x);
    double exp(double x);
    double round(double x);
    double M_PI = 3.141592653589793;
    double M_E = 2.718281828459045;
    int STRING_LITERAL = 0;
    """
    code = re.sub(r'#include\s*<\w+\.h>', '', code)
    full_code = fake_header + code
    with tempfile.NamedTemporaryFile(suffix=".c", delete=False, mode="w") as temp_file:
        temp_file.write(full_code)
        temp_file_path = temp_file.name
    try:
        result = subprocess.run(
            ["cpp", "-P", temp_file_path],
            capture_output=True,
            text=True,
            check=True
        )
        preprocessed = result.stdout
        preprocessed = re.sub(r'"[^"]*"', 'STRING_LITERAL', preprocessed)
        return preprocessed
    except subprocess.CalledProcessError:
        return full_code
    finally:
        os.remove(temp_file_path)

def _get_c_ast_nodes(node: c_ast.Node) -> List[str]:
    nodes = [type(node).__name__]
    for _, child in node.children():
        nodes.extend(_get_c_ast_nodes(child))
    return nodes

def _get_java_ast_nodes(tree) -> List[str]:
    nodes = []
    for path, node in tree:
        nodes.append(type(node).__name__)
    return nodes

def compute_ast_similarity_and_diff(code1: str, code2: str, language: str) -> Tuple[float, List[str], List[str]]:
    if language == "c":
        try:
            code1_pre = _preprocess_c_code(code1)
            code2_pre = _preprocess_c_code(code2)
            parser = c_parser.CParser()
            ast1 = parser.parse(code1_pre, filename='<none>')
            ast2 = parser.parse(code2_pre, filename='<none>')
            nodes1 = _get_c_ast_nodes(ast1)
            nodes2 = _get_c_ast_nodes(ast2)
        except Exception:
            ratio = difflib.SequenceMatcher(None, normalize_code(code1), normalize_code(code2)).ratio()
            return ratio, [], []
    elif language == "java":
        try:
            tree1 = javalang.parse.parse(code1)
            tree2 = javalang.parse.parse(code2)
            nodes1 = _get_java_ast_nodes(tree1)
            nodes2 = _get_java_ast_nodes(tree2)
        except Exception:
            ratio = difflib.SequenceMatcher(None, normalize_code(code1), normalize_code(code2)).ratio()
            return ratio, [], []
    else:
        return 0.0, [], []

    c1 = Counter(nodes1)
    c2 = Counter(nodes2)
    
    common_nodes_counter = c1 & c2
    common_nodes_count = sum(common_nodes_counter.values())
    total_nodes = max(len(nodes1), len(nodes2))
    ast_sim = common_nodes_count / total_nodes if total_nodes > 0 else 1.0

    dissimilar_student = list((c1 - common_nodes_counter).keys())
    dissimilar_correct = list((c2 - common_nodes_counter).keys())

    return ast_sim, dissimilar_student, dissimilar_correct

def train_surrogate_model() -> RandomForestRegressor:
    np.random.seed(42)
    X_sim = np.random.uniform(0.4, 1.0, (200, 3))
    X_stats = np.random.uniform(10, 150, (200, 3))
    X = np.hstack((X_sim, X_stats))
    
    y = (0.5 * X[:, 0]) + (0.35 * X[:, 1]) + (0.15 * X[:, 2])
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    return rf_model

def explain_diff(code1: str, code2: str) -> List[str]:
    lines1 = code1.splitlines(keepends=True)
    lines2 = code2.splitlines(keepends=True)
    diff = difflib.unified_diff(lines1, lines2, fromfile="Student", tofile="Correct", lineterm="")
    diff_lines = [line.rstrip() for line in diff if line.startswith('+') or line.startswith('-') or line.startswith('@@')]
    return diff_lines if diff_lines else ["No significant differences detected."]

def _generate_shap_explanation(shap_values: np.ndarray, feature_names: List[str]) -> str:
    explanation = ["Here’s why these codes are similar or different:"]
    for feature, value in zip(feature_names, shap_values[0]):
        if value > 0:
            explanation.append(f"- {feature}: Increases similarity by {value:.4f}")
        else:
            explanation.append(f"- {feature}: Decreases similarity by {abs(value):.4f}")
    return "\n".join(explanation)

def analyze_code(student_code: str, correct_code: str):
    lang1 = detect_language(student_code)
    lang2 = detect_language(correct_code)
    if lang1 != lang2 or lang1 == "unknown":
        print("Error: Codes are in different or unsupported languages.")
        return

    norm_student = normalize_code(student_code)
    norm_correct = normalize_code(correct_code)
    
    gcb_sim = compute_gcb_similarity(norm_student, norm_correct)
    ast_sim, diss_student, diss_correct = compute_ast_similarity_and_diff(student_code, correct_code, lang1)
    token_sim = token_overlap_similarity(norm_student, norm_correct)
    
    similarity = (0.5 * gcb_sim) + (0.35 * ast_sim) + (0.15 * token_sim)
    
    features = np.array([
        gcb_sim, ast_sim, token_sim,
        len(norm_student.split()), len(norm_correct.split()), len(norm_student.splitlines())
    ])
    
    rf_model = train_surrogate_model()
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(features.reshape(1, -1))
    
    feature_names = [
        "GraphCodeBERT similarity", "AST similarity", "Token overlap",
        "Tokens in student code", "Tokens in correct code", "Lines in student code"
    ]
    shap_explanation = _generate_shap_explanation(shap_values, feature_names)

    print(f"Combined Similarity with correct solution: {similarity:.2f}")
    print("\nExplanation of Differences:")
    for line in explain_diff(student_code, correct_code):
        print(line)
        
    print("\nSHAP Explanation:")
    print(shap_explanation)
    
    if (shap_values[0][1] < 0 or similarity < 0.90) and (diss_student or diss_correct):
        print("\nTargeted Structural Feedback (AST Deviations):")
        if diss_student:
            print(f"  - Structural structures found in your code but absent from solution: {', '.join(diss_student)}")
        if diss_correct:
            print(f"  - Expected structures missing from your implementation: {', '.join(diss_correct)}")

    print("\nFeedback Evaluation Categorization:")
    if similarity > 0.98:
        print("  Excellent work! Your logic and structure almost perfectly match the correct solution.")
    elif similarity > 0.90:
        print("  Great work! Your code is very close to the correct implementation, with minor differences in logic.")
    elif similarity > 0.65:
        print("  Your code works, but there are noticeable differences in structure from the expected solution.")
    else:
        print("  Your approach is quite different from the expected solution. Review the logic and structure.")

def read_code_from_file(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def main():
    print("Code Comparison Tool (Validated Alignment Version)")
    print("==================================================")
    language = input("Enter the programming language (java or c): ").lower()
    if language not in ["java", "c"]:
        print("Error: Please enter 'java' or 'c' only.")
        return
    student_file = input(f"Enter the path to the student's {language} code file: ")
    correct_file = input(f"Enter the path to the correct {language} solution file: ")
    try:
        student_code = read_code_from_file(student_file)
        correct_code = read_code_from_file(correct_file)
        print(f"\n{language.upper()} Code Comparison: {os.path.basename(student_file)} vs {os.path.basename(correct_file)}")
        analyze_code(student_code, correct_code)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
