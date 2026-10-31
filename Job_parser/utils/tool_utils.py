TOOLS = {
    "python", "tensorflow", "pytorch", "scikit-learn", "sql", "pandas", "numpy",
    "aws", "azure", "gcp", "docker", "kubernetes", "mlflow", "spark", "hadoop",
    "langchain", "openai", "llm", "semantic kernel", "airflow", "fastapi"
}

def extract_tools_from_text(text):
    text_lower = text.lower()
    return sorted({tool for tool in TOOLS if tool in text_lower})
