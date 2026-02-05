CACHE_DIR = f"YOUR CACHE DIR HERE"
DATA_DIR = "../data/interpretability_data"
INSTRUCTION="Please answer the question directly WITHOUT showing the reasoning process, you MUST write the answer as an integer without listing the equation."
LOGIT_ATTR_RESULT_DIR = "logit_attribution_results"
ACTIVATION_PATCHING_RESULT_DIR = "activation_patching_results"

PROBLEM_TYPES = [
        "addition", "subtraction", 
        "multiplication", "division"
]
OPERATOR_PAIRS = [
    ("addition", "subtraction", "addition-subtraction"),
    ("multiplication", "division", "multiplication-division"),
]
CROSS_PATCHING_CLEAN_CORR_MAP={
    "addition": [
        'paired_subtraction', 'subtraction',
        'multiplication', 'division'
        ],
    "subtraction": [
        'paired_addition', 'addition',
        'multiplication', 'division'
        ],
    "multiplication": [
        'paired_division', 'division',
        'addition', 'subtraction'
        ],
    "division": [
        'paired_multiplication', 'multiplication',
        'addition', 'subtraction'
        ],
}

SYSTEM_PROMPT = "You are a chatbot who is capable of performing the arithmetic problems."
MODEL_NAME_MAP = {
    "Qwen/Qwen2.5-7B-Instruct": "qwen-7b-instruct",
    "Qwen/Qwen2.5-14B-Instruct": "qwen-14b-instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct": "llama-8b-instruct", 
}
