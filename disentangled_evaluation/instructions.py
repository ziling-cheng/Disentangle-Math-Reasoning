
SYSTEM_PROMPT = "You are a chatbot who is capable of performing the arithmetic problems."

# zero-shot
symbolic_abstraction_zs_instruction = "Please answer the question directly WITHOUT showing the reasoning process, \
you MUST write the expression with appropriate round brackets after '####' without including the units, and you DO NOT need to simplify the expression."
original_zs_instruction = "Please answer the question directly WITHOUT showing the reasoning process, you MUST write the answer as an integer after '####', without including the equation or units."
numerical_abstraction_zs_instruction = "Please answer the question directly without showing the reasoning process, \
you MUST write the expression or equation with appropriate round brackets after '####', without including the equation or units, and you DO NOT need to simplify the expression."
computation_zs_instruction = "Please answer the question directly WITHOUT showing the reasoning process, you MUST write the answer as an integer after '####'"

# CoT
symbolic_abstraction_cot_instruction = "Let's think step by step, at the end, you MUST write the expression with appropriate round brackets after '####' without including the units, but you DO NOT need to simplify the expression."
original_cot_instruction = "Let's think step by step, you MUST write the answer as an integer after '####' without including the units. Write the answer at the end."
numerical_abstraction_cot_instruction =  "Let's think step by step, at the end, you MUST write the expression with round brackets after '####' without including the units, but you DO NOT need to simplify the expression."
computation_cot_instruction = "Let's think step by step, you MUST write the answer as an integer after '####'. Write the answer at the end."


TASK_CONFIG = {
    'original': {
        "question_col": "question",
        "zs": original_zs_instruction,
        "cot": original_cot_instruction,
    },
    
    'symbolic_abstraction': {
        "question_col": "symbolic_question",
        "zs": symbolic_abstraction_zs_instruction,
        "cot": symbolic_abstraction_cot_instruction,
    },
    
    'numerical_abstraction': {
        "question_col": "question",
        "zs": numerical_abstraction_zs_instruction,
        "cot": numerical_abstraction_cot_instruction,
    },
    
    'arithmetic_computation': {
        "question_col": "arithmetic_question",
        "zs": computation_zs_instruction,
        "cot": computation_cot_instruction,
    }
}