import re
from openai import OpenAI
import pandas as pd
from sympy import (
    sympify,
    symbols,
    Eq,
    solve,
    simplify,
)


def postprocess_zs_generation(generation):
    '''
        Extract the first number (int or float) from the generation when no #### is used.
    '''
    generation = str(generation)
    match = re.search(r"\d+(\.\d+)?", generation)
    if match:
        return match.group(0)
    else:
        print("No numeric answer found.")
        return generation
    
######### final answer-based evaluation helpers (original)
def normalize(generation):
    for remove_char in [',', '$', '%', 'g']:
        generation = generation.replace(remove_char, '')

        return float(generation)


def extract_last_number(generation): # in cot generation the answer is not just a number
    # Find all numbers in the text using a regular expression
    numbers = re.findall(r'\d+\.?\d*', generation)  # Match integers or floats
    
    if numbers:
        # Return the last number found in the text
        return numbers[-1]
    else:
        # Return None if no number is found
        return None
    
######### symbolic abstractioin evaluation helpers
def get_api_response(message: str,
                     client,
                    max_tokens: int = 512,
                    temperature: int = 0,
                    use_gpt_4o: bool = True,
                   ):
    messages=[{
            "role": "user",
            "content": message
        }]
    model = "gpt-4o" if use_gpt_4o else "gpt-4o-mini"
    response = client.chat.completions.create(model=model,
                                        messages=messages,
                                        max_tokens=max_tokens,
                                        temperature=temperature)
    return response.choices[0].message.content

######### numerical abstractioin evaluation helpers
def extract_lhs(expr: str) -> str:
    # Take only the part before the first '=' if multiple '=' exist
    return expr.split('=')[0].strip()

def clean_expression(expr: str) -> str:
    """
    Cleans a mathematical expression by replacing symbols like %, $, and :.
    """
    
    expr = expr.strip()
    
    expr = expr.replace("'", "")
    
    expr = expr.replace("[", "(")
    expr = expr.replace("]", ")")
    
    expr = expr.replace("|", "")
    
    expr = expr.replace(":", "")
    
    expr = expr.replace('\\times', '*').replace(r'\times', '*')
    expr = expr.replace('×', '*')
    
    expr = expr.replace('÷', '/')
    
    # Remove $ signs
    expr = expr.replace('$', '')

    # Replace percentages (50% → 50/100)
    expr = re.sub(r'(\d+(?:\.\d+)?)\s*%', r'(\1/100)', expr)

    # Convert time-like format (e.g., 11:30 → 11.30)
    expr = re.sub(r'(\d+):(\d+)', r'\1.\2', expr)

    # Remove commas (e.g., 1,600 → 1600)
    expr = expr.replace(',', '')
    
    expr = re.sub(r'(?<=[\d\)])\s*(?=\()', '*', expr)
    
    
    # Replace 'x' with '*' only when it's clearly used as multiplication
    expr = re.sub(r'(?<=[\d\)])\s*[xX]\s*(?=[\d\(])', '*', expr)

    return expr

def solve_for_x(expr: str, expected_value: str, verbose: bool = False):
    try:
        if '=' not in expr or 'x' not in expr:
            return pd.NA

        expr = clean_expression(expr)
        lhs_raw, rhs_raw = map(str.strip, expr.split('=', 1))
        lhs = sympify(lhs_raw)
        rhs = sympify(rhs_raw)

        x = symbols('x')
        equation = Eq(lhs, rhs)
        solution = solve(equation, x)

        expected = simplify(clean_expression(str(expected_value)))

        if verbose:
            print(f"Solving equation: {lhs} = {rhs} → x = {solution}, expected = {expected}")

        return bool(solution and simplify(solution[0]) == expected)
    except Exception as e:
        if verbose:
            print(f"Error solving for x in '<{expected_value}>': {e}")
        return pd.NA


def latex_to_python_math(latex_str: str) -> str:
    """
    Converts LaTeX math expressions to Python-compatible strings.
    Handles fractions, parentheses, and special LaTeX commands safely.
    
    Args:
        latex_str: LaTeX math expression (e.g., r"\frac{1}{2}")
    
    Returns:
        Python-compatible math expression (e.g., "(1/2)")
    
    Examples:
        >>> latex_to_python_math(r"\frac{3}{5}")
        '(3/5)'
        >>> latex_to_python_math(r"\left(50 + \frac{1}{2}\right)")
        '(50 + (1/2))'
    """
    # Remove LaTeX delimiters \( and \) safely
    expr = re.sub(r"\\([()])", "", latex_str)
    
    # Dictionary of LaTeX to Python replacements
    replacements = {
        r"\left(": "(",
        r"\right)": ")",
        r"\times": "*",
        r"\cdot": "*",
        r"\div": "/"
    }

    for latex, python in replacements.items():
        expr = expr.replace(latex, python)

    
    # Convert fractions - using raw strings to avoid escape issues
    expr = re.sub(r"\\frac{([^}]+)}{([^}]+)}", r"(\1/\2)", expr)
    
    # Clean up whitespace
    expr = re.sub(r"\s+", " ", expr).strip()
    
    # Final validation
    if not all(c in "0123456789+-*/(). " for c in expr):
        raise ValueError("Result contains invalid characters")
    
    return expr