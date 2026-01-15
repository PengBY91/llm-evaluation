"""
Utility functions for determining API type based on task requirements.
"""
from typing import List


# Task output types that require loglikelihood
LOGLIKELIHOOD_OUTPUT_TYPES = {"loglikelihood", "loglikelihood_rolling", "multiple_choice"}


def get_task_output_types(task_names: List[str]) -> dict:
    """
    Get output types for a list of tasks.
    
    NOTE: This is a simplified version. The full implementation is in tasks.py
    and requires TaskManager initialization.
    """
    from api.tasks import get_task_output_types as get_output_types
    return get_output_types(task_names)


def requires_loglikelihood(task_names: List[str]) -> bool:
    """
    Determine if any task in the list requires loglikelihood.
    
    Args:
        task_names: List of task names to check
        
    Returns:
        True if any task requires loglikelihood, False otherwise
    """
    from api.tasks import requires_loglikelihood as check_loglikelihood
    return check_loglikelihood(task_names)


def determine_api_type(backend_type: str, task_names: List[str]) -> str:
    """
    Automatically determine which lm_eval model type to use based on backend and task requirements.
    
    Args:
        backend_type: The backend type (e.g., 'openai-api', 'huggingface')
        task_names: List of task names to evaluate
        
    Returns:
        The lm_eval model type to use (e.g., 'openai-chat-completions', 'openai-completions', 'hf')
        
    Examples:
        >>> determine_api_type("openai-api", ["gsm8k"])
        'openai-chat-completions'  # generate-until task
        
        >>> determine_api_type("openai-api", ["mmlu"])
        'openai-completions'  # loglikelihood task
        
        >>> determine_api_type("huggingface", ["gsm8k"])
        'hf'  # HuggingFace backend
    """
    if backend_type == "openai-api":
        # Check if tasks require loglikelihood
        if requires_loglikelihood(task_names):
            return "openai-completions"
        else:
            return "openai-chat-completions"
    elif backend_type == "huggingface":
        return "hf"
    else:
        # For custom or unknown backends, return as-is
        return backend_type
