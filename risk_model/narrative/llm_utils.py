# llm_utils.py

from openai import OpenAI
from config import LLM_CONFIG

def send_prompt_to_llm(prompt: str) -> str:
    """
    Sends a prompt to the LLM (OpenAI) and returns the response.

    Parameters:
    - prompt (str): The prompt/question to send to the model.

    Returns:
    - response_text (str): The text response from the model.
    """
    api_key = LLM_CONFIG.get("api_key")
    model_name = LLM_CONFIG.get("model_name", "gpt-4o")
    temperature = LLM_CONFIG.get("temperature", 0.2)

    if not api_key:
        raise ValueError("❌ OPENAI_API_KEY not found. Please set it in your environment variables.")

    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Please provide concise explanations."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature
    )

    response_text = response.choices[0].message.content.strip()
    return response_text
