# from azure.ai.openai import OpenAIClient
from azure.identity import DefaultAzureCredential

from azure.identity import CertificateCredential
from openai import AzureOpenAI
from config.config import AZURE_CONFIG, LLM_CONFIG, PROMPT_CONFIG

import random

class AzureClient:
    def __init__(self):
        print("🔧 Initializing Azure Client...")

        # Initialize Certificate Credential
        creds = CertificateCredential(
            tenant_id=AZURE_CONFIG["tenant_id"],
            client_id=AZURE_CONFIG["client_id"],
            certificate_path=AZURE_CONFIG["certificate_path"],
        )

        # Get access token
        access_token = creds.get_token("https://cognitiveservices.azure.com/.default").token
        auth_header = f"Bearer {access_token}"
        print("✅ Access token obtained successfully.")

        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            api_key=AZURE_CONFIG["api_key"],
            azure_endpoint=AZURE_CONFIG["endpoint"],
            azure_deployment=AZURE_CONFIG["deployment"],
            api_version=AZURE_CONFIG["api_version"],
            default_headers={
                "Authorization": auth_header
            }
        )
        print("✅ Azure Client setup successful.")

    def get_response(
        self,
        prompt: str,
        model_name: str = None,
        system_content: str = None,
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        """
        Generate a response using Azure OpenAI.

        Parameters:
        - prompt (str): User prompt.
        - model_name (str, optional): LLM deployment name. Defaults to config.
        - system_content (str, optional): System instructions. Defaults to config.
        - temperature (float, optional): Response creativity. Defaults to config.
        - max_tokens (int, optional): Max tokens in response. Defaults to config.

        Returns:
        - str: The generated response text.
        """
        # Use provided values or fall back to config defaults
        model_name = model_name or LLM_CONFIG["model_name"]
        system_content = system_content or PROMPT_CONFIG["content"]
        temperature = temperature if temperature is not None else LLM_CONFIG.get("temperature", 0.2)
        max_tokens = max_tokens or LLM_CONFIG.get("max_tokens", 500)

        print(f"✉️ Sending prompt using model '{model_name}'")
        response = self.client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )

        response_text = response.choices[0].message.content.strip()
        print("✅ Response received successfully.")
        return response_text


class MockAzureClient:
    def __init__(self):
        print("🧪 Mock Azure Client initialized. No actual API calls will be made.")

    def get_response(
        self,
        prompt: str,
        model_name: str = None,
        system_content: str = None,
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        """
        Simulates a response for testing purposes. Returns a mocked narrative or evaluation summary.
        """
        # Example logic: vary output slightly based on model type
        if model_name and "judge" in model_name.lower():
            # Simulate judge model feedback with scores embedded
            scores = {
                "clarity": random.randint(1, 5),
                "conciseness": random.randint(1, 5),
                "completeness": random.randint(1, 5)
            }
            return (
                f"Clarity: {scores['clarity']}/5\n"
                f"Conciseness: {scores['conciseness']}/5\n"
                f"Completeness: {scores['completeness']}/5\n"
                "This explanation is generally clear and well-structured, with room for improvement."
            )
        else:
            # Simulate narrative generation
            return (
                "This entity has a high risk score of 85% primarily due to three factors: "
                "a high number of inbound wires, significant exposure to high-risk countries, "
                "and a central position in transaction networks. Each of these increases the risk profile significantly."
            )


def get_azure_client(use_mock=False):
    if use_mock:
        return MockAzureClient()
    else:
        return AzureClient()
