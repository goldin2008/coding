from azure.ai.openai import OpenAIClient
from azure.identity import DefaultAzureCredential


from azure.identity import CertificateCredential
from openai import AzureOpenAI
from config import AZURE_CONFIG, LLM_CONFIG, PROMPT_CONFIG

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

    def get_response(self, prompt: str) -> str:
        print(f"✉️ Sending prompt: '{prompt}'")
        response = self.client.chat.completions.create(
            model=LLM_CONFIG["model_name"],
            messages=[
                {"role": "system", "content": PROMPT_CONFIG["content"]},
                {"role": "user", "content": prompt}
            ],
            temperature=LLM_CONFIG.get("temperature", 0.2),
            max_tokens=LLM_CONFIG.get("max_tokens", 500)
        )
        response_text = response.choices[0].message.content.strip()
        print("✅ Response received successfully.")
        return response_text


# def get_azure_openai_client(endpoint):
#     return OpenAIClient(
#         endpoint=endpoint,
#         credential=DefaultAzureCredential()
#     )

# def generate_narrative(client, deployment_name, prompt):
#     response = client.chat.completions.create(
#         model=deployment_name,
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant for risk model explanations."},
#             {"role": "user", "content": prompt}
#         ],
#         temperature=0.2
#     )
#     return response.choices[0].message.content.strip()
