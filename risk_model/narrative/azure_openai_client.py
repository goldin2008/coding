from azure.ai.openai import OpenAIClient
from azure.identity import DefaultAzureCredential

def get_azure_openai_client(endpoint):
    return OpenAIClient(
        endpoint=endpoint,
        credential=DefaultAzureCredential()
    )

def generate_narrative(client, deployment_name, prompt):
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant for risk model explanations."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()
