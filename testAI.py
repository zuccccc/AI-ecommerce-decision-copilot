from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

response = client.chat.completions.create(
    model="openrouter/free",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful data analytics assistant."
        },
        {
            "role": "user",
            "content": "Explain what customer segmentation means in simple terms."
        }
    ]
)

print(response.choices[0].message.content)