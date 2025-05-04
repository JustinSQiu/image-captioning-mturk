from openai import OpenAI

from keys import openai_api_key

client = OpenAI(api_key=openai_api_key)


def get_gpt_response(messages, **kwargs):
    """Fetch GPT response for a given prompt."""
    response = client.chat.completions.create(
        messages=messages,
        **kwargs,
    )
    return response.choices[0].message.content.strip()
