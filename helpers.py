from openai import OpenAI

from keys import openai_api_key

client = OpenAI(api_key=openai_api_key)


def get_gpt_response(messages, model='gpt-4o-mini', **kwargs):
    """Fetch GPT response for a given prompt."""
    response = client.chat.completions.create(
        messages=messages,
        model=model,
        **kwargs,
    )
    return response.choices[0].message.content.strip()
