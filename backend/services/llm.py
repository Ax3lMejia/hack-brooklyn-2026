from openai import AsyncOpenAI
from config import settings

FEATHERLESS_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"

_client: AsyncOpenAI | None = None


def get_llm_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=settings.featherless_api_key,
            base_url="https://api.featherless.ai/v1",
            timeout=30.0,
        )
    return _client


async def chat_complete(prompt: str) -> str:
    """Send a single user message and return the assistant text."""
    client = get_llm_client()
    response = await client.chat.completions.create(
        model=FEATHERLESS_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1024,
    )
    return response.choices[0].message.content or ""
