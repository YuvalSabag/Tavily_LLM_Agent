from dotenv import load_dotenv
import os

# Load API keys
load_dotenv()


def get_api_keys():
    """
    Load API keys for Tavily and OpenAI from environment variables.
    """
    tavily_key = os.getenv("TAVILY_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    # Raise an error if any API key is missing
    if not tavily_key:
        raise ValueError("Tavily API Key is missing. Check your .env file.")
    if not openai_key:
        raise ValueError("OpenAI API Key is missing. Check your .env file.")

    return tavily_key, openai_key

