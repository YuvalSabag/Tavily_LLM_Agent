import time
import requests
import openai
import re
import logging

from src.utils.config import get_api_keys

# Setup logging
logger = logging.getLogger(__name__)

# Tavily API Node
class TavilyAPI:
    """
    Tavily API Node: Handles search queries to retrieve relevant data.
    """

    def __init__(self):
        self.api_key, _ = get_api_keys()
        self.base_url = "https://api.tavily.com/search"

    def search(self, query):
        """
        Send a query to Tavily and retrieve results.

        Args:
        - query: str, the search query

        Returns:
        - dict, the JSON response from Tavily API or None if an error occurred
        """
        if not query.strip():
            logger.error("Query is empty. Please provide a valid search query.")
            return None

        payload = {"query": query, "api_key": self.api_key}  # API request payload
        try:
            logger.info(f"Sending request to Tavily for query: '{query}'...")
            response = requests.post(self.base_url, json=payload)
            response.raise_for_status()
            logger.info("Tavily API response received successfully.")
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching results from Tavily API: {e}")
            return None


# OpenAI LLM Node
class OpenAINode:
    """
    OpenAI LLM Node: Handles interaction with the OpenAI GPT API.
    """

    def __init__(self, model="gpt-4"):
        _, self.api_key = get_api_keys()
        self.model = model

    def generate_response(self, context, query):
        """
        Generate a response based on the search context and query.

        Args:
        - context: str, the retrieved search results
        - query: str, the user's search query

        Returns:
        -   str: The generated response from the GPT-4 model or None if an error occurred.
        """
        if not context.strip() or not query.strip():
            logger.error("Context or query is empty. Cannot generate a response.")
            return None

        openai.api_key = self.api_key  # Set the OpenAI API key
        prompt = f"Using the following search results:\n{context}\n\nAnswer the query: {query}"

        try:
            logger.info("Sending request to OpenAI GPT-4...")
            response = openai.chat.completions.create(
                model=self.model,
                # model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )

            # Access the correct part of the response
            generated_response = response.choices[0].message.content
            # logger.info("OpenAI response generated successfully.")
            return generated_response

        except openai.RateLimitError as e:
            logger.warning("Rate limit exceeded. Retrying after delay...")
            time.sleep(5)
            return self.generate_response(context, query)
        except openai.AuthenticationError:
            logger.error("Error: Invalid OpenAI API key.")
        except Exception as e:
            logger.error(f"Error communicating with OpenAI: {e}")

        return None


def clean_content(text):
    """
    Cleans up unnecessary content like headers, footers, or repetitive words.
    """
    try:
        # Remove repetitive or unnecessary lines
        text = re.sub(r"\b(Share|Popular|Deep Dive|Advertise|About|Help|Stay connected|Subscribe)\b.*\n?", "", text)
        text = re.sub(r"©.*\n?", "", text)  # Remove copyright lines
        text = re.sub(r"[\n\r]+", "\n", text)  # Normalize newlines
        return text.strip()

    except Exception as e:
        logger.error(f"Error cleaning content: {e}")
        return text

