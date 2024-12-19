import unittest
from unittest.mock import patch
from src.utils.config import get_api_keys


class TestConfig(unittest.TestCase):
    """
    Unit tests for the config module.
    """

    @patch("os.getenv")
    def test_get_api_keys_success(self, mock_getenv):
        """
        Test successful retrieval of API keys.
        """
        # Mock environment variables
        mock_getenv.side_effect = lambda key: {
            "TAVILY_API_KEY": "mock_tavily_key",
            "OPENAI_API_KEY": "mock_openai_key"
        }.get(key, None)

        tavily_key, openai_key = get_api_keys()

        self.assertEqual(tavily_key, "mock_tavily_key", "Tavily API key should match mocked value")
        self.assertEqual(openai_key, "mock_openai_key", "OpenAI API key should match mocked value")

    @patch("os.getenv")
    def test_get_api_keys_missing_tavily_key(self, mock_getenv):
        """
        Test missing Tavily API key.
        """
        # Mock environment variables
        mock_getenv.side_effect = lambda key: {
            "TAVILY_API_KEY": None,
            "OPENAI_API_KEY": "mock_openai_key"
        }.get(key, None)

        with self.assertRaises(ValueError) as context:
            get_api_keys()

        self.assertIn("Tavily API Key is missing", str(context.exception), "Error message should indicate missing Tavily API key")

    @patch("os.getenv")
    def test_get_api_keys_missing_openai_key(self, mock_getenv):
        """
        Test missing OpenAI API key.
        """
        # Mock environment variables
        mock_getenv.side_effect = lambda key: {
            "TAVILY_API_KEY": "mock_tavily_key",
            "OPENAI_API_KEY": None
        }.get(key, None)

        with self.assertRaises(ValueError) as context:
            get_api_keys()

        self.assertIn("OpenAI API Key is missing", str(context.exception), "Error message should indicate missing OpenAI API key")

    @patch("os.getenv")
    def test_get_api_keys_both_keys_missing(self, mock_getenv):
        """
        Test both Tavily and OpenAI API keys missing.
        """
        # Mock environment variables
        mock_getenv.side_effect = lambda key: None

        with self.assertRaises(ValueError) as context:
            get_api_keys()

        self.assertIn("Tavily API Key is missing", str(context.exception), "Error message should indicate missing Tavily API key")
        # The first missing key should raise the error, so no need to assert for OpenAI here


if __name__ == "__main__":
    unittest.main()

