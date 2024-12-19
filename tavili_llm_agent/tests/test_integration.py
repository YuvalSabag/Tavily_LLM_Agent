import unittest
from unittest.mock import patch, MagicMock
from src.integration_nodes import TavilyAPI, OpenAINode, clean_content


class TestIntegrationNodes(unittest.TestCase):
    @patch("src.integration_nodes.requests.post")
    def test_tavily_api_search(self, mock_post):
        # Mock response for Tavily API
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "results": [
                {"title": "AI Trends", "content": "AI is transforming industries.", "url": "http://example.com/ai-trends"},
                {"title": "AI Predictions", "content": "Predictions for AI in 2024.", "url": "http://example.com/ai-2024"}
            ]
        }

        tavily_api = TavilyAPI()
        query = "AI advancements"
        response = tavily_api.search(query)

        # Assertions
        self.assertIsNotNone(response, "Response should not be None")
        self.assertIn("results", response, "Response should include 'results'")
        self.assertEqual(len(response["results"]), 2, "There should be two results")
        self.assertEqual(response["results"][0]["title"], "AI Trends", "First result title should match")
        self.assertEqual(response["results"][1]["title"], "AI Predictions", "Second result title should match")

    @patch("src.integration_nodes.openai.chat.completions.create")
    def test_openai_node_generate_response(self, mock_openai_create):
        # Mock response for OpenAI API
        mock_openai_create.return_value = MagicMock(
            choices=[
                MagicMock(message=MagicMock(content="AI is advancing with GPT-4 and multimodal systems."))
            ]
        )

        openai_node = OpenAINode()
        context = "AI is transforming industries."
        query = "What are the latest AI advancements?"
        response = openai_node.generate_response(context, query)

        # Assertions
        self.assertIsNotNone(response, "Response should not be None")
        self.assertEqual(
            response, "AI is advancing with GPT-4 and multimodal systems.",
            "Response should match the mocked value"
        )

    def test_clean_content(self):
        # Input text with unnecessary content
        input_text = (
            "Share this article\nAI is transforming industries.\n© 2024 Company\nStay connected\nAI predictions"
        )

        # Expected cleaned content
        expected_output = "AI is transforming industries.\nAI predictions"

        # Clean content
        cleaned_text = clean_content(input_text)

        # Assertions
        self.assertEqual(cleaned_text, expected_output, "Cleaned text should match the expected output")


if __name__ == "__main__":
    unittest.main()
