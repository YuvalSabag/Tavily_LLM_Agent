import unittest
from unittest.mock import patch, MagicMock
from src.integration_nodes import TavilyAPI, OpenAINode
from src.langgraph_workflow import (
    fetch_search_results,
    process_search_results,
    generate_response,
    ai_workflow
)


class TestLangGraphWorkflow(unittest.TestCase):
    @patch("src.integration_nodes.TavilyAPI.search")
    def test_fetch_search_results(self, mock_search):
        """
        Test fetching search results from Tavily API.
        """
        # Mock Tavily API response
        mock_search.return_value = {
            "results": [
                {
                    "title": "AI Trends",
                    "content": "AI is transforming industries.",
                    "url": "http://example.com/ai-trends"
                },
                {
                    "title": "AI in 2024",
                    "content": "Predictions for AI in 2024.",
                    "url": "http://example.com/ai-2024"
                }
            ]
        }

        tavily_api = TavilyAPI()
        results = fetch_search_results(tavily_api, "AI advancements")

        self.assertIsNotNone(results, "Results should not be None")
        self.assertEqual(len(results["results"]), 2, "There should be two results")
        self.assertIn("content", results["results"][0], "Result should contain 'content'")
        self.assertIn("title", results["results"][0], "Result should contain 'title'")
        self.assertIn("url", results["results"][0], "Result should contain 'url'")

    def test_process_search_results(self):
        """
        Test processing search results and token management.
        """
        # Mock search results
        search_results = {
            "results": [
                {"content": "AI is transforming industries."},
                {"content": "Predictions for AI in 2024."},
                {"content": "Generative AI is the next big thing."}
            ]
        }

        context = process_search_results(search_results, max_results=2)

        self.assertIsNotNone(context, "Context should not be None")
        self.assertIn("AI is transforming industries.", context, "First result should be included")
        self.assertIn("Predictions for AI in 2024.", context, "Second result should be included")
        self.assertNotIn("Generative AI", context, "Excess results should not be included")

    @patch("src.integration_nodes.OpenAINode.generate_response")
    def test_generate_response(self, mock_openai_response):
        """
        Test generating a response using OpenAI with MagicMock for multiple calls.
        """
        # Mock OpenAI response with side effects for multiple calls
        mock_openai_response.side_effect = [
            "Response to query 1",
            "Response to query 2"
        ]

        openai_node = OpenAINode()

        # First call
        response1 = generate_response(openai_node, "Context for query 1", "Query 1")
        self.assertEqual(response1, "Response to query 1", "First response should match the mocked value")

        # Second call
        response2 = generate_response(openai_node, "Context for query 2", "Query 2")
        self.assertEqual(response2, "Response to query 2", "Second response should match the mocked value")

    @patch("src.integration_nodes.TavilyAPI.search")
    @patch("src.integration_nodes.OpenAINode.generate_response")
    def test_ai_workflow(self, mock_openai_response, mock_tavily_search):
        """
        Test the end-to-end AI workflow.
        """
        # Mock Tavily API response
        mock_tavily_search.return_value = {
            "results": [
                {"content": "AI is transforming industries."},
                {"content": "Predictions for AI in 2024."}
            ]
        }

        # Mock OpenAI response
        mock_openai_response.return_value = "The latest AI advancements include multimodal models and predictive analysis."

        user_query = "AI advancements"
        results = ai_workflow(user_query)

        # Assertions for the workflow output
        self.assertIn("search_results", results, "Workflow should include search results")
        self.assertIn("gpt_response", results, "Workflow should include GPT response")
        self.assertEqual(
            results["gpt_response"],
            "The latest AI advancements include multimodal models and predictive analysis.",
            "Response should match the mocked value"
        )
        self.assertEqual(
            len(results["search_results"]["results"]),
            2,
            "Search results should include two items"
        )


if __name__ == "__main__":
    unittest.main()
