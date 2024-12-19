from src.utils.logger_config import configure_logging
from src.utils.config import get_api_keys
from src.integration_nodes import TavilyAPI, OpenAINode, logger
from src.langgraph_workflow import fetch_search_results, process_search_results, generate_response, logger
import logging
from time import time
import re


# Set logging level to INFO for detailed logs
configure_logging(level=logging.INFO)
def check_config():
    """
    Validates the configuration by checking API keys.
    """
    try:
        tavily_key, openai_key = get_api_keys()
        if not tavily_key or not openai_key:
            raise ValueError("Missing one or both API keys.")
        logger.info("API keys loaded successfully.")
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        raise SystemExit(f"Configuration error: {e}")


def run_workflow_for_query(query, tavily_api, openai_node):
    """
    Executes the workflow for a single query, including fetching, processing, and generating results.

    Args:
        query (str): User's search query.
        tavily_api (TavilyAPI): Instance of TavilyAPI.
        openai_node (OpenAINode): Instance of OpenAINode.

    Returns:
        dict: Workflow results or error message.
    """
    # Validate query
    if not isinstance(query, str) or not query.strip():
        logger.warning("Query is invalid or empty. Skipping this query.")
        return {"error": "Invalid or empty query. Please provide a meaningful input."}

    logger.info(f"Running workflow for query: '{query}'")

    if not query.strip():
        logger.warning("Query is empty. Skipping this query.")
        return {"error": "Query is empty."}

    # Fetch search results
    search_results = fetch_search_results(tavily_api, query)
    if "error" in search_results:
        logger.error(search_results["error"])  # Log the error from the function
        return search_results

    # Process search results
    context = process_search_results(search_results)
    if "error" in context:
        logger.error(context["error"])
        return context

    # Generate response with OpenAI
    response = generate_response(openai_node, context, f"Summarize information about: {query}")
    if "error" in response:
        logger.error(response["error"])
        return response

    logger.info("Workflow completed successfully.")
    return {
        "search_results": search_results,
        "gpt_response": response
    }

def run_test_ai_workflow():
    """
    Runs tests for various queries through the LangGraph workflow and Tavily/OpenAI integration.
    """
    test_queries = [
        "What is LangChain?",
        "  ",
        "AI trends in 2025",
        None,
        "Applications of GPT models in business",
        "What is artificial intelligence?",
        "Explain the role of AI in healthcare",
        "abcxyz123",  # Non-informative query
        ""  # Empty query to test edge case
    ]

    tavily_api = TavilyAPI()
    openai_node = OpenAINode()

    successful_tests = 0
    start_time = time()

    for query in test_queries:
        logger.info(f"\n--- Running Workflow for Query: '{query}' ---")
        workflow_start = time()
        results = run_workflow_for_query(query, tavily_api, openai_node)
        workflow_time = time() - workflow_start

        print("\n--- Workflow Results ---")
        # print(f"Query: {query or '[EMPTY QUERY]'}\n{'-' * 50}")
        print(f"Query: {query or '[EMPTY/INVALID QUERY]'}\n{'-' * 50}")

        if "error" in results:
            print(f"Error: {results['error']}")
        else:
            # Display search results
            print("\nSearch Results:")
            for idx, result in enumerate(results["search_results"].get("results", [])[:3], 1):
                title = result.get("title", "No Title")
                url = result.get("url", "No URL")
                print(f"{idx}. {title} - {url}")

            # Display generated response
            print("\nGenerated Response:")
            response = results["gpt_response"]
            formatted_response = ""
            lines = response.strip().split("\n")
            for line in lines:
                sentences = re.split(r'(?<=[.!?])\s+', line.strip())
                for sentence in sentences:
                    if sentence.strip():
                        formatted_response += f"{sentence.strip()}\n"
            print(formatted_response.strip())

            # Increment successful tests if no error and valid results
            if results["search_results"] and results["gpt_response"]:
                successful_tests += 1

        logger.info(f"Workflow for query '{query}' completed in {workflow_time:.2f} seconds.\n")

    total_time = time() - start_time

    print("\n--- Test Timing Summary ---")
    print(f"Total Queries: {len(test_queries)}")
    print(f"Successful Workflows: {successful_tests}")
    print(f"Failed Workflows: {len(test_queries) - successful_tests}")
    print(f"Total Execution Time: {total_time:.2f} seconds")
    print(f"Average Time Per Workflow: {total_time / len(test_queries):.2f} seconds")


if __name__ == "__main__":
    # Validate configuration
    check_config()

    # Run all tests
    logger.info("\n--- Starting LangGraph Workflow Tests ---")
    run_test_ai_workflow()
