import logging
import re
from src.utils.logger_config import configure_logging
from src.utils.config import get_api_keys
from src.integration_nodes import TavilyAPI, OpenAINode, logger
from src.langgraph_workflow import fetch_search_results, process_search_results, generate_response, logger


# Set logging level to WARNING for minimal logs
configure_logging(level=logging.WARNING)
def process_query(query, tavily_api, openai_node):
    logging.info(f"Processing query: {query}")

    if not isinstance(query, str) or not query.strip():
        logging.warning("Query is empty. Skipping...")
        # return {"error": "Query is empty. Please provide a valid query."}
        return {
            "error": (
                "The query cannot be empty. Please type a question or phrase to search for. "
                "Example: 'Explain the role of AI in healthcare.'"
            )
        }

    try:
        logging.info("Fetching search results from Tavily API...")
        search_results = fetch_search_results(tavily_api, query)
    except Exception as e:
        logging.error(f"Unexpected error while fetching search results: {e}")
        # return {"error": "An unexpected error occurred while fetching search results."}
        return {
            "error": (
                "An unexpected error occurred while fetching results from Tavily. "
                "Please check your internet connection or try again later."
            )
        }

    if not search_results:
        logging.error("Failed to fetch search results.")
        return {"error": "No results were found for your query. Ensure your query is relevant and try again."}

    logging.info("Processing search results...")
    context = process_search_results(search_results)
    if not context:
        logging.warning("No valid content found for context generation.")
        return {
            "search_results": search_results,
            "error": (
                "The search results did not contain enough information to generate a response. "
                "Try rephrasing your query or being more specific."
            )
        }

    logging.info("Generating response with OpenAI...")
    gpt_response = generate_response(openai_node, context, query)
    if not gpt_response:
        logging.error("Failed to generate response from OpenAI.")
        return {
            "search_results": search_results,
            "error": (
                "The system was unable to generate a response from OpenAI. "
                "Please verify your API key or try again later. You might also refine your query for better results."
            )
        }

    logging.info("Workflow completed successfully.")
    return {"search_results": search_results, "gpt_response": gpt_response}


def display_results(query, results):
    print(f"\n✨ === Workflow Results for Query: '{query}' === ✨\n")

    if "error" in results:
        print(f"❌ Error: {results['error']}")
        return

    # Display search results
    print("🔍 --- Search Results ---\n")
    for idx, result in enumerate(results.get("search_results", {}).get("results", [])[:3], 1):
        title = result.get("title", "No Title")
        url = result.get("url", "No URL")
        print(f"{idx}. {title}\n   URL: {url}")

    # Display generated response
    gpt_response = results.get("gpt_response", "No response generated.")
    print("\n🧠 --- Generated Response ---\n")

    try:
        formatted_response = ""
        lines = gpt_response.strip().split("\n")
        for line in lines:
            if re.match(r"^\d+\.\s*", line):
                line = re.sub(r"^(\d+)\.", r"[\1]", line.strip())
                formatted_response += f"{line}\n"
            else:
                sentences = re.split(r'(?<=[.!?])\s+', line.strip())
                formatted_response += "\n".join(sentence.strip() for sentence in sentences if sentence.strip()) + "\n"
        print(formatted_response.strip())
    except Exception as e:
        print(f"An error occurred while formatting the response: {e}")
        print(gpt_response)


def predefined_demo(tavily_api, openai_node):
    predefined_queries = [
        "Who is serving as the President of the United States in 2024?",
        "Latest AI advancements",
        "Explain the role of AI in healthcare",
        "How is GPT-4 used in business?",
        "What is LangChain?",
        "The future of AI by 2030",
        None,  # Skip None queries
        ''  # Empty query to test edge case
    ]

    print("\n--- Running Predefined Demo Queries ---")
    for query in predefined_queries:
        if query is None:  # Skip None queries explicitly
            print("\n❌ Skipping invalid query: None")
            continue
        results = process_query(query, tavily_api, openai_node)
        display_results(query, results)


def demo():
    try:
        tavily_key, openai_key = get_api_keys()
        logging.info("API keys loaded successfully.")
    except ValueError as e:
        logging.error(f"Configuration error: {e}")
        raise SystemExit("Missing API keys. Please check your .env file.")

    tavily_api = TavilyAPI()
    openai_node = OpenAINode()

    print("\n🌟 Welcome to the LangGraph Workflow Demo! 🌟")
    print("Type 'demo' to see predefined queries or enter your question below. Type 'exit' to quit.")

    while True:
        query = input("\nEnter your query (or 'demo' for predefined queries, 'exit' to quit): ").strip()
        if query.lower() == "exit":
            print("\n🙏 Thank you for exploring the LangGraph Workflow Demo! \nSee you next time!")
            break
        elif query.lower() == "demo":
            predefined_demo(tavily_api, openai_node)
        else:
            logging.info(f"Processing query: {query}")
            results = process_query(query, tavily_api, openai_node)
            display_results(query, results)


if __name__ == "__main__":
    demo()
