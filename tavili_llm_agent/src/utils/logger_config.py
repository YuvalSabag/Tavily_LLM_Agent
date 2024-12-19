import logging

def configure_logging(level=logging.INFO):
    """
    Configures logging for the application.

    Args:
        level (int): The logging level (e.g., logging.INFO, logging.WARNING).
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
