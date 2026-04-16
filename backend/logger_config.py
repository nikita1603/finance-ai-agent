import logging

def setup_logger():
    """Configure the root logger for the application.

    This sets a sensible default `INFO` level and a compact log format that
    includes timestamp, level, logger name and the message. The function
    intentionally uses `logging.basicConfig` so callers can call it once at
    startup and then use `logging.getLogger(__name__)` elsewhere.
    """
    # Set up basic console logging with a consistent format across modules
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )