import logging

# Logging - more informative than `print` with timestamps
logger = logging.getLogger(__name__.split(".")[0])
stream_handler = logging.StreamHandler()  # default handler
stream_handler.setFormatter(
    logging.Formatter(
        "[%(asctime)s %(levelname)-8s]: %(message)s",
        "%M:%S",
    )
)
logger.handlers = [stream_handler]
