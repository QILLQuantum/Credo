# credo_logger.py
import logging
import os

LOG_FILE = 'credo_mercy_trace.log'

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s — %(levelname)s — %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()  # Console for debug; remove for full silence
    ]
)

logger = logging.getLogger('credo')

# Usage example in any file:
# from credo_logger import logger
# try: ... except Exception as e: logger.error(f"Veiled error: {e}")