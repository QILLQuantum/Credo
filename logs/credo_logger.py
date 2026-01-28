# credo_logger.py
import logging
import os

LOG_FILE = 'logs/credo_mercy_trace.log'
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(levelname)s — %(message)s',
    handlers=[logging.FileHandler(LOG_FILE)]
)

logger = logging.getLogger('credo_mercy')
