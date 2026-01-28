import os 
 
LOG_FILE = 'logs/credo_mercy_trace.log' 
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True) 
 
logging.basicConfig( 
    level=logging.INFO,  # INFO for prod silence; DEBUG for tracing 
    format='%(asctime)s - %(levelname)s - %(message)s', 
    handlers=[ 
        logging.FileHandler(LOG_FILE), 
        # logging.StreamHandler()  # Uncomment for console trace 
    ] 
) 
 
logger = logging.getLogger('credo_mercy') 
