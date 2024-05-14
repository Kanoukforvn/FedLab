import logging
import time

# Configure logging to both stdout and a log file
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(), logging.FileHandler('log')])

logging.info("test")
