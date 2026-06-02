import logging
import os
from datetime import datetime

# Creating log file name with current timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Creating logs folder path
logs_path = os.path.join(os.getcwd(), "logs")

# Create logs directory if it does not exist
os.makedirs(logs_path, exist_ok=True)

# Full path of log file
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Test log
logging.info("Logging has started")