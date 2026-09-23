import os
import logging
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

logs_path = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH, # Sends all logs to the file instead of terminal
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,  # Logs INFO, WARNING, ERROR, CRITICAL 
                         # Does NOT log DEBUG messages
)
#  Field	          Meaning
# %(asctime)s	  Time when log was created
# %(lineno)d	  Line number where log occurred
# %(name)s	      Logger name
# %(levelname)s	  Log level (INFO, ERROR, WARNING)
# %(message)s     Actual log message