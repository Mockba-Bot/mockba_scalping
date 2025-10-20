import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
import os

class DateRotatingFileHandler(RotatingFileHandler):
    def doRollover(self):
        if self.stream:
            self.stream.close()
            self.stream = None
        
        # Create a new log file name with the current date and time
        dfn = self.baseFilename + datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
        
        # Ensure the new log file name does not already exist
        if os.path.exists(dfn):
            os.remove(dfn)
        
        # Rotate the files
        self.rotate(self.baseFilename, dfn)
        
        # Open a new log file
        if not self.delay:
            self.stream = self._open()


# Configure trader logger spot
trader_logger = logging.getLogger('trader_logger')
if not trader_logger.hasHandlers():
    trader_logger.setLevel(logging.DEBUG)

    # Local file handler
    trader_handler = DateRotatingFileHandler(os.path.join(os.path.dirname(__file__), 'trader.log'), maxBytes=5*1024*1024, backupCount=0)
    trader_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    trader_handler.setFormatter(formatter)
    trader_logger.addHandler(trader_handler)