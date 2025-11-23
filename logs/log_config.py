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



# Configure Binance trader logger
binance_trader_logger = logging.getLogger('binance_trader_logger')
if not binance_trader_logger.hasHandlers():
    binance_trader_logger.setLevel(logging.DEBUG)

    # Binance file handler
    binance_handler = DateRotatingFileHandler(
        os.path.join(os.path.dirname(__file__), 'binance_trader.log'), 
        maxBytes=5*1024*1024, 
        backupCount=5  # Changed from 0 to keep some backups
    )
    binance_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    binance_handler.setFormatter(formatter)
    binance_trader_logger.addHandler(binance_handler)  # ✅ CORRECT - adding to binance logger
