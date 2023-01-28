import logging
import logzero
from logzero import setup_logger

logger = setup_logger(name="Service Name",
                      level=logzero.DEBUG,
                      formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                      )
