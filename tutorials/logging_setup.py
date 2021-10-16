import copy
import logging

class MyFormatter(logging.Formatter):
    def format(self, record):
        record = copy.copy(record)
        record.name = '.'.join(record.name.split('.')[-2:])
        return super().format(record)

logger = logging.getLogger('hubbard')
if len(logger.handlers) <= 1:
    def _setupLogger(logger):
        for h in logger.handlers:
            if h.formatter._fmt == logging.BASIC_FORMAT:
                logger.removeHandler(h)

        if not logger.hasHandlers():
            import sys
            hdlr = logging.StreamHandler(sys.stdout)
            frmt = MyFormatter('%(name)-23s| %(levelname)8s - %(message)s')
            hdlr.setFormatter(frmt)
            logger.addHandler(hdlr)
            logger.propagate = False
    _setupLogger(logger)
    del _setupLogger
