'''A hacky logging wrapper to log both to the console and a file
'''
import logging

_format_string = '%(asctime)s %(module)s(%(lineno)d):%(levelname)s %(message)s'
logger = logging.NullHandler()

logging.basicConfig(format=_format_string, 
        level=logging.DEBUG,
        datefmt='%Y%m%d %H:%M:%S')
logger = logging.getLogger(__name__)
filelogger = logging.FileHandler(filename='./log/catflap.log')
filelogger.setFormatter(logging.Formatter(_format_string))
filelogger.setLevel(logging.DEBUG)

logger.addHandler(filelogger)


logger.debug("Logging started")