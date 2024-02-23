'''A hacky logging wrapper to log both to the console and a file
'''
import logging
import os
from datetime import datetime


_format_string = '%(asctime)s %(module)s(%(lineno)d):%(levelname)s %(message)s'
logger = logging.NullHandler()

logging.basicConfig(format=_format_string, 
        level=logging.DEBUG,
        datefmt='%Y%m%d %H:%M:%S')
logger = logging.getLogger(__name__)

directory_path = './log/'
file_name = 'catflap.log'

if not os.path.exists(directory_path):
        try:
                os.makedirs(directory_path)
                print(f"Directory '{directory_path}' created successfully.")
        except OSError as e:
                print(f"Error creating directory '{directory_path}': {e}")
elif os.path.exists(os.path.join(directory_path, file_name)):
       current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
       base, ext = os.path.splitext(file_name)
       new_filename = f"{base}_{current_datetime}{ext}"
       os.rename(os.path.join(directory_path, file_name), os.path.join(directory_path, new_filename))

filelogger = logging.FileHandler(filename=directory_path + file_name)
filelogger.setFormatter(logging.Formatter(_format_string))

filelogger.setLevel(logging.INFO)
logger.setLevel(logging.INFO)

logger.addHandler(filelogger)


logger.debug("Logging started")