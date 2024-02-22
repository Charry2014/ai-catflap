import cv2 as cv
import time
import sys

from base_logger import logger
from states import CatFlapFSM, Event
from imgsrcfactory import ImageSourceFactory
from statetypes import GlobalData

import socketio

# Shut off noisy messages -
# connectionpool(292):DEBUG Resetting dropped connection: 10.0.0.38
# connectionpool(546):DEBUG http://10.0.0.38:5000 "POST /socket.io/?transport=polling&EIO=4&sid=4BN7BycU05USNayrAAGE HTTP/1.1" 200 None
import logging
logging.getLogger("urllib3").setLevel(logging.WARNING)


''' The main loop entry point
'''
def main_loop(args):
    img_src = ImageSourceFactory.create_source(args.stream)
    img_src.open()

    # Connect to the Flask server using socketio
    sio = None
    if hasattr(args, 'web') and args.web is not None:
        logger.info(f"Connecting to web server at {args.web}")
        sio = socketio.Client()
        try:
            sio.connect(args.web)
        except:
            logger.error(f"Connection to {args.web} failed")
            delattr(args, 'web')

    exit_code = 0
    try:
        logger.info("Started cat flap control")
        state_machine = CatFlapFSM(GlobalData(args, event=Event(img_src.get_image())))

        # Main loop - here we go
        while img_src.isopen == True:
            state_machine.event_handle(Event(img_src.get_image()))
    except Exception as e:
        logger.exception(f"Caught exception {e.__class__} - {e}")
        time.sleep(2)
        exit_code = 1
    finally:
        # Goodbye, world
        cv.destroyAllWindows()
        state_machine.exit()
        img_src.close()
        return exit_code



def main(): 
    import argparse
    parser = argparse.ArgumentParser(description="MotionDetect - Motion detection AI CatFlap")
    parser.add_argument('--record_path', 
        help="Specify output directory for recorded video",
        default='./recording', # See copy_recordings.sh
        required=False,
        type=str, action='store',)
    parser.add_argument(
        '--trigger', 
        help="Define the trigger area coordinates x,y,w,h",
        action='store', 
        required=True)
    parser.add_argument(
        '--record_overlays', 
        help="Record overlays in the video stream",
        action='store_true', 
        required=False)
    parser.add_argument(
        '--show_trigger', 
        help="Show trigger area for motion detection",
        action='store_true', 
        required=False)
    parser.add_argument(
        '--headless', 
        help="Run headless and show no video windows",
        action='store_true', 
        required=False)
    parser.add_argument(
        '--web', 
        help="Define the monitoring website address",
        action='store', 
        type=str, 
        required=False)
    parser.add_argument(
        '--label-json', 
        help="JSON file containing the detection labels",
        action='store', 
        required=False, 
        type=str, 
        default='./labels.json')
    parser.add_argument(
        '--trigger-json', 
        help="JSON file containing the evaluation configuration for the triggering state",
        action='store', 
        required=False, 
        type=str, 
        default='./trigger_config.json')
    parser.add_argument(
        '--eval-json', 
        help="JSON file containing the evaluation configuration for the motion locked state",
        action='store', 
        required=False, 
        type=str,   
        default='./eval_config.json')
    #  ML Parameters
    parser.add_argument(
        '--model',
        help='Path of the object detection model.',
        required=False,
        #   default='efficientdet_lite0.tflite')
        default='cats.tflite')
    parser.add_argument(
        '--stream', 
        help='Link to stream or ID of camera.', 
        required=False, 
        type=str, 
        default='0')
    parser.add_argument(
        '--frameWidth',
        help='Width of frame to capture from camera.',
        required=False,
        type=int,
        default=640)
    parser.add_argument(
        '--frameHeight',
        help='Height of frame to capture from camera.',
        required=False,
        type=int,
        default=480)
    parser.add_argument(
        '--num_threads',
        help='Number of CPU threads to run the model.',
        required=False,
        type=int,
        default=4)
    parser.add_argument(
        '--enable_edgetpu',
        help='Whether to run the model on EdgeTPU.',
        action='store_true',
        required=False,
        default=False)
    
    args = parser.parse_args()

    logger.info("Started cat flap control")
    sys.exit(main_loop(args))
    

if __name__ == "__main__":
    main()