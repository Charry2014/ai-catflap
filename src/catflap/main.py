from array import array
import cv2 as cv
from videorecorder import VideoRecorder

from tflite_detect import TFLiteDetect
import time

# Hacks
from datetime import datetime, timedelta
from os import path
import sys, os

from base_logger import logger
from detections import TFLDetections, BoundingRect
from states import CatFlapFSM
from imgsrcfactory import ImageSourceFactory


# Visualization parameters
row_size = 20  # pixels
left_margin = 24  # pixels
text_color = (255, 55, 0)
font_size = 1
font_thickness = 1
fps_avg_frame_count = 10

def create_overlays(frame1, detections):
    for d in detections:
        (name, score, bounds) = d
        cv.rectangle(frame1, (bounds.origin_x, bounds.origin_y), 
                    (bounds.origin_x+bounds.width, bounds.origin_y+bounds.height), (255, 55, 0), 2)
        text_location = (bounds.origin_x, bounds.origin_y)
        text = f"{name} {score:.2f}"
        cv.putText(frame1, text, text_location, cv.FONT_HERSHEY_PLAIN,
                     font_size, text_color, font_thickness)
    return frame1


def make_outfile_name(record_path, label, prob):
    datestr = datetime.now().strftime("%Y%m%d-%H%M%S.%f")[:-3]
    prob = int(prob * 100)
    outfile = path.join(record_path, f"{label}-{str(prob)}-{datestr}_catcam.jpg")
    return outfile


def motionDetection(img_src, state_machine:CatFlapFSM, args):
    frame1 = img_src.get_image()
    frame2 = img_src.get_image()

    tflite = TFLiteDetect(args.model, args.enable_edgetpu, args.num_threads)

    # Set the movement target area, mark it with a rectangle
    bc, br, bw, bh = [int(x) for x in args.trigger.split(',')]

    while True:
        movement = False

        cp1 = frame1[br:br+bh, bc:bc+bw]
        cp2 = frame2[br:br+bh, bc:bc+bw]
        diff = cv.absdiff(cp1, cp2)

        # Show the motion trigger area video as well as a separate video window
        if args.show_trigger == True:
            cv.imshow("Motion trigger area", cp1)
            cv.waitKey(30)

        # Motion detection magic
        diff_gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(diff_gray, (5, 5), 0)
        _, thresh = cv.threshold(blur, 20, 255, cv.THRESH_BINARY)
        dilated = cv.dilate(thresh, None, iterations=3)
        contours, _ = cv.findContours(
            dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # Display video preview window
        if args.headless == False:
            if args.record_overlays == True:
                cv.rectangle(frame1, (bc, br), (bc+bw, br+bh), (255, 55, 0), 2)
            cv.imshow("Video frame 1", frame1)
            cv.waitKey(30)
 
        # Check if enough movement is found
        for contour in contours:
            area = cv.contourArea(contour)
            if area > 1200:
                movement = True
                break

        if movement == False:
            # Nothing happening in front of the camera, rest a bit then go again
            # TODO - this should be linked to the state, so in idle the delay is
            # longer and in other states there is no delay. For now just reduce.
            # time.sleep(0.5)
            time.sleep(0.1)
            frame1 = frame2
            frame2 = img_src.get_image()
            continue

        logger.info(f"Movement detected")

        # Run tflite model and deal with the results
        tfl_detection = tflite.detect(frame1)
        detections = TFLDetections(args.stream)
        for d in tfl_detection.detections:
            c = d.categories[0]
            logger.info(f"Detection {c.category_name} index {c.index} score {c.score:.2f}")

            # Filter out "Cat-body" or "Background" and noisy low confidence results
            # temporarily let Cat-body through for testing.
            if c.index > 2 or c.score < 0.3:
                continue
            detections.add(c.category_name, c.index, c.score, 
                            BoundingRect(d.bounding_box.origin_x, d.bounding_box.origin_y, 
                                        d.bounding_box.width, d.bounding_box.height ))

        
        if len(detections) > 0:
            logger.info(f"Number of detections {len(detections)} in {detections}")
            if(args.record_overlays == True):
                frame1 = create_overlays(frame1, detections)
            outfile = make_outfile_name(args.record_path, detections[0].label, detections[0].score)
            cv.imwrite(outfile, frame1)
            # Call state machine to evaluate impact of new image
            # Always do this, for all events the camera sees
            for d in detections:
                state_machine.event_handle(d)
        else:
            logger.info("Movement - but no cat events")

        frame1 = frame2
        frame2 = img_src.get_image()
 
    cap.release()
    cv.destroyAllWindows()

def image_classification(args):
    '''Take an image and run the classification of it
    Displays the image and waits for key.
    '''
    img_src = ImageSourceFactory.create_source(args.stream)
    img_src.open_image_source()
    frame1 = img_src.get_image()

    tflite = TFLiteDetect(args.model, args.enable_edgetpu, args.num_threads)
    tfl_detection = tflite.detect(frame1)

    detections = TFLDetections(args.stream)
    for d in tfl_detection.detections :
        c = d.categories[0]
        detections.add(c.category_name, c.index, c.score, 
                                        BoundingRect(d.bounding_box.origin_x, d.bounding_box.origin_y, 
                                                    d.bounding_box.width, d.bounding_box.height ))

    logger.info(f"Number of detections {len(detections)} in {detections}")
    for d in detections:
        logger.debug(d)
    if len(detections) > 0:
        if(args.record_overlays == True):
            frame1 = create_overlays(frame1, detections)
    if args.headless == False:
        cv.imshow("Video frame 1", frame1)
        cv.imwrite(os.path.join(args.record_path, 'testimg.jpg'), frame1)
        cv.waitKey(0)
        cv.destroyAllWindows()



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

    # Run a simple image classification given a single image
    # Very hacky, but at this point I don't care
    if args.stream.endswith(".jpg"):
        print(f"Running image classification on {args.stream}")
        image_classification(args)
        return

    state_machine = CatFlapFSM()
    try:
        logger.info("Started motion detection")
        img_src = ImageSourceFactory.create_source(args.stream)
        img_src.open_image_source()
        motionDetection(img_src, state_machine, args)
    except Exception as e:
        logger.exception(f"Caught exception {e.__class__} - {e}")
        time.sleep(2)
    finally:
        # Goodbye, world
        cv.destroyAllWindows()
        state_machine.exit()
        sys.exit(1)
    

if __name__ == "__main__":
    main()