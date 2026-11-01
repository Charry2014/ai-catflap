import cv2 as cv
import numpy as np
import queue
import time
import sys

from base_logger import logger
from states import CatFlapFSM, Event
from imgsrcfactory import ImageSourceFactory
from statetypes import GlobalData
from tflite_detect import TFLiteDetect
from motion import MotionDetector
from analysisthread import AnalysisThread
from videorecorder import ClipRecorder
from statetypes import EventKind
from framebus import FrameBus
from capturethread import CaptureThread
from backgroundworker import BackgroundWorker

import socketio
import base64

# Shut off noisy messages -
# connectionpool(292):DEBUG Resetting dropped connection: 10.0.0.38
# connectionpool(546):DEBUG http://10.0.0.38:5000 "POST /socket.io/?transport=polling&EIO=4&sid=4BN7BycU05USNayrAAGE HTTP/1.1" 200 None
import logging
logging.getLogger("urllib3").setLevel(logging.WARNING)


FPS = 15               # Frame rate of the pipeline: camera, pre-roll and video clips
WEB_MAX_FPS = 10        # The website only needs a preview, not every frame
STATS_INTERVAL = 30.0   # Seconds between pipeline statistics log lines


class WebPublisher():
    '''Sends preview images to the website from its own thread, so encoding and
    network delays never hold up capture or detection. Only the newest frame is
    kept: a slow website just sees fewer frames.'''
    def __init__(self, sio) -> None:
        self._sio = sio
        self._last_ts = None
        self._worker = BackgroundWorker(self._send, "web-publisher", maxsize=1, drop_oldest=True)

    def offer(self, event) -> None:
        if event.payload is None:
            return
        if self._last_ts is not None and event.ts - self._last_ts < 1.0 / WEB_MAX_FPS:
            return
        self._last_ts = event.ts
        # Time of day is read here, on the main thread, as the frame's own time is not kept in events
        self._worker.submit((event.payload, time.time()))

    def _send(self, item) -> None:
        # Overlay the time on a copy (the frame itself is read-only)
        frame, wall = item
        stamp = time.strftime('%H:%M:%S', time.localtime(wall))
        image = cv.putText(frame.copy(), stamp, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        _, buffer = cv.imencode('.jpg', image)
        jpeg_bytes = base64.b64encode(buffer.tobytes()).decode('utf-8')
        # Sending does crash sometimes, so we catch it
        try:
            self._sio.emit('image_data', {'image': jpeg_bytes})
        except Exception as e:
            logger.error(f"Error sending image to web: {e}")

    def close(self) -> None:
        self._worker.close()


def _safely(fn, *args):
    try:
        fn(*args)
    except Exception:
        logger.exception(f"Error during shutdown in {getattr(fn, '__name__', fn)}")


def create_detector(args, size):
    '''Load the model once, and run a dummy frame through it so the first real
    detection is not slow.'''
    detector = TFLiteDetect(args.model, args.enable_edgetpu, args.num_threads)
    blank = np.zeros((size[1], size[0], 3), np.uint8)
    for _ in detector.detect(blank):
        pass
    return detector


''' The main loop entry point.

    Threads:
      capture   - camera -> FrameBus (framebus.py)
      analysis  - newest frame -> movement boxes (idle) or ML detections (armed) -> events
      recorder  - FrameBus -> video clips with pre-roll (optional)
      web, image writer - background workers for slow output
      main      - this loop: events -> state machine, GPIO, windows. Nothing slow runs here.
'''
def main_loop(args):
    img_src = ImageSourceFactory.create_source(args.stream)
    if img_src is None:
        logger.error(f"No image source can handle {args.stream}")
        return 1
    if img_src.open() != 0:
        return 1

    # Connect to the Flask server using socketio
    web = None
    if getattr(args, 'web', None) is not None:
        logger.info(f"Connecting to web server at {args.web}")
        sio = socketio.Client()
        try:
            sio.connect(args.web)
        except Exception:
            logger.error(f"Connection to {args.web} failed")
        else:
            logger.info(f"Connected to web server")
            web = WebPublisher(sio)

    exit_code = 0
    # The ring buffer must hold the pre-roll plus some slack for the recorder
    keep = int((args.preroll + 2) * FPS) if args.record_video else FPS
    bus = FrameBus(maxlen=max(keep, FPS))
    capture = CaptureThread(img_src, bus)
    analysis = None
    state_machine = None
    data = None
    try:
        logger.info("Started cat flap control")
        capture.start()
        frame = bus.wait_newer(-1, timeout=10)
        if frame is None:
            logger.error("No images from the image source")
            return 1
        height, width = frame.image.shape[:2]
        events = queue.Queue(maxsize=32)
        data = GlobalData(args, event=Event(frame.image, frame.ts, frame.id))
        data.events = events
        if args.record_video:
            data.clips = ClipRecorder(bus, args.record_path, fps=FPS, preroll=args.preroll,
                                      postroll=args.postroll,
                                      annotations=data.annotations if args.record_overlays else None)
        detector = create_detector(args, (width, height))
        motion = MotionDetector(data.trigger_bc, data.trigger_br, data.trigger_bcw, data.trigger_brh)
        analysis = AnalysisThread(bus, events, detector, motion)
        data.analysis = analysis
        state_machine = CatFlapFSM(data)

        analysis.start()
        stats_at = time.monotonic() + STATS_INTERVAL
        while True:
            try:
                event = events.get(timeout=1.0)
            except queue.Empty:
                continue
            if event.kind == EventKind.END:
                break
            if web is not None and event.kind == EventKind.FRAME:
                web.offer(event)
            state_machine.event_handle(event)
            if time.monotonic() >= stats_at:
                latency = 1000 * (time.monotonic() - event.ts) if event.ts else 0
                logger.info(f"Pipeline: captured {bus.published}, analysed {analysis.processed}, "
                            f"skipped {analysis.skipped}, dropped {analysis.dropped}, latency {latency:.0f} ms")
                stats_at = time.monotonic() + STATS_INTERVAL
        if analysis.failed:
            exit_code = 1
    except Exception as e:
        logger.exception(f"Caught exception {e.__class__} - {e}")
        time.sleep(2)
        exit_code = 1
    finally:
        # Goodbye, world. Each step is isolated so one failure cannot skip the
        # others - above all the one that unlocks the cat flap.
        if state_machine is not None:
            _safely(state_machine.exit)
        capture.stop()
        if analysis is not None:
            analysis.stop()
        _safely(capture.join, 5)
        if not args.headless:
            _safely(cv.destroyAllWindows)
        if web is not None:
            _safely(web.close)
        if data is not None:
            _safely(data.close)
        _safely(img_src.close)
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
        '--record_video',
        help="Record a video clip around every event, including the seconds before it",
        action='store_true',
        required=False)
    parser.add_argument(
        '--preroll',
        help="Seconds of video to keep from before the event",
        type=float, default=3.0, required=False)
    parser.add_argument(
        '--postroll',
        help="Seconds of video to keep after the last activity",
        type=float, default=5.0, required=False)
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