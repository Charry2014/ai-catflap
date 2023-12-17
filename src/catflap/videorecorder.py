# import sys
# print(sys.path)

from datetime import datetime, timedelta
import cv2 as cv
import numpy as np
from os import path
from base_logger import logger



class VideoRecorder(object):

    def __init__(self, size, recordpath: str = "./", duration: int = 10):
        if(not path.isdir(recordpath)):
            import sys
            logger.error(f"Cannot record video to {recordpath} - not a directory")
            sys.exit(1)
        logger.debug(f"Output directory set to {recordpath}")
        self.recordpath = recordpath
        self.duration = duration
        self._recording = False
        self.end_time = None
        self.video_writer = None
        self._size = size
        self._fps = 15.0

        # Define the codec and create VideoWriter object
        # self.fourcc = cv.VideoWriter_fourcc(*'mp4v') # mp4
        self.fourcc = cv.VideoWriter_fourcc(*'XVID') # XVID.mp4/avi   X264,

    @property
    def recording(self) -> bool:
        return self._recording

    def start_recording(self, frame:np.array):
        if(self.recording == False):
            self._start_recording()
        self.record(frame)


    def _start_recording(self):
        self._recording = True
        datestr = datetime.now().strftime("%Y%m%d-%H%M%S")
        outfile = path.join(self.recordpath, f"{datestr}_catcam.mp4")
        self.video_writer = cv.VideoWriter(outfile, self.fourcc, self._fps, self._size)
        self.end_time = datetime.now() + timedelta(seconds=self.duration)
        logger.info(f"Recording started to {outfile} - ends at {self.end_time}")

    def record(self, frame: np.ndarray) -> bool:
        retval = True
        self.video_writer.write(frame)
        if(datetime.now() > self.end_time):
            logger.info("Recording done")
            self.video_writer.release()
            self._recording = False
            self.video_writer = None
            retval = False
        return retval

