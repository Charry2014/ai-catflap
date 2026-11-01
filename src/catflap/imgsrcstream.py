from abstractimagesource import AbstractImageSource
from array import array
import cv2
import time

from base_logger import logger


class ImageSourceStream(AbstractImageSource):
    '''A network video stream: MJPEG over http, RTSP, or raw tcp. This is how a
    container gets the Pi camera, which is served from the host.
    If the stream drops it reconnects instead of ending the program.'''
    SCHEMES = ("http://", "https://", "rtsp://", "tcp://", "udp://")
    RECONNECT_TRIES = 30       # With the delay below: give up after about a minute
    RECONNECT_DELAY = 2.0

    def __init__(self, **kwargs):
        super(ImageSourceStream, self).__init__(**kwargs)
        self.cap = None

    @staticmethod
    def can_supply_images(source: str) -> bool:
        return source.lower().startswith(ImageSourceStream.SCHEMES)

    def _connect(self) -> bool:
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.source)
        return self.cap.isOpened()

    def open(self) -> int:
        '''Returns 0 for success, or anything else for an error'''
        for _ in range(self.RECONNECT_TRIES):
            if self._connect():
                self._isopen = True
                return 0
            logger.warning(f"Cannot open stream {self.source}, retrying")
            time.sleep(self.RECONNECT_DELAY)
        logger.error(f"Cannot open stream {self.source}")
        return 1

    def close(self) -> int:
        self._isopen = False
        if self.cap is not None:
            self.cap.release()
        return 0

    def get_image(self) -> array:
        '''Returns the next image, or None when the stream is gone for good'''
        for _ in range(self.RECONNECT_TRIES):
            success, frame = self.cap.read()
            if success and frame is not None:
                return frame
            if not self._isopen:
                return None
            logger.warning("Stream interrupted, reconnecting")
            time.sleep(self.RECONNECT_DELAY)
            self._connect()
        logger.error("Stream lost")
        self._isopen = False
        return None
