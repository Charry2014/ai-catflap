'''The camera thread: the only code that talks to an image source.'''
import threading
import time

from base_logger import logger
from framebus import FrameBus


class CaptureThread(threading.Thread):
    '''Pulls images from an AbstractImageSource and publishes them on a FrameBus.

    Live sources (camera) deliver at their own cadence, so get_image() blocks
    for the right time. File sources return instantly, so when the source
    reports `fps` we pace them to real time, as if they were a live camera.
    '''

    def __init__(self, source, bus: FrameBus) -> None:
        super().__init__(name="capture", daemon=True)
        self._source = source
        self._bus = bus
        self._halt = threading.Event()

    def stop(self) -> None:
        self._halt.set()

    def run(self) -> None:
        period = None
        fps = getattr(self._source, 'fps', None)
        if fps:
            period = 1.0 / fps
        next_due = time.monotonic()
        try:
            while not self._halt.is_set() and self._source.isopen:
                image = self._source.get_image()
                if image is None:
                    logger.info("Image source finished")
                    break
                self._bus.publish(image)
                if period is not None:
                    next_due += period
                    delay = next_due - time.monotonic()
                    if delay > 0:
                        self._halt.wait(delay)
                    else:
                        next_due = time.monotonic()  # fell behind, don't try to catch up
        except Exception:
            logger.exception("Capture thread failed")
        finally:
            self._bus.close()
