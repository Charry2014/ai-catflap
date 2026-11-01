from base_logger import logger

from imgsrcmp4 import ImageSourceMP4Video
from imgsrcimg import ImageSourceSingleImage
from imgsrcstream import ImageSourceStream
from imgsrcwebcam import ImageSourceWebCam

# The Pi camera needs picamera2 and libcamera, which only exist on a Pi
# (and are not in the container). Without them a numeric source is a
# webcam, which is what a desktop has.
try:
    from imgsrcpicam import ImageSourcePiCamera
except ImportError:
    ImageSourcePiCamera = None
    logger.info("picamera2 not available, camera numbers will open a webcam")


class ImageSourceFactory():
    # Build the list of image sources, simple & hacky but sufficient
    __classlist__ = [ImageSourceMP4Video, ImageSourceSingleImage, ImageSourceStream]
    __classlist__.append(ImageSourcePiCamera if ImageSourcePiCamera is not None else ImageSourceWebCam)

    def __init__(self) -> None:
        pass

    def __iter__(self):
        for x in ImageSourceFactory.__classlist__:
            yield x

    @staticmethod
    def create_source(source: str):
        retval = None
        for c in ImageSourceFactory.__classlist__:
            if c.can_supply_images(source) == True:
                retval = c(source=source)
                break
        return retval
