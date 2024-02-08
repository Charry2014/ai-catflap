from abstractimagesource import AbstractImageSource
from array import array

from picamera2 import Picamera2
import libcamera

from base_logger import logger

Picamera2.set_logging(Picamera2.INFO)

class ImageSourcePiCamera(AbstractImageSource):
    def __init__(self, **kwargs):
        super(ImageSourcePiCamera, self).__init__(**kwargs)

        self.picam2 = Picamera2()
        config = self.picam2.create_still_configuration(main={"size": (640, 480), "format": "RGB888"}, 
                                            transform=libcamera.Transform(hflip=1, vflip=1))
        self.picam2.configure(config)
        self.picam2.still_configuration.align()

    @staticmethod
    def can_supply_images(source: str) -> bool:
        '''Returns True if this class can supply images from the source'''
        retval = False
        if source.isdigit() == True:
            retval = True
        return retval

    def open(self) -> int:
        '''Opens an image source from the provided arguments 
        Returns 0 for success, or anything else for an error'''
        self.picam2.start()
        self._isopen = True
        return 0

    def close(self) -> int:
        '''Closes an image source
        Returns 0 for success, or anything else for an error'''
        self.picam2.close()
        self._isopen = False
        return 0

    def get_image(self) -> array:
        '''Gets the next image from an image source
        Returns the image, or None when there are no more'''
        frame = self.picam2.capture_array()
        if type(frame) == type(None):
            frame = None
        return frame




def main():
    logger.info("Hello world")
    test = ImageSourcePiCamera()
    pass

if __name__ == "__main__":
    main()