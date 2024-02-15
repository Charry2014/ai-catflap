from abstractimagesource import AbstractImageSource
from array import array
import cv2
import time
import sys

from base_logger import logger

class ImageSourceWebCam(AbstractImageSource):
    def __init__(self, **kwargs):
        super(ImageSourceWebCam, self).__init__(**kwargs)

        frameWidth = 640
        frameHeight = 480

        success = False
        max_retries = 10
        while success == False:
            self.cap = cv2.VideoCapture(int(kwargs["source"]))
            self.cap.set(3, frameWidth)
            self.cap.set(4, frameHeight)
            success, frame = self.cap.read()
            if type(frame) == type(None) or success == False:
                max_retries -= 1
                if max_retries == 0:
                    logger.error('Failed to open web cam image source.')
                    sys.exit(1)
                self.cap.release()
                time.sleep(0.2)

        # self.cap.set(10,150)

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
        self._isopen = True
        return 0

    def close(self) -> int:
        '''Closes an image source
        Returns 0 for success, or anything else for an error'''
        self._isopen = False
        self.cap.release()
        return 0

    def get_image(self) -> array:
        '''Gets the next image from an image source
        Returns the image, or None when there are no more'''
        success, frame = self.cap.read()
        if type(frame) == type(None) or success == False:
            frame = None
        return frame




def main():
    logger.info("Hello world")
    test = ImageSourceWebCam()
    pass

if __name__ == "__main__":
    main()