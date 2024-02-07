from abstractimagesource import AbstractImageSource
from array import array
import cv2

from base_logger import logger

class ImageSourceWebCam(AbstractImageSource):
    def __init__(self, **kwargs):
        super(ImageSourceWebCam, self).__init__(**kwargs)

        frameWidth = 640
        frameHeight = 480
        self.cap = cv2.VideoCapture(int(kwargs["source"]))
        self.cap.set(3, frameWidth)
        self.cap.set(4, frameHeight)
        # self.cap.set(10,150)

    @staticmethod
    def can_supply_images(source: str) -> bool:
        '''Returns True if this class can supply images from the source'''
        retval = False
        if source.isdigit() == True:
            retval = True
        return retval

    def open_image_source(self) -> int:
        '''Opens an image source from the provided arguments 
        Returns 0 for success, or anything else for an error'''
        self._isopen = True
        return 0

    def close_image_source(self) -> int:
        '''Closes an image source
        Returns 0 for success, or anything else for an error'''
        self._isopen = False
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