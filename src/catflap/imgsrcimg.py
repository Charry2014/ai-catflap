from abstractimagesource import AbstractImageSource
from array import array
import cv2 as cv
import os

from base_logger import logger


class ImageSourceSingleImage(AbstractImageSource):
    """Read a single image from the filesystem"""
    def __init__(self, **kwargs):
        super(ImageSourceSingleImage, self).__init__(**kwargs)

    @staticmethod
    def can_supply_images(source: str) -> bool:
        '''Returns True if this class can supply images from the source'''
        retval = False
        if source.endswith(".jpg") or source.endswith(".jpeg") or source.endswith(".png"):
            retval = True
        return retval


    def open(self) -> int:
        '''Opens an image source from the provided arguments 
        Returns 0 for success, or anything else for an error'''
        if os.path.isfile(self.source) == False:
            logger.error(f"File {self.source} does not exist")
            return 1
        self._isopen = True
        return 0

    def close(self) -> int:
        '''Closes an image source
        Returns 0 for success, or anything else for an error'''
        self._isopen = False
        return 0

    def get_image(self) -> array:
        '''Gets the next image from an image source
        Returns the image, or None when there are no more'''
        frame = None
        if self._isopen == True:
            frame = cv.imread(self.source)
            self._isopen = False
        return frame
    


data = "./projects/catflap/data/incoming/Cat-alone/vlcsnap-2022-10-10-17h07m40s527.png"

def main():
    logger.info("Hello world")
    test = ImageSourceSingleImage(source=data)

    test.open()
    while True:
        img = test.get_image()
        if type(img) == type(None):
            test.close()
            break
        cv.imshow("image", img)
        cv.waitKey(50)

if __name__ == "__main__":
    main()