from abstractimagesource import AbstractImageSource
from array import array
import cv2 as cv
import os
from base_logger import logger


class ImageSourceMP4Video(AbstractImageSource):
    def __init__(self, **kwargs):
        super(ImageSourceMP4Video, self).__init__(**kwargs)
        self.cap = cv.VideoCapture(self.source)

    @staticmethod
    def can_supply_images(source: str) -> bool:
        '''Returns True if this class can supply images from the source'''
        retval = False
        if source.endswith(".mp4"):
            retval = True
        return retval


    def open(self) -> int:
        '''Opens an image source from the provided arguments 
        Returns 0 for success, or anything else for an error'''
        if os.path.isfile(self.source) == False:
            logger.error(f"File {self.source} does not exist")
            return 1
        if self.cap.isOpened() == False:
            print(f"Could not open capture device. Check your video file {self.source}")
            return 1
        self.size = (int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH)),
            int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
        self._isopen = True
        return 0

    def close(self) -> int:
        '''Closes an image source
        Returns 0 for success, or anything else for an error'''
        self.cap.release()
        self._isopen = False
        return 0

    def get_image(self) -> array:
        '''Gets the next image from an image source
        Returns the image, or None when there are no more'''
        _, frame = self.cap.read()
        if type(frame) == type(None):
            self._isopen = False
        return frame
    


data = "./projects/catflap/data/incoming/Cat-with-mouse/20230310/20221016-014509_catcam.mp4"
# './projects/catflap/data/incoming/Cat-with-mouse/20221010/Video/20230224-040121_catcam.mp4'

def main():
    logger.info("Hello world")
    test = ImageSourceMP4Video(source=data)

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