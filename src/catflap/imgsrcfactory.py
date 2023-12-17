from base_logger import logger
import os
HOST_OS = os.uname().sysname

from imgsrcmp4 import ImageSourceMP4Video
from imgsrcimg import ImageSourceSingleImage
if HOST_OS == "Darwin":
    # ie. macOS
    pass
else:
    # ie. Raspberry Pi
    from imgsrcpicam import ImageSourcePiCamera



class ImageSourceFactory():
    # Build the list of image sources, simple & hacky but sufficient
    __classlist__ = [ImageSourceMP4Video, ImageSourceSingleImage]
    if HOST_OS == "Darwin": pass
    else:
        # On the Pi add the Pi Camera class
        __classlist__.append(ImageSourcePiCamera)

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



import cv2 as cv
# data = "~/projects/catflap/data/incoming/Cat-with-mouse/20230310/20221016-014509_catcam.mp4"
# '~/projects/catflap/data/incoming/Cat-with-mouse/20221010/Video/20220920-012720_catcam.mp4'
data = "~/projects/catflap/data/incoming/Cat-with-mouse/20221010/Training/vlcsnap-2022-10-10-14h19m49s015.png"
def main():
    logger.info("Hello world")

    test = ImageSourceFactory.create_source(source=data)
    test.open_image_source()
    while True:
        img = test.get_image()
        if type(img) == type(None):
            test.close_image_source()
            break
        cv.imshow("image", img)
        cv.waitKey(50)

    print("Done")



if __name__ == "__main__":
    main()