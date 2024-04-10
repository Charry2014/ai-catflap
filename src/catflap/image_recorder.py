from datetime import datetime
from os import path
from array import array
import cv2 as cv


def image_recorder(out_path: str) -> callable:

    record_path = out_path

    # check path exists and is writeable
    if not path.exists(record_path):
        # try to create the directory
        try:
            path.mkdir(record_path)
        except OSError:
            raise ValueError(f"Path {record_path} does not exist and could not be created")
    if not path.isdir(record_path):
        raise ValueError(f"Path {record_path} is not a directory")  


    def _make_outfile_name(record_path, label, prob):
        datestr = datetime.now().strftime("%Y%m%d-%H%M%S.%f")[:-3]
        prob = int(prob * 100)
        outfile = path.join(record_path, f"{label}-{str(prob)}-{datestr}_catcam.jpg")
        return outfile


    def record_image(image: array, event: str):
        cv.imwrite(_make_outfile_name(record_path, event, 1), image)

    return record_image


if __name__ == "__main__":
    # create a test image with opencv
    import  numpy as np   
    img = np.zeros((480, 640, 3), np.uint8)
    recorder = image_recorder('.')
    recorder(img, 'test')
