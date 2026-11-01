from datetime import datetime
from os import makedirs, path

import cv2 as cv
import numpy as np

from backgroundworker import BackgroundWorker


class ImageRecorder(object):
    '''Saves still images. Calling it only queues the image; the JPEG encode and
    disk write happen on a background thread, so the main loop is never stalled.
    The file name is fixed when the image is submitted, not when it is written.'''

    def __init__(self, out_path: str, queue_size: int = 64) -> None:
        try:
            makedirs(out_path, exist_ok=True)
        except OSError:
            raise ValueError(f"Path {out_path} does not exist and could not be created")
        if not path.isdir(out_path):
            raise ValueError(f"Path {out_path} is not a directory")
        self._path = out_path
        # Order matters for evidence, so when overloaded drop the newest, not the oldest
        self._worker = BackgroundWorker(self._write, "image-recorder",
                                        maxsize=queue_size, drop_oldest=False)

    @staticmethod
    def make_outfile_name(record_path: str, label: str, prob: float = 1, when: datetime = None) -> str:
        when = when or datetime.now()
        datestr = when.strftime("%Y%m%d-%H%M%S.%f")[:-3]
        return path.join(record_path, f"{label}-{int(prob * 100)}-{datestr}_catcam.jpg")

    def __call__(self, image: np.ndarray, event: str) -> None:
        outfile = self.make_outfile_name(self._path, event)
        self._worker.submit((outfile, image))

    def _write(self, item) -> None:
        outfile, image = item
        cv.imwrite(outfile, image)

    def close(self) -> None:
        self._worker.close()


def image_recorder(out_path: str) -> ImageRecorder:
    return ImageRecorder(out_path)


if __name__ == "__main__":
    img = np.zeros((480, 640, 3), np.uint8)
    recorder = image_recorder('.')
    recorder(img, 'test')
    recorder.close()
