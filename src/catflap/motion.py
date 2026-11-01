'''Movement detection in the trigger area, used while the flap is idle.'''
from collections import deque, namedtuple

import cv2 as cv

# Position and size in full-frame coordinates, and the contour area
MotionBox = namedtuple('MotionBox', 'x y w h area')


class MotionDetector:
    # Each frame is compared with the newest frame at least this old (seconds).
    # Frames arrive continuously, but slow movement still needs a gap between
    # the two images to show as a difference.
    REFERENCE_AGE = 1.0

    def __init__(self, left: int, top: int, right: int, bottom: int) -> None:
        self._l, self._t, self._r, self._b = left, top, right, bottom
        self._history = deque()

    def reset(self) -> None:
        '''Forget earlier frames, e.g. after the camera was not being watched'''
        self._history.clear()

    def process(self, ts: float, image) -> list:
        crop = image[self._t:self._b, self._l:self._r].copy()
        history = self._history
        history.append((ts, crop))
        while len(history) > 1 and history[1][0] <= ts - self.REFERENCE_AGE:
            history.popleft()
        reference = history[0][1]

        diff = cv.absdiff(reference, crop)
        diff_gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(diff_gray, (5, 5), 0)
        _, thresh = cv.threshold(blur, 20, 255, cv.THRESH_BINARY)
        dilated = cv.dilate(thresh, None, iterations=3)
        contours, _ = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours:
            x, y, w, h = cv.boundingRect(c)
            boxes.append(MotionBox(x + self._l, y + self._t, w, h, cv.contourArea(c)))
        return boxes
