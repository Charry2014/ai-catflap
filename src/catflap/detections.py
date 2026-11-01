'''Plain data types for detection results, and drawing them.
Kept free of tflite imports so they can be used (and tested) anywhere.'''
import cv2 as cv


class BoundingRect():
    def __init__(self, origin_x: int, origin_y: int, width: int, height: int) -> None:
        '''The same as used by tflite but defined here for ease of use'''
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.width = width
        self.height = height

    def __str__(self) -> str:
        return f"BoundingRect(origin_x={self.origin_x}, origin_y={self.origin_y}, width={self.width}, height={self.height})"


class TFLDetection():
    def __init__(self, label: str, index: int, score: float, box: BoundingRect) -> None:
        self.label = label
        self.index = index
        self.score = score
        self.box = box

    def __iter__(self):
        yield self.label
        yield self.score
        yield self.box

    def __str__(self) -> str:
        return f"{self.index}-{self.label}({self.score:.2f})"

    @property
    def detail(self) -> str:
        return f"{self} {self.score:.2f} {self.box}"


BOX_COLOUR = (255, 55, 0)


def draw_detections(frame, detections):
    '''Draw boxes and labels on frame, in place, and return it. Pass a copy
    if the original must stay untouched.'''
    for d in detections:
        b = d.box
        cv.rectangle(frame, (b.origin_x, b.origin_y),
                     (b.origin_x + b.width, b.origin_y + b.height), BOX_COLOUR, 2)
        cv.putText(frame, f"{d.label} {d.score:.2f}", (b.origin_x, b.origin_y),
                   cv.FONT_HERSHEY_PLAIN, 1, BOX_COLOUR, 1)
    return frame
