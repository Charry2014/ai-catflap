from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision

import cv2 as cv


from datetime import datetime
from os import path


# Visualization parameters
row_size = 20  # pixels
left_margin = 24  # pixels
text_color = (255, 55, 0)
font_size = 1
font_thickness = 1




def make_outfile_name(record_path, label, prob=1):
    datestr = datetime.now().strftime("%Y%m%d-%H%M%S.%f")[:-3]
    prob = int(prob * 100)
    outfile = path.join(record_path, f"{label}-{str(prob)}-{datestr}_catcam.jpg")
    return outfile


class BoundingRect():
    def __init__(self, origin_x: int, origin_y:int, width:int, height:int ) -> None:
        '''The same as used by tflite but defined here for ease of use'''
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.width = width
        self.height = height

    def __str__(self) -> str:
        return f"BoundingRect(origin_x={self.origin_x}, origin_y={self.origin_y}, width={self.width}, height={self.height})"

class TFLDetection():
    def __init__(self, label:str, index: int, score:float, box:BoundingRect) -> None:
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
    def detail(self) ->str:
        return f"{self} {self.score:.2f} {self.box}"

class TFLiteDetect(object):


    def __init__(self, model, use_coral, num_threads):
        self.file_name=model
        self.use_coral=use_coral
        self.num_threads=num_threads

        # Initialize the object detection model
        self.base_options = core.BaseOptions(
            file_name=self.file_name, use_coral=use_coral, num_threads=num_threads)
        self.detection_options = processor.DetectionOptions(
            max_results=2, score_threshold=0.3)
        self.options = vision.ObjectDetectorOptions(
            base_options=self.base_options, detection_options=self.detection_options)
        self.detector = vision.ObjectDetector.create_from_options(self.options)

        self._last_result = []

    def __detection(self, image):
        '''Returns an array of tuples of label, probability score
        ['label', 0.55]
        '''
        # Convert the image from BGR to RGB as required by the TFLite model.
        rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # Create a TensorImage object from the RGB image.
        input_tensor = vision.TensorImage.create_from_array(rgb_image)

        # Run object detection estimation using the model.
        self._last_result = self.detector.detect(input_tensor).detections
        return self._last_result


    def __len__(self):
        return len(self._last_result)

    def detect(self, image) -> TFLDetection:
        self.__detection(image)
        for d in self._last_result:
            rect = BoundingRect(d.bounding_box.origin_x, d.bounding_box.origin_y, 
                                        d.bounding_box.width, d.bounding_box.height )
            for c in d.categories:
                yield TFLDetection(c.category_name, c.index, c.score, rect)


    def create_overlays(self, frame):
        for d in self._last_result:
            bounds = d.bounding_box
            for c in d.categories:
                name, score = c.category_name, c.score
                cv.rectangle(frame, (bounds.origin_x, bounds.origin_y), 
                            (bounds.origin_x+bounds.width, bounds.origin_y+bounds.height), (255, 55, 0), 2)
                text_location = (bounds.origin_x, bounds.origin_y)
                text = f"{name} {score:.2f}"
                cv.putText(frame, text, text_location, cv.FONT_HERSHEY_PLAIN,
                            font_size, text_color, font_thickness)
        return frame
