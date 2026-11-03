from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision

import cv2 as cv

from detections import BoundingRect, TFLDetection


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
