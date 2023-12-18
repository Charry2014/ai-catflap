# ML stuff
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
# import utils
import cv2



class TFLiteDetect(object):


    def __init__(self, model, use_coral, num_threads):
        self.file_name=model
        self.use_coral=use_coral
        self.num_threads=num_threads

        # Initialize the object detection model
        self.base_options = core.BaseOptions(
            file_name=self.file_name, use_coral=use_coral, num_threads=num_threads)
        self.detection_options = processor.DetectionOptions(
            max_results=3, score_threshold=0.3)
        self.options = vision.ObjectDetectorOptions(
            base_options=self.base_options, detection_options=self.detection_options)
        self.detector = vision.ObjectDetector.create_from_options(self.options)

        
    def detect(self, image):
        '''Returns an array of tuples of label, probability score
        ['label', 0.55]
        '''
        # Convert the image from BGR to RGB as required by the TFLite model.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create a TensorImage object from the RGB image.
        input_tensor = vision.TensorImage.create_from_array(rgb_image)

        # Run object detection estimation using the model.
        detection_result = self.detector.detect(input_tensor)

        return detection_result



