'''Object detection with ai-edge-litert (the maintained successor of the TensorFlow
Lite runtime), as a replacement for tflite_detect.py which needs tflite-support
and with it Python 3.9.

It runs the same .tflite files, including the ones tflite-model-maker exports, so
models and training stay as they are. What tflite-support's ObjectDetector did for
us, this does itself:
  - resize the image to the model's input size and feed it as the model expects
  - read the four output tensors of the model's built in non-max-suppression
  - turn class numbers into names with the labelmap stored inside the model
  - drop everything under the score threshold and keep the best few
'''
import zipfile

import cv2 as cv
import numpy as np

from detections import BoundingRect, TFLDetection

try:
    from ai_edge_litert.interpreter import Interpreter, load_delegate
except ImportError:                     # Older name of the same thing
    from tflite_runtime.interpreter import Interpreter, load_delegate


def read_labelmap(model_file: str) -> list:
    '''Class names stored in the model file. Models from tflite-model-maker carry a
    labelmap.txt, packed into the file as a zip appended to the flatbuffer.'''
    try:
        with zipfile.ZipFile(model_file) as z:
            for name in z.namelist():
                if name.endswith('.txt'):
                    return [l.strip() for l in z.read(name).decode('utf-8').splitlines() if l.strip()]
    except zipfile.BadZipFile:
        pass
    return []


class LiteRTDetect(object):
    # Same values tflite_detect.py gives the Task library
    MAX_RESULTS = 2
    SCORE_THRESHOLD = 0.3

    def __init__(self, model, use_coral=False, num_threads=4, labels=None):
        self.file_name = model
        self.labels = read_labelmap(model) or list(labels or [])
        delegates = [load_delegate('libedgetpu.so.1')] if use_coral else None
        self._interpreter = Interpreter(model_path=model, num_threads=num_threads,
                                        experimental_delegates=delegates)
        self._interpreter.allocate_tensors()
        inp = self._interpreter.get_input_details()[0]
        self._input_index = inp['index']
        self._input_dtype = inp['dtype']
        _, self._height, self._width, _ = inp['shape']
        self._find_outputs()
        self._last_result = []

    def _find_outputs(self):
        '''Tell the outputs apart by shape: boxes are [1,N,4], the count is [1], and there
        are two [1,N] tensors - classes and scores - which are told apart by their values.'''
        self._boxes = self._count = None
        self._pair = []
        for d in self._interpreter.get_output_details():
            shape = tuple(d['shape'])
            if len(shape) == 3 and shape[-1] == 4:
                self._boxes = d['index']
            elif len(shape) == 1:
                self._count = d['index']
            elif len(shape) == 2:
                self._pair.append(d['index'])
        if self._boxes is None or len(self._pair) != 2:
            raise ValueError(f"{self.file_name} does not look like an object detection model with "
                             f"built in post processing (boxes, classes, scores, count)")
        # Default for models from tflite-model-maker; confirmed from the values on every frame
        self._scores, self._classes = self._pair

    def _preprocess(self, image):
        rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        # Averaging when shrinking, like the Task library, avoids aliasing
        shrinking = image.shape[0] > self._height or image.shape[1] > self._width
        rgb = cv.resize(rgb, (self._width, self._height),
                        interpolation=cv.INTER_AREA if shrinking else cv.INTER_LINEAR)
        if self._input_dtype == np.uint8:
            return rgb[np.newaxis]
        return ((rgb.astype(np.float32) - 127.5) / 127.5)[np.newaxis]

    def _assign_pair(self, a, b):
        '''Class numbers are whole numbers, scores are not (all zero tells nothing, then
        keep what we had).'''
        whole_a, whole_b = bool(np.all(a == np.round(a))), bool(np.all(b == np.round(b)))
        if whole_a and not whole_b:
            self._classes, self._scores = self._pair
        elif whole_b and not whole_a:
            self._scores, self._classes = self._pair

    def _detection(self, image):
        height, width = image.shape[:2]
        it = self._interpreter
        it.set_tensor(self._input_index, self._preprocess(image))
        it.invoke()
        a, b = (it.get_tensor(i)[0] for i in self._pair)
        self._assign_pair(a, b)
        scores = it.get_tensor(self._scores)[0]
        classes = it.get_tensor(self._classes)[0].astype(int)
        boxes = it.get_tensor(self._boxes)[0]       # ymin, xmin, ymax, xmax, 0..1
        found = []
        for i in np.argsort(-scores):
            if scores[i] < self.SCORE_THRESHOLD or len(found) >= self.MAX_RESULTS:
                break
            ymin, xmin, ymax, xmax = boxes[i]
            rect = BoundingRect(int(round(xmin * width)), int(round(ymin * height)),
                                int(round((xmax - xmin) * width)), int(round((ymax - ymin) * height)))
            index = int(classes[i])
            label = self.labels[index] if 0 <= index < len(self.labels) else str(index)
            found.append(TFLDetection(label, index, float(scores[i]), rect))
        self._last_result = found
        return found

    def __len__(self):
        return len(self._last_result)

    def detect(self, image):
        '''Yields a TFLDetection for the best detections, best first. image is BGR, like OpenCV'''
        yield from self._detection(image)
