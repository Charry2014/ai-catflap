'''Chooses the object detection backend.

  litert         ai-edge-litert (maintained, any recent Python)         -> litert_detect.py
  tflite-support the TensorFlow Lite Task library (Python 3.9 only)    -> tflite_detect.py
  auto           litert if it is installed, otherwise tflite-support
'''
from base_logger import logger

BACKENDS = ('auto', 'litert', 'tflite-support')


def create_detector(backend: str, model: str, use_coral: bool, num_threads: int, labels=None):
    if backend not in BACKENDS:
        raise ValueError(f"Unknown detector {backend}, choose one of {BACKENDS}")
    if backend == 'auto':
        try:
            import ai_edge_litert  # noqa: F401
            backend = 'litert'
        except ImportError:
            try:
                import tflite_runtime  # noqa: F401
                backend = 'litert'
            except ImportError:
                backend = 'tflite-support'
    logger.info(f"Object detection with {backend}")
    if backend == 'litert':
        from litert_detect import LiteRTDetect
        return LiteRTDetect(model, use_coral, num_threads, labels)
    from tflite_detect import TFLiteDetect
    return TFLiteDetect(model, use_coral, num_threads)
